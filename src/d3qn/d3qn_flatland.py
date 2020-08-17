import random
from argparse import Namespace

import numpy as np
import torch
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from torch.utils.tensorboard import SummaryWriter

from src.common.observation import normalize_observation
from src.common.timer import Timer
from src.d3qn.policy import D3QNPolicy


def train_multiple_agents(env_params, train_params):
    # Environment parameters
    seed = env_params.seed

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_radius = env_params.observation_radius
    observation_max_path_depth = env_params.observation_max_path_depth

    # Training parameters
    eps_start = train_params.eps_start
    eps_end = train_params.eps_end
    eps_decay = train_params.eps_decay

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environment
    env = RailEnv(
        width=env_params.x_dim,
        height=env_params.y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=env_params.n_cities,
            grid_mode=False,
            max_rails_between_cities=env_params.max_rails_between_cities,
            max_rails_in_city=env_params.max_rails_in_city
        ),
        schedule_generator=sparse_schedule_generator(env_params.speed_profiles),
        number_of_agents=env_params.n_agents,
        malfunction_generator_and_process_data=malfunction_from_params(env_params.malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed
    )

    env.reset(regenerate_schedule=True, regenerate_rail=True)

    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])

    state_size = n_features_per_node * n_nodes

    # The action space of flatland is 5 discrete actions
    action_size = env.action_space[0]

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    max_steps = int(4 * 2 * (env.height + env.width + (env.get_num_agents() / env_params.n_cities)))

    # Double Dueling DQN policy
    policy = D3QNPolicy(state_size, action_size, train_params)

    # TensorBoard writer
    writer = SummaryWriter(train_params.tensorboard_path)
    writer.add_hparams(vars(train_params), {})
    # Remove attributes not printable by Tensorboard
    board_env_params = vars(env_params)
    del board_env_params["speed_profiles"]
    del board_env_params["malfunction_parameters"]
    writer.add_hparams(board_env_params, {})

    ####################################################################################################################
    training_timer = Timer()
    training_timer.start()

    print("\nTraining {} trains on {}x{} grid for {} episodes.\n"
          .format(env.get_num_agents(), env_params.x_dim, env_params.y_dim, train_params.n_episodes))

    action_count = [0] * action_size
    action_dict = dict()

    accumulated_normalized_score = []
    accumulated_completion = []
    # accumulated_deadlocks = []

    agent_obs = [None] * env.get_num_agents()
    agent_prev_obs = [None] * env.get_num_agents()
    agent_prev_action = [2] * env.get_num_agents()
    update_values = False

    for episode in range(train_params.n_episodes + 1):
        # Timers
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()

        # Reset environment
        reset_timer.start()
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        reset_timer.end()

        # Setup renderer
        if train_params.render:
            env_renderer = RenderTool(env, gl="PGL")
        else:
            env_renderer = None
        if train_params.render:
            env_renderer.set_new_rail()

        score = 0
        actions_taken = []

        # Build agent specific observations
        for agent in env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()

        # Run episode
        for step in range(max_steps - 1):
            for agent in env.get_agent_handles():
                if info['action_required'][agent]:
                    # If an action is required, we want to store the obs at that step as well as the action
                    # TODO: Update values outside?
                    update_values = True
                    action = policy.act(agent_obs[agent], eps=eps_start)
                    action_count[action] += 1
                    actions_taken.append(action)
                else:
                    update_values = False
                    action = 0
                action_dict.update({agent: action})

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = env.step(action_dict)
            step_timer.end()

            if train_params.render and episode % train_params.checkpoint_interval == 0:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

            for agent in range(env.get_num_agents()):
                """
                Update replay buffer and train agent. Only update the values when we are done or when an action was 
                taken and thus relevant information is present
                """
                if update_values or done[agent]:
                    learn_timer.start()
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent],
                                done[agent])
                    learn_timer.end()

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                # Preprocess the new observations
                if next_obs[agent]:
                    preproc_timer.start()
                    agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth,
                                                             observation_radius)
                    preproc_timer.end()

                score += all_rewards[agent]

            if done['__all__']:
                break

        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collection information about training
        normalized_score = score / (max_steps * env.get_num_agents())
        tasks_finished = sum(info["status"][a] in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]
                             for a in env.get_agent_handles())
        completion_percentage = tasks_finished / max(1, env.get_num_agents())
        # deadlocks_percentage = sum(deadlocks) / env.get_num_agents()
        action_probs = action_count / np.sum(action_count)
        action_count = [1] * action_size

        # Mean values for terminal display and for more stable hyper-parameter tuning
        accumulated_normalized_score.append(normalized_score)
        accumulated_completion.append(completion_percentage)
        # accumulated_deadlocks.append(deadlocks_percentage)

        # Save checkpoints
        if train_params.checkpoint_interval is not None and episode % train_params.checkpoint_interval == 0:
            if train_params.save_model_path is not None:
                policy.save(train_params.save_model_path)
        # Rendering
        if train_params.render:
            env_renderer.close_window()

        # TODO: Deadlocks
        print(
            "\rEpisode {}"
            "\tScore: {:.3f}"
            " Avg: {:.3f}"
            "\tDone: {:.2f}%"
            " Avg: {:.2f}%"
            "\tAction Probs: {}".format(
                episode,
                normalized_score,
                np.mean(accumulated_normalized_score),
                100 * completion_percentage,
                100 * np.mean(accumulated_completion),
                format_action_prob(action_probs)
            ), end=" ")


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


if __name__ == "__main__":
    myseed = 14

    environment_parameters = {
        "n_agents": 3,
        "x_dim": 16 * 3,
        "y_dim": 9 * 3,
        "n_cities": 5,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "seed": myseed,
        "observation_tree_depth": 3,
        "observation_radius": 10,
        "observation_max_path_depth": 30,
        # Malfunctions
        "malfunction_parameters": MalfunctionParameters(
            malfunction_rate=0,
            min_duration=15,
            max_duration=50),
        # Speeds
        "speed_profiles": {
            1.: 1.0,
            1. / 2.: 0.0,
            1. / 3.: 0.0,
            1. / 4.: 0.0},
    }

    training_parameters = {
        # ============================
        # Network architecture
        # ============================
        "shared": False,
        "hidden_size": 256,
        "hidden_layers": 2,
        "update_every": 8,

        # epsilon greedy decay regulators
        "eps_decay": 0.99,
        "eps_start": 1.0,
        "eps_end": 0.01,

        "learning_rate": 0.52e-4,
        # To compute q targets
        "gamma": 0.99,
        # To compute target network soft update
        "tau": 1e-3,

        # ============================
        # Training setup
        # ============================
        "n_episodes": 2500,
        "batch_size": 32,
        # Memory maximum size
        "buffer_size": int(1e6),
        # Minimum number of samples to start learning
        "buffer_min_size": 0,

        # ============================
        # Optimization and rendering
        # ============================
        "checkpoint_interval": 100,
        "eval_episodes": 25,
        "use_gpu": False,
        "render": False,
        "save_model_path": "checkpoint.pt",
        "load_model_path": "checkpoint.pt",
        "tensorboard_path": "log/",
    }

    train_multiple_agents(Namespace(**environment_parameters), Namespace(**training_parameters))
