from flatland.envs.agent_utils import RailAgentStatus


import random
from argparse import Namespace

from flatland.utils.rendertools import RenderTool
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

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
    # Evaluation statics
    accumulated_eval_normalized_score = []
    accumulated_eval_completion = []
    # accumulated_eval_deads = []

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
        nb_steps = 0
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

            nb_steps = step

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

        # TODO: Evaluate
        """
        # Evaluate policy
        if episode % train_params.checkpoint_interval == 0:
            scores, completions, nb_steps_eval = eval_policy(env, policy, train_params.eval_episodes, max_steps)
            writer.add_scalar("evaluation/scores_min", np.min(scores), episode)
            writer.add_scalar("evaluation/scores_max", np.max(scores), episode)
            writer.add_scalar("evaluation/scores_mean", np.mean(scores), episode)
            writer.add_scalar("evaluation/scores_std", np.std(scores), episode)
            writer.add_histogram("evaluation/scores", np.array(scores), episode)
            writer.add_scalar("evaluation/completions_min", np.min(completions), episode)
            writer.add_scalar("evaluation/completions_max", np.max(completions), episode)
            writer.add_scalar("evaluation/completions_mean", np.mean(completions), episode)
            writer.add_scalar("evaluation/completions_std", np.std(completions), episode)
            writer.add_histogram("evaluation/completions", np.array(completions), episode)
            writer.add_scalar("evaluation/nb_steps_min", np.min(nb_steps_eval), episode)
            writer.add_scalar("evaluation/nb_steps_max", np.max(nb_steps_eval), episode)
            writer.add_scalar("evaluation/nb_steps_mean", np.mean(nb_steps_eval), episode)
            writer.add_scalar("evaluation/nb_steps_std", np.std(nb_steps_eval), episode)
            writer.add_histogram("evaluation/nb_steps", np.array(nb_steps_eval), episode)

            smoothing = 0.9
            smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(scores) * (
                        1.0 - smoothing)
            smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)
            writer.add_scalar("evaluation/smoothed_score", smoothed_eval_normalized_score, episode)
            writer.add_scalar("evaluation/smoothed_completion", smoothed_eval_completion, episode)

        # Save logs to tensorboard
        writer.add_scalar("training/score", normalized_score, episode)
        writer.add_scalar("training/smoothed_score", smoothed_normalized_score, episode)
        writer.add_scalar("training/completion", np.mean(completion), episode)
        writer.add_scalar("training/smoothed_completion", np.mean(smoothed_completion), episode)
        writer.add_scalar("training/nb_steps", nb_steps, episode)
        writer.add_histogram("actions/distribution", np.array(actions_taken), episode)
        writer.add_scalar("actions/nothing", action_probs[RailEnvActions.DO_NOTHING], episode)
        writer.add_scalar("actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode)
        writer.add_scalar("actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode)
        writer.add_scalar("actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode)
        writer.add_scalar("actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode)
        writer.add_scalar("training/epsilon", eps_start, episode)
        writer.add_scalar("training/buffer_size", len(policy.memory), episode)
        writer.add_scalar("training/loss", policy.loss, episode)
        writer.add_scalar("timer/reset", reset_timer.get(), episode)
        writer.add_scalar("timer/step", step_timer.get(), episode)
        writer.add_scalar("timer/learn", learn_timer.get(), episode)
        writer.add_scalar("timer/preproc", preproc_timer.get(), episode)
        writer.add_scalar("timer/total", training_timer.get_current(), episode)
        """


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


"""
def eval_policy(env, policy, train_params, env_params, n_eval_episodes, max_steps):
    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []

    for episode in range(n_eval_episodes):
        agent_obs = [None] * env.get_num_agents()
        score = 0.0

        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        final_step = 0

        for step in range(max_steps - 1):
            for agent in env.get_agent_handles():
                if obs[agent]:
                    agent_obs[agent] = normalize_observation(obs[agent], env_params.observation_tree_depth,
                                                             observation_radius=env_params.observation_radius)

                action = 0
                if info['action_required'][agent]:
                    action = policy.act(agent_obs[agent], eps=0.0)
                action_dict.update({agent: action})

            obs, all_rewards, done, info = env.step(action_dict)

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

    print("\tEval: score {:.3f} done {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0))

    return scores, completions, nb_steps
"""

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
        "n_episodes": 2500,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.99,
        "buffer_size": int(1e6),
        "buffer_min_size": 0,
        "batch_size": 32,
        "gamma": 0.99,
        "tau": 1e-3,
        "learning_rate": 0.52e-4,
        "hidden_size": 256,
        "update_every": 8,

        "checkpoint_interval": 100,
        "eval_episodes": 25,
        "use_gpu": False,
        "num_threads": 1,
        "render": False,
        "save_model_path": "checkpoint.pt",
        "load_model_path": "checkpoint.pt",
        "tensorboard_path": "log/",
    }

    train_multiple_agents(Namespace(**environment_parameters), Namespace(**training_parameters))
