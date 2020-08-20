import random
from argparse import Namespace

import numpy as np
import torch
from flatland.envs.malfunction_generators import MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from torch.utils.tensorboard import SummaryWriter

from src.common.flatland_random_railenv import FlatlandRandomRailEnv
from src.common.timer import Timer
from src.d3qn.policy import D3QNPolicy


def train_multiple_agents(env_params, train_params):
    # Environment parameters
    seed = env_params.seed

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_max_path_depth = env_params.observation_max_path_depth

    # Training parameters
    eps_start = train_params.eps_start
    eps_end = train_params.eps_end
    eps_decay = train_params.eps_decay

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environment
    env = FlatlandRandomRailEnv(train_params, env_params, tree_observation, reward_wrapper=False, stats_wrapper=False)
    env.reset()

    # The action space of flatland is 5 discrete actions
    action_size = env.get_rail_env().action_space[0]

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    max_steps = int(4 * 2 * (env_params.y_dim + env_params.x_dim + (env_params.n_agents / env_params.n_cities)))

    # Double Dueling DQN policy
    policy = D3QNPolicy(env.state_size - 1, action_size, train_params)

    # TensorBoard writer
    writer = SummaryWriter(train_params.tensorboard_path)
    writer.add_hparams(vars(train_params), {})
    # Remove attributes not printable by Tensorboard
    board_env_params = vars(env_params)
    del board_env_params["speed_profiles"]
    del board_env_params["malfunction_parameters"]
    writer.add_hparams(board_env_params, {})

    ####################################################################################################################
    # Training starts
    training_timer = Timer()
    training_timer.start()

    print("\nTraining {} trains on {}x{} grid for {} episodes.\n"
          .format(env_params.n_agents, env_params.x_dim, env_params.y_dim, train_params.n_episodes))

    # Variables to compute statistics
    action_count = [0] * action_size
    action_dict = dict()

    agent_prev_obs = [None] * env_params.n_agents
    agent_prev_action = [2] * env_params.n_agents
    update_values = False

    for episode in range(train_params.n_episodes + 1):
        # Timers
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()

        # Reset environment
        reset_timer.start()
        obs, info = env.reset()
        reset_timer.end()

        actions_taken = []

        # Build agent specific observations
        for agent in range(env_params.n_agents):
            if obs[agent] is not None:
                agent_prev_obs[agent] = obs[agent].copy()

        # Run episode
        for step in range(max_steps - 1):
            for agent in range(env_params.n_agents):
                if info['action_required'][agent]:
                    # If an action is required, we want to store the obs at that step as well as the action
                    # TODO: Update values outside?
                    update_values = True
                    action = policy.act(obs[agent], eps=eps_start)
                    action_count[action] += 1
                    actions_taken.append(action)
                else:
                    update_values = True
                    action = 0
                action_dict.update({agent: action})

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = env.step(action_dict)
            step_timer.end()

            for agent in range(env_params.n_agents):
                """
                Update replay buffer and train agent. Only update the values when we are done or when an action was 
                taken and thus relevant information is present
                """
                if update_values or done[agent]:
                    learn_timer.start()
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], obs[agent],
                                done[agent])
                    learn_timer.end()

                    agent_prev_obs[agent] = obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                if next_obs[agent] is not None:
                    obs[agent] = next_obs[agent]

            if train_params.render:
                env._env.show_render()

            if done['__all__']:
                break

        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Save checkpoints
        if train_params.checkpoint_interval is not None and episode % train_params.checkpoint_interval == 0:
            if train_params.save_model_path is not None:
                policy.save(train_params.save_model_path)
        # Rendering
        if train_params.render:
            env._env.close()


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

        # ============================
        # Custom observations&rewards
        # ============================
        "custom_observations": True,

        "stop_penalty": 0.0,
        "invalid_action_penalty": 0.0,
        "deadlock_penalty": 0.0,
        "shortest_path_penalty_coefficient": 1.0,
        # 1.0 for skipping
        "done_bonus": 0.0,
    }

    training_parameters = {
        # ============================
        # Network architecture
        # ============================
        "double_dqn": True,
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
        # Minimum number of samples to start learning
        "buffer_min_size": 0,

        # ============================
        # Memory
        # ============================
        # Memory maximum size
        "buffer_size": int(1e6),
        # memory type uer or per
        "memory_type": "per",


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
