import random

import numpy as np
import torch
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from src.common.flatland_random_railenv import FlatlandRailEnv
from src.d3qn.policy import D3QNPolicy


def eval_policy(env_params, train_params):
    # Environment parameters
    seed = env_params.seed

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_max_path_depth = env_params.observation_max_path_depth

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environment
    env = FlatlandRailEnv(train_params,
                          env_params,
                          tree_observation,
                          env_params.custom_observations,
                          env_params.reward_shaping,
                          train_params.print_stats)
    env.reset()

    # The action space of flatland is 5 discrete actions
    action_size = env.get_rail_env().action_space[0]

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    max_steps = int(4 * 2 * (env_params.y_dim + env_params.x_dim + (env_params.n_agents / env_params.n_cities)))

    # Double Dueling DQN policy
    policy = D3QNPolicy(env.state_size, action_size, train_params)

    ####################################################################################################################

    print("\nEvaluating {} trains on {}x{} grid for {} episodes.\n"
          .format(env_params.n_agents, env_params.x_dim, env_params.y_dim, train_params.eval_episodes))

    action_dict = dict()

    for episode in range(train_params.eval_episodes):

        # Reset environment
        obs, info = env.reset()

        for step in range(max_steps):
            for agent in range(env_params.n_agents):
                if info['action_required'][agent]:
                    action = policy.act(obs[agent])
                else:
                    action = 0
                action_dict.update({agent: action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)

            if train_params.render:
                env._env.show_render()

            if done['__all__']:
                break

        # Rendering
        if train_params.render:
            env._env.close()