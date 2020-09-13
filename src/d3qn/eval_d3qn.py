import random

import numpy as np
import torch

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from src.common.action_skipping_masking import get_action_masking
from src.common.flatland_railenv import FlatlandRailEnv
from src.d3qn.d3qn_flatland import add_fingerprints
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
    if train_params.fingerprints:
        # With fingerprints
        policy = D3QNPolicy(env.state_size + 2, action_size, train_params)
    else:
        # Without fingerprints
        policy = D3QNPolicy(env.state_size, action_size, train_params)

    ####################################################################################################################

    print("\nEvaluating {} trains on {}x{} grid for {} episodes.\n"
          .format(env_params.n_agents, env_params.x_dim, env_params.y_dim, train_params.eval_episodes))

    agent_prev_obs = [None] * env_params.n_agents
    agent_prev_action = [2] * env_params.n_agents
    timestep = 0
    eps_start = 0

    for episode in range(train_params.eval_episodes):

        # Reset environment
        obs, info = env.reset()

        if train_params.fingerprints:
            obs = add_fingerprints(obs, env_params.n_agents, eps_start, timestep)

        # Build agent specific observations
        for agent in range(env_params.n_agents):
            if obs[agent] is not None:
                agent_prev_obs[agent] = obs[agent].copy()

        for step in range(max_steps):
            # Action dictionary to feed to step
            action_dict = dict()

            # Set used to track agents that didn't skipped the action
            agents_in_action = set()

            for agent in range(env_params.n_agents):
                # Create action mask
                action_mask = get_action_masking(env, agent, action_size, train_params)

                # Fill action dict
                # If agent is not arrived, moving between two cells or trapped in a deadlock (the latter is caught only
                # when the agent is moving in the deadlock triggering the second case)
                if info["action_required"][agent]:
                    # If an action is required, the actor predicts an action
                    agents_in_action.add(agent)
                    action_dict[agent] = policy.act(obs[agent], action_mask=action_mask, eps=eps_start)

            # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)

            if train_params.fingerprints:
                next_obs = add_fingerprints(next_obs, env_params.n_agents, eps_start, timestep)

            for agent in range(env_params.n_agents):
                if next_obs[agent] is not None:
                    obs[agent] = next_obs[agent]

            if train_params.render:
                env.env.show_render()

            if done['__all__']:
                break

        # Rendering
        if train_params.render:
            env.env.close()
