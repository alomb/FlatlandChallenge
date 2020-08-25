import random

from flatland.envs.agent_utils import RailAgentStatus
import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
import torch
from flatland.envs.rail_env import RailEnvActions

from src.common.action_skipping_masking import find_decision_cells
from src.common.flatland_random_railenv import FlatlandRailEnv
from src.psppo.policy import PsPPOPolicy


def eval_policy(env_params, train_params):
    # Action counter used for statistics
    action_dict = dict()

    seed = env_params.seed

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_max_path_depth = env_params.observation_max_path_depth

    n_eval_episodes = train_params.eval_episodes

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

    # + 1 because the agent id is used as input in the neural network
    ppo = PsPPOPolicy(env.state_size + 1,
                      action_size,
                      train_params,
                      env_params.n_agents)

    print("\nEvaluating {} trains on {}x{} grid for {} episodes.\n"
          .format(env_params.n_agents, env_params.x_dim, env_params.y_dim, n_eval_episodes))

    for episode_idx in range(n_eval_episodes):

        # Mask initialization
        action_mask = [[1 * (0 if action == 0 and not train_params.allow_no_op else 1)
                        for action in range(action_size)] for _ in range(env_params.n_agents)]

        obs, info = env.reset()
        decision_cells = find_decision_cells(env.get_rail_env())

        for step in range(max_steps):
            for agent in range(env_params.n_agents):
                if obs[agent] is not None:
                    # Action mask modification only if action masking is True
                    if train_params.action_masking:
                        for action in range(action_size):
                            if env.get_rail_env().agents[agent].status != RailAgentStatus.READY_TO_DEPART:
                                _, cell_valid, _, _, transition_valid = env.get_rail_env()._check_action_on_agent(
                                    RailEnvActions(action),
                                    env.get_rail_env().agents[agent])
                                if not all([cell_valid, transition_valid]):
                                    action_mask[agent][action] = 0

                if train_params.action_skipping and env.get_rail_env().agents[agent].position is not None \
                        and env.get_rail_env().agents[agent].position not in decision_cells \
                        and step != max_steps - 1:
                    action_dict[agent] = int(RailEnvActions.MOVE_FORWARD)
                elif info["action_required"][agent]:
                    action_dict[agent] = ppo.act(np.append(obs[agent], [agent]), action_mask[agent])
                else:
                    action_dict[agent] = int(RailEnvActions.DO_NOTHING)

            obs, all_rewards, done, info = env.step(action_dict)

            if train_params.render:
                env.env.show_render()

            if done['__all__']:
                break

        if train_params.render:
            env.env.close()
