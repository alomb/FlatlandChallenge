import random

import numpy as np
import torch

from flatland.envs.rail_env import RailEnvActions
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.agent_utils import RailAgentStatus

from src.common.action_skipping import find_decision_cells
from src.common.flatland_random_railenv import FlatlandRailEnv
from src.common.utils import Timer, TensorBoardLogger
from src.psppo.policy import PsPPOPolicy


def train_multiple_agents(env_params, train_params):
    # Environment parameters
    seed = env_params.seed

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_max_path_depth = env_params.observation_max_path_depth

    # Training setup parameters
    n_episodes = train_params.n_episodes
    horizon = train_params.horizon

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

    # Timers
    training_timer = Timer()
    step_timer = Timer()
    reset_timer = Timer()
    learn_timer = Timer()

    # Remove attributes not printable by Tensorboard
    board_env_params = vars(env_params)
    del board_env_params["speed_profiles"]
    del board_env_params["malfunction_parameters"]

    # TensorBoard writer
    tensorboard_logger = TensorBoardLogger(train_params.tensorboard_path, board_env_params, vars(train_params))

    ####################################################################################################################
    # Training starts
    training_timer.start()

    print("\nTraining {} trains on {}x{} grid for {} episodes. Update every {} timesteps.\n"
          .format(env_params.n_agents, env_params.x_dim, env_params.y_dim, n_episodes, horizon))

    for episode in range(1, n_episodes + 1):
        # Reset timers
        step_timer.reset()
        reset_timer.reset()
        learn_timer.reset()

        # Reset environment
        reset_timer.start()
        obs, info = env.reset()

        decision_cells = find_decision_cells(env.get_rail_env())
        reset_timer.end()

        # Run episode
        for step in range(max_steps):
            # Action counter used for statistics
            action_dict = dict()

            # Set used to track agents that didn't skipped the action
            agents_in_action = set()

            # Mask initialization
            action_mask = [[1 * (0 if action == 0 and not train_params.allow_no_op else 1)
                            for action in range(action_size)] for _ in range(env_params.n_agents)]

            # Collect and preprocess observations and fill action dictionary
            for agent in obs:
                """
                Agents always enter in the if at least once in the episode so there is no further controls.
                When obs is absent because the agent has reached its final goal the observation remains the same.
                """
                # Action mask modification only if action masking is True
                if train_params.action_masking:
                    for action in range(action_size):
                        if env.get_rail_env().agents[agent].status == RailAgentStatus.ACTIVE or \
                                env.get_rail_env().agents[agent].status == RailAgentStatus.DONE:
                            _, cell_valid, _, _, transition_valid = env.get_rail_env()._check_action_on_agent(
                                RailEnvActions(action),
                                env.get_rail_env().agents[agent])
                            if not all([cell_valid, transition_valid]):
                                action_mask[agent][action] = 0

                # Fill action dict
                # Action skipping if in correct cell and not in last time step which is always inserted in memory
                if train_params.action_skipping and env.get_rail_env().agents[agent].position is not None \
                        and env.get_rail_env().agents[agent].position not in decision_cells \
                        and step != max_steps - 1:
                    action_dict[agent] = int(RailEnvActions.MOVE_FORWARD)
                # If agent is not arrived or moving between two cells
                elif info["action_required"][agent]:
                    # If an action is required, we want to store the obs at that step as well as the action
                    action_dict[agent] = ppo.act(np.append(obs[agent], [agent]), action_mask[agent])
                    agents_in_action.add(agent)
                # It is not necessary an else branch.
                # By default when no action is given to RailEnv.step() DO_NOTHING is performed

            # Environment step
            step_timer.start()
            obs, rewards, done, info = env.step(action_dict)
            step_timer.end()

            # Update dones and rewards for each agent that performed act()
            for a in agents_in_action:
                learn_timer.start()
                #ppo.step(a, float(sum(rewards.values())), done, step == max_steps - 1)
                ppo.step(a, rewards[a], done, step == max_steps - 1)
                learn_timer.end()

            if train_params.render:
                env.env.show_render()

            if done["__all__"]:
                break

        # Save checkpoints
        if train_params.checkpoint_interval is not None and episode % train_params.checkpoint_interval == 0:
            if train_params.save_model_path is not None:
                ppo.save(train_params.save_model_path)
        # Rendering
        if train_params.render:
            env.env.close()

        # Update total time
        training_timer.end()

        # Update Tensorboard statistics
        if train_params.print_stats:
            tensorboard_logger.update_tensorboard(episode,
                                                  env.env,
                                                  {"state_estimated_value": ppo.state_estimated_value_metric,
                                                   "probs_ratio": ppo.probs_ratio_metric,
                                                   "advantage": ppo.advantage_metric,
                                                   "policy_loss": ppo.policy_loss_metric,
                                                   "value_loss": ppo.value_loss_metric,
                                                   "entropy_loss": ppo.entropy_loss_metric,
                                                   "total_loss": ppo.loss_metric},
                                                  {"step": step_timer,
                                                   "reset": reset_timer,
                                                   "learn": learn_timer,
                                                   "train": training_timer})

    return env.env.accumulated_normalized_score, \
           env.env.accumulated_completion, \
           env.env.accumulated_deadlocks, \
           training_timer.get()
