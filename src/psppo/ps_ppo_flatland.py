import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from flatland.envs.rail_env import RailEnvActions
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus

from src.common.flatland_random_railenv import FlatlandRailEnv
from src.common.timer import Timer
from src.psppo.policy import PsPPOPolicy


def find_decision_cells(env):
    switches = []
    switches_neighbors = []
    directions = list(range(4))
    for h in range(env.height):
        for w in range(env.width):
            pos = (h, w)
            is_switch = False
            # Check for switch counting the outgoing transition
            for orientation in directions:
                possible_transitions = env.rail.get_transitions(*pos, orientation)
                num_transitions = np.count_nonzero(possible_transitions)
                if num_transitions > 1:
                    switches.append(pos)
                    is_switch = True
                    break
            if is_switch:
                # Add all neighbouring rails, if pos is a switch
                for orientation in directions:
                    possible_transitions = env.rail.get_transitions(*pos, orientation)
                    for movement in directions:
                        if possible_transitions[movement]:
                            switches_neighbors.append(get_new_position(pos, movement))

    return set(switches).union(set(switches_neighbors))


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

    print("\nTraining {} trains on {}x{} grid for {} episodes. Update every {} timesteps.\n"
          .format(env_params.n_agents, env_params.x_dim, env_params.y_dim, n_episodes, horizon))

    for episode in range(1, n_episodes + 1):
        # Timers
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()

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
            for agent in range(env_params.n_agents):
                """
                Agents always enter in the if at least once in the episode so there is no further controls.
                When obs is absent because the agent has reached its final goal the observation remains the same.
                """
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

                    # Fill action dict
                    # If an agent is in deadlock leave him learn
                    """
                    if info["deadlocks"][agent]:
                        action_dict[agent] = \
                            ppo.policy_old.act(np.append(obs[agent], [agent]), memory, action_mask[agent],
                                               action=torch.tensor(int(RailEnvActions.DO_NOTHING)).to(device))
                        agents_in_action.add(agent)
                    # If can skip
                    elif train_params.action_skipping and env.get_rail_env().agents[agent].position is not None \
                            and env.get_rail_env().agents[agent].position not in decision_cells:
                        # We always insert in memory the last time step
                        if step == max_steps - 1:
                            action_dict[agent] = \
                                ppo.policy_old.act(np.append(obs[agent], [agent]), memory, action_mask[agent],
                                                   action=torch.tensor(int(RailEnvActions.MOVE_FORWARD)).to(device))
                            agents_in_action.add(agent)
                        # Otherwise skip
                        else:
                            action_dict[agent] = int(RailEnvActions.MOVE_FORWARD)
                    # Else
                    elif info["status"][agent] in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                        action_dict[agent] = \
                            ppo.policy_old.act(np.append(obs[agent], [agent]), memory, action_mask[agent],
                                               action=torch.tensor(int(RailEnvActions.DO_NOTHING)).to(device))
                        agents_in_action.add(agent)
                    else:
                        action_dict[agent] = \
                            ppo.policy_old.act(np.append(obs[agent], [agent]), memory, action_mask[agent])
                        agents_in_action.add(agent)
                    """
                    if info["action_required"][agent]:
                        # If an action is required, we want to store the obs at that step as well as the action
                        action_dict[agent] = \
                            ppo.act(np.append(obs[agent], [agent]), action_mask[agent])
                        agents_in_action.add(agent)
                    else:
                        action_dict[agent] = int(RailEnvActions.DO_NOTHING)

            # Environment step
            step_timer.start()
            obs, rewards, done, info = env.step(action_dict)
            step_timer.end()

            # Update score and compute total rewards equal to each agent considering the rewards shaped or normal
            total_timestep_reward_shaped = sum(rewards[agent] if "rewards_shaped" not in info
                                               else info["rewards_shaped"][agent]
                                               for agent in range(env_params.n_agents))

            # Update dones and rewards for each agent that performed act()
            for a in agents_in_action:
                learn_timer.start()
                ppo.step(a, total_timestep_reward_shaped, done, step == max_steps - 1)
                learn_timer.end()

            if train_params.render:
                env._env.show_render()

            if done["__all__"]:
                break

        # Save checkpoints
        if train_params.checkpoint_interval is not None and episode % train_params.checkpoint_interval == 0:
            if train_params.save_model_path is not None:
                ppo.save(train_params.save_model_path)
        # Rendering
        if train_params.render:
            env._env.close()

        if train_params.print_stats:
            # Save logs to Tensorboard
            writer.add_scalar("training/score", env._env.normalized_score, episode)
            writer.add_scalar("training/accumulated_score", np.mean(env._env.accumulated_normalized_score), episode)
            writer.add_scalar("training/completion", env._env.completion_percentage, episode)
            writer.add_scalar("training/accumulated_completion", np.mean(env._env.accumulated_completion), episode)
            writer.add_scalar("training/deadlocks", env._env.deadlocks_percentage, episode)
            writer.add_scalar("training/accumulated_deadlocks", np.mean(env._env.accumulated_deadlocks), episode)
            writer.add_histogram("actions/distribution", np.array(env._env.action_probs), episode)
            writer.add_scalar("actions/nothing", env._env.action_probs[RailEnvActions.DO_NOTHING], episode)
            writer.add_scalar("actions/left", env._env.action_probs[RailEnvActions.MOVE_LEFT], episode)
            writer.add_scalar("actions/forward", env._env.action_probs[RailEnvActions.MOVE_FORWARD], episode)
            writer.add_scalar("actions/right", env._env.action_probs[RailEnvActions.MOVE_RIGHT], episode)
            writer.add_scalar("actions/stop", env._env.action_probs[RailEnvActions.STOP_MOVING], episode)
            writer.add_scalar("training/loss", ppo.loss, episode)
            writer.add_scalar("timer/reset", reset_timer.get(), episode)
            writer.add_scalar("timer/step", step_timer.get(), episode)
            writer.add_scalar("timer/learn", learn_timer.get(), episode)
            writer.add_scalar("timer/total", training_timer.get_current(), episode)

    training_timer.end()

    return env._env.accumulated_normalized_score, env._env.accumulated_completion, env._env.accumulated_deadlocks
