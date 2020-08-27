import random

import numpy as np
import torch

from flatland.envs.rail_env import RailEnvActions
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from src.common.action_skipping_masking import find_decision_cells, get_action_masking
from src.common.flatland_random_railenv import FlatlandRailEnv
from src.common.utils import Timer, TensorBoardLogger
from src.psppo.policy import PsPPOPolicy


def get_agent_ids(agents, malfunction_rate):

    max_speed = 1.0
    max_malfunction_rate = 0.1 * 10
    max_value = max_speed + max_malfunction_rate

    return {a.handle: (a.speed_data["speed"] + malfunction_rate * 10) / max_value for a in agents}


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
    board_env_params = vars(env_params).copy()
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
        prev_obs, info = env.reset()

        decision_cells = find_decision_cells(env.get_rail_env())

        agent_ids = get_agent_ids(env.get_rail_env().agents, env_params.malfunction_parameters.malfunction_rate)
        reset_timer.end()

        # Run episode
        for step in range(max_steps):
            # Action dictionary to feed to step
            action_dict = dict()

            # Set used to track agents that didn't skipped the action
            agents_in_action = set()

            # Flag to control the last step of an episode
            is_last_step = step == (max_steps - 1)

            """
            Collect trajectories and fill action dictionary.
            When an agent's observation is absent is because the agent has reached its final goal, to allow also agents
            that reached their goal to fill trajectories prev_obs[a] is updated only when th agent has not reached 
            the goal. In conclusion, prev_obs[a] can be a new observation or the last one.
            """
            for agent in prev_obs:
                # Create action mask
                action_mask = get_action_masking(env, agent, action_size, train_params)

                # Fill action dict
                # Action skipping if the agent is in not in a decision cell and not in last step.
                # TODO: action skipping may skips agents arrival and done agent
                if train_params.action_skipping and env.get_rail_env().agents[agent].position is not None \
                        and env.get_rail_env().agents[agent].position not in decision_cells\
                        and not is_last_step:
                    action_dict[agent] = int(RailEnvActions.MOVE_FORWARD)
                # If agent is moving between two cells or trapped in a deadlock (the latter is caught only
                # when the agent is moving in the deadlock triggering the first case) or the step is the last or the
                # agent has reached its destination.
                elif info["action_required"][agent]: #or is_last_step or done[agent]:

                    # If an action is required, the actor predicts an action and the obs, actions, masks are stored
                    action_dict[agent] = ppo.act(np.append(prev_obs[agent], [agent_ids[agent]]),
                                                 action_mask, agent_id=agent)
                    agents_in_action.add(agent)
                """
                Here it is not necessary an else branch to update the dict.
                By default when no action is given to RailEnv.step() for a specific agent, DO_NOTHING is assigned by the
                Flatland engine.
                """

            # Environment step
            step_timer.start()
            next_obs, rewards, done, info = env.step(action_dict)
            step_timer.end()

            """
            Update observation only if agent has not reached the target. When an agent reaches its target the returned 
            obs is None.
            """
            for a in range(env_params.n_agents):
                if not done[a]:
                    prev_obs[a] = next_obs[a].copy()

            for a in range(env_params.n_agents):
                # Update dones and rewards for each agent that performed act() or step is the episode's last or has
                # finished

                # To represent the end of the episode inside the trajectory of each agent.
                if is_last_step:
                    done[a] = True

                if a in agents_in_action:
                    learn_timer.start()
                    ppo.step(a, rewards[a], done[a])
                    learn_timer.end()

            if train_params.render:
                env.env.show_render()

            # If all agents have been arrived and this is not the last step do another one, otherwise stop
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
