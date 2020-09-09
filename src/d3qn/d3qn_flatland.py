import random

import numpy as np
import torch
from flatland.envs.rail_env import RailEnvActions

try:
    import wandb
    use_wandb = True
except ImportError as e:
    raise ImportError("Install wandb and login to load TensorBoard logs.")

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from src.common.action_skipping_masking import find_decision_cells, get_action_masking
from src.common.flatland_railenv import FlatlandRailEnv
from src.common.utils import Timer, TensorBoardLogger
from src.d3qn.policy import D3QNPolicy


def add_fingerprints(obs, num_agents, eps, step):
    """

    :param obs: The observation to modify
    :param num_agents: Total number of agents
    :param eps: the current exploration rate
    :param step: the current step
    :return: the observations with fingerprints
    """

    for a in range(num_agents):
        if obs[a] is not None:
            obs[a] = np.append(obs[a], [eps, step])

    return obs


def train_multiple_agents(env_params, train_params):
    if use_wandb:
        wandb.init(project="flatland-challenge-d3qn-er",
                   entity="lomb",
                   tags="d3qn",
                   config={**vars(train_params), **vars(env_params)},
                   sync_tensorboard=True)

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

    if use_wandb:
        wandb.watch(policy.qnetwork_local)

    # Timers
    training_timer = Timer()
    step_timer = Timer()
    reset_timer = Timer()
    learn_timer = Timer()

    # TensorBoard writer
    tensorboard_logger = TensorBoardLogger(wandb.run.dir)

    ####################################################################################################################
    # Training starts
    training_timer.start()

    print("\nTraining {} trains on {}x{} grid for {} episodes.\n"
          .format(env_params.n_agents, env_params.x_dim, env_params.y_dim, train_params.n_episodes))

    agent_prev_obs = [None] * env_params.n_agents
    agent_prev_action = [2] * env_params.n_agents

    timestep = 0

    for episode in range(train_params.n_episodes + 1):
        # Reset timers
        step_timer.reset()
        reset_timer.reset()
        learn_timer.reset()

        # Reset environment
        reset_timer.start()
        obs, info = env.reset()
        if train_params.fingerprints:
            obs = add_fingerprints(obs, env_params.n_agents, eps_start, timestep)

        decision_cells = find_decision_cells(env.get_rail_env())
        reset_timer.end()

        # Build agent specific observations
        for agent in range(env_params.n_agents):
            if obs[agent] is not None:
                agent_prev_obs[agent] = obs[agent].copy()

        # Run episode
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
                """
                Here it is not necessary an else branch to update the dict.
                By default when no action is given to RailEnv.step() for a specific agent, DO_NOTHING is assigned by the
                Flatland engine.
                """

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = env.step(action_dict)
            if train_params.fingerprints:
                next_obs = add_fingerprints(next_obs, env_params.n_agents, eps_start, timestep)
            step_timer.end()

            for agent in range(env_params.n_agents):
                """
                Update memory and try to perform a learning step only when the agent has finished or when an action was 
                taken and thus relevant information is present, otherwise, for example when action is skipped or an 
                agent is moving from a cell to another, the agent is ignored.
                """
                if agent in agents_in_action or (done[agent] and train_params.type == 1):
                    learn_timer.start()
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], obs[agent],
                                done[agent])
                    learn_timer.end()

                    agent_prev_obs[agent] = obs[agent].copy()

                    # Agent shouldn't be in action_dict in order to print correctly the action's stats
                    if agent not in action_dict:
                        agent_prev_action[agent] = int(RailEnvActions.DO_NOTHING)
                    else:
                        agent_prev_action[agent] = action_dict[agent]

                if next_obs[agent] is not None:
                    obs[agent] = next_obs[agent]

            if train_params.render:
                env.env.show_render()

            timestep += 1

            if done["__all__"]:
                break

        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Save checkpoints
        if train_params.checkpoint_interval is not None and episode % train_params.checkpoint_interval == 0:
            if train_params.save_model_path is not None:
                policy.save(train_params.save_model_path)
        # Rendering
        if train_params.render:
            env.env.close()

        # Update total time
        training_timer.end()

        if train_params.print_stats:
            tensorboard_logger.update_tensorboard(env.env,
                                                  {"loss": policy.get_stat("loss"),
                                                   "q_expected": policy.get_stat("q_expected"),
                                                   "q_targets": policy.get_stat("q_targets"),
                                                   "eps": eps_start}
                                                  if policy.are_stats_ready() else {},
                                                  {"step": step_timer,
                                                   "reset": reset_timer,
                                                   "learn": learn_timer,
                                                   "train": training_timer})
        policy.reset_stats()

    return env.env.accumulated_normalized_score, \
           env.env.accumulated_completion, \
           env.env.accumulated_deadlocks, \
           training_timer.get()
