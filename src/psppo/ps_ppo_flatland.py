import random
from argparse import Namespace
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from flatland.envs.rail_env import RailEnvActions
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.malfunction_generators import MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus

from src.common.flatland_random_railenv import FlatlandRandomRailEnv
from src.common.timer import Timer
from src.psppo.algorithm import PsPPO
from src.psppo.memory import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    env = FlatlandRandomRailEnv(train_params, env_params, tree_observation)
    env.reset()

    # The action space of flatland is 5 discrete actions
    action_size = env.get_rail_env().action_space[0]

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    max_steps = int(4 * 2 * (env_params.y_dim + env_params.x_dim + (env_params.n_agents / env_params.n_cities)))

    memory = Memory(env_params.n_agents)

    ppo = PsPPO(env.state_size,
                action_size,
                device,
                train_params)

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
                    if info["deadlocks"][agent]:
                        action_dict[agent] = \
                            ppo.policy_old.act(np.append(obs[agent], [agent]), memory, action_mask[agent],
                                               action=torch.tensor(int(RailEnvActions.DO_NOTHING)).to(device))
                        agents_in_action.add(agent)
                    # If can skip
                    elif train_params.action_skipping \
                            and env.get_rail_env().agents[
                        agent].position is not None and env.get_rail_env().rail.get_full_transitions(
                        env.get_rail_env().agents[agent].position[0],
                        env.get_rail_env().agents[agent].position[1]) not in decision_cells:
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

            # Environment step
            step_timer.start()
            obs, rewards, done, info = env.step(action_dict)
            step_timer.end()
            # Update score and compute total rewards equal to each agent
            total_timestep_reward_shaped = sum(rewards[agent] if isinstance(rewards[agent], float) or isinstance(rewards[agent], int) else
                                               rewards[agent]["rewards_shaped"] for agent in range(env_params.n_agents))

            # Update dones and rewards for each agent that performed act()
            for a in agents_in_action:
                memory.rewards[a].append(total_timestep_reward_shaped)
                memory.dones[a].append(done["__all__"])

                # Set dones to True when the episode is finished because the maximum number of steps has been reached
                if step == max_steps - 1:
                    memory.dones[a][-1] = True

            for a in range(env_params.n_agents):
                # Update if agent's horizon has been reached
                if len(memory.states[a]) % (horizon + 1) == 0:
                    learn_timer.start()
                    ppo.update(memory, a)
                    learn_timer.end()

                    """
                    Leave last memory unit because the batch includes an additional step which has not been considered 
                    in the current trajectory (it has been inserted to compute the advantage) but must be considered in 
                    the next trajectory or will be lost.
                    """
                    memory.clear_memory_except_last(a)

            if train_params.render:
                env._env.show_render()

            """
            if done["__all__"]:
                break
            """

        # Save checkpoints
        if train_params.checkpoint_interval is not None and episode % train_params.checkpoint_interval == 0:
            if train_params.save_model_path is not None:
                ppo.policy.save(train_params.save_model_path)
        # Rendering
        if train_params.render:
            env._env.close()

        # Evaluation
        if train_params.checkpoint_interval is not None and episode % train_params.checkpoint_interval == 0:
            with torch.no_grad():
                scores, completions, deads = eval_policy(env, action_size, ppo, train_params,
                                                         train_params.eval_episodes, max_steps)
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
            writer.add_scalar("evaluation/deadlocks_min", np.min(deads), episode)
            writer.add_scalar("evaluation/deadlocks_max", np.max(deads), episode)
            writer.add_scalar("evaluation/deadlocks_mean", np.mean(deads), episode)
            writer.add_scalar("evaluation/deadlocks_std", np.std(deads), episode)
            writer.add_histogram("evaluation/deadlocks", np.array(deads), episode)
            env._env.accumulated_eval_normalized_score.append(np.mean(scores))
            env._env.accumulated_eval_completion.append(np.mean(completions))
            env._env.accumulated_eval_deads.append(np.mean(deads))
            writer.add_scalar("evaluation/accumulated_score", np.mean(env._env.accumulated_eval_normalized_score),
                              episode)
            writer.add_scalar("evaluation/accumulated_completion", np.mean(env._env.accumulated_eval_completion),
                              episode)
            writer.add_scalar("evaluation/accumulated_deadlocks", np.mean(env._env.accumulated_eval_deads), episode)
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


def eval_policy(env, action_size, ppo, train_params, n_eval_episodes, max_steps):
    action_count = [1] * action_size
    scores = []
    completions = []
    deads = []

    for episode in range(1, n_eval_episodes + 1):

        # Reset environment
        obs, info = env.reset()
        decision_cells = find_decision_cells(env.get_rail_env())

        # Score of the episode as a sum of scores of each step for statistics
        score = 0.0

        # Run episode
        for step in range(max_steps):
            # Action counter used for statistics
            action_dict = dict()

            # Set used to track agents that didn't skipped the action
            agents_in_action = set()

            # Mask initialization
            action_mask = [[1 * (0 if action == 0 and not train_params.allow_no_op else 1)
                            for action in range(action_size)] for _ in range(env.get_rail_env().get_num_agents())]

            # Collect and preprocess observations and fill action dictionary
            for agent in env.get_rail_env().get_agent_handles():
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
                                    action_mask[agent][action] = RailEnvActions.DO_NOTHING

                    # Fill action dict
                    # If an agent is in deadlock leave him learn
                    if info["deadlocks"][agent]:
                        action_dict[agent] = \
                            ppo.policy_old.act(np.append(obs[agent], [agent]), None, action_mask[agent],
                                               action=torch.tensor(int(RailEnvActions.DO_NOTHING)).to(device))
                        agents_in_action.add(agent)
                    # If can skip
                    elif train_params.action_skipping \
                            and env.get_rail_env().agents[
                        agent].position is not None and env.get_rail_env().rail.get_full_transitions(
                        env.get_rail_env().agents[agent].position[0],
                        env.get_rail_env().agents[agent].position[1]) in decision_cells:
                        # We always insert in memory the last time step
                        if step == max_steps - 1:
                            action_dict[agent] = \
                                ppo.policy_old.act(np.append(obs[agent], [agent]), None, action_mask[agent],
                                                   action=torch.tensor(int(RailEnvActions.MOVE_FORWARD)).to(device))
                            agents_in_action.add(agent)
                        # Otherwise skip
                        else:
                            action_dict[agent] = int(RailEnvActions.MOVE_FORWARD)
                    # Else
                    elif info["status"][agent] in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                        print("DONE")
                        action_dict[agent] = \
                            ppo.policy_old.act(np.append(obs[agent], [agent]), None, action_mask[agent],
                                               action=torch.tensor(int(RailEnvActions.DO_NOTHING)).to(device))
                        agents_in_action.add(agent)
                    else:
                        action_dict[agent] = \
                            ppo.policy_old.act(np.append(obs[agent], [agent]), None, action_mask[agent])
                        agents_in_action.add(agent)

            # Update statistics
            for a in list(action_dict.values()):
                action_count[a] += 1

            # Environment step
            obs, rewards, done, info = env.step(action_dict)
            # Update deadlocks
            # Update score and compute total rewards equal to each agent
            score += np.sum(rewards[agent] if isinstance(rewards[agent], float) or isinstance(rewards[agent], int) else
                            rewards[agent]["standard_rewards"] for agent in range(env.get_rail_env().get_num_agents()))

        scores.append(score / (max_steps * env.get_rail_env().get_num_agents()))
        tasks_finished = sum(info["status"][a] in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]
                             for a in env.get_rail_env().get_agent_handles())
        completions.append(tasks_finished / max(1, env.get_rail_env().get_num_agents()))
        deads.append(sum([info["deadlocks"][agent] for agent in range(env.get_rail_env().get_num_agents())])
                     / env.get_rail_env().get_num_agents())

    print("\t Eval: score {:.3f} done {:.1f} dead {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0,
                                                                  np.mean(deads) * 100.0))

    return scores, completions, deads


if __name__ == "__main__":
    myseed = 14

    datehour = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    print(datehour)

    environment_parameters = {
        "n_agents": 3,
        "x_dim": 16 * 3,
        "y_dim": 9 * 3,
        "n_cities": 5,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "seed": myseed,
        "observation_tree_depth": 5,
        "observation_radius": 35,
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
        # Shared actor-critic layer
        # If shared is True then the considered sizes are taken from the critic
        "shared": False,
        # Policy network
        "critic_mlp_width": 256,
        "critic_mlp_depth": 4,
        "last_critic_layer_scaling": 0.1,
        # Actor network
        "actor_mlp_width": 128,
        "actor_mlp_depth": 4,
        "last_actor_layer_scaling": 0.01,
        # Adam learning rate
        "learning_rate": 0.001,
        # Adam epsilon
        "adam_eps": 1e-5,
        # Activation
        "activation": "Tanh",
        "lmbda": 0.95,
        "entropy_coefficient": 0.1,
        # Called also baseline cost in shared setting (0.5)
        # (C54): {0.001, 0.1, 1.0, 10.0, 100.0}
        "value_loss_coefficient": 0.001,

        # ============================
        # Training setup
        # ============================
        "n_episodes": 2500,
        # 512, 1024, 2048, 4096
        "horizon": 1024,
        "epochs": 4,
        # 64, 128, 256
        "batch_size": 64,

        # ============================
        # Normalization and clipping
        # ============================
        # Discount factor (0.95, 0.97, 0.99, 0.999)
        "discount_factor": 0.99,
        "max_grad_norm": 0.5,
        # PPO-style value clipping
        "eps_clip": 0.25,

        # ============================
        # Advantage estimation
        # ============================
        # gae or n-steps
        "advantage_estimator": "gae",

        # ============================
        # Optimization and rendering
        # ============================
        # Save and evaluate interval
        "checkpoint_interval": None,
        "eval_episodes": None,
        "use_gpu": False,
        "render": True,
        "save_model_path": "checkpoint.pt",
        "load_model_path": "checkpoint.pt",
        "tensorboard_path": "log/",

        # ============================
        # Action Masking / Skipping
        # ============================
        "action_masking": True,
        "allow_no_op": False,
        "action_skipping": True
    }

    """
    # Save on Google Drive on Colab
    "save_model_path": "/content/drive/My Drive/Colab Notebooks/models/" + datehour + ".pt",
    "load_model_path": "/content/drive/My Drive/Colab Notebooks/models/todo.pt",
    "tensorboard_path": "/content/drive/My Drive/Colab Notebooks/logs" + datehour + "/",
    """

    """
    # Mount Drive on Colab
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)

    # Show Tensorboard on Colab
    import tensorflow
    %load_ext tensorboard
    % tensorboard --logdir "/content/drive/My Drive/Colab Notebooks/logs_todo"
    """

    train_multiple_agents(Namespace(**environment_parameters), Namespace(**training_parameters))
