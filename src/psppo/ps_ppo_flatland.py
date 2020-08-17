import torch
from flatland.envs.agent_utils import RailAgentStatus

from torch.utils.tensorboard import SummaryWriter

import random
from argparse import Namespace

from flatland.utils.rendertools import RenderTool
import numpy as np

from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from flatland.core.grid.grid4_utils import get_new_position

from src.common.env_wrapper import EnvWrapper
from src.common.observation import NormalizeObservations
from src.common.stats import Stats
from src.common.timer import Timer
from src.psppo.algorithm import PsPPO
from src.psppo.memory import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_multiple_agents(env_params, train_params):
    # Environment parameters
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    seed = env_params.seed

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_radius = env_params.observation_radius
    observation_max_path_depth = env_params.observation_max_path_depth

    # Custom observations&rewards
    custom_observations = env_params.custom_observations
    stop_penalty = env_params.stop_penalty
    invalid_action_penalty = env_params.invalid_action_penalty
    done_bonus = env_params.done_bonus
    deadlock_penalty = env_params.deadlock_penalty
    shortest_path_penalty_coefficient = env_params.shortest_path_penalty_coefficient

    # Training setup parameters
    n_episodes = train_params.n_episodes
    horizon = train_params.horizon

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environment
    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=env_params.max_rails_between_cities,
            max_rails_in_city=env_params.max_rails_in_city,
            seed=seed
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

    # State size depends on features per nodes in observations, custom observations and + 1 (agent id of PS-PPO)
    # state_size = n_features_per_node * n_nodes + (custom_observations * (env.width * env.height * 16 + 8)) + 1
    state_size = n_features_per_node * n_nodes + (custom_observations * (env.width * env.height * 23 + 1)) + 1

    # The action space of flatland is 5 discrete actions
    action_size = env.action_space[0]

    normalize_observations = NormalizeObservations(n_features_per_node,
                                                   n_nodes,
                                                   custom_observations,
                                                   state_size,
                                                   observation_tree_depth,
                                                   observation_radius)

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    max_steps = int(4 * 2 * (env.height + env.width + (env.get_num_agents() / n_cities)))

    memory = Memory(env.get_num_agents())

    ppo = PsPPO(state_size,
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
          .format(env.get_num_agents(), x_dim, y_dim, n_episodes, horizon))

    # Variables to compute statistics
    action_count = [0] * action_size

    env_wrapper = EnvWrapper(env,
                             invalid_action_penalty,
                             stop_penalty,
                             deadlock_penalty,
                             shortest_path_penalty_coefficient,
                             done_bonus)
    stats = Stats()

    for episode in range(1, n_episodes + 1):
        # Timers
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()

        # Reset environment
        reset_timer.start()
        obs, info = env_wrapper.reset()

        rail_obs = normalize_observations.reset_rail_obs(env)

        decision_cells = env_wrapper.find_decision_cells()
        reset_timer.end()

        # Setup renderer
        if train_params.render:
            env_renderer = RenderTool(env, gl="PGL")
        else:
            env_renderer = None
        if train_params.render:
            env_renderer.set_new_rail()

        # Score of the episode as a sum of scores of each step for statistics
        score = 0

        # Observation related information
        agent_obs = [None] * env.get_num_agents()

        # Run episode
        for step in range(max_steps):
            # Action counter used for statistics
            action_dict = dict()

            # Set used to track agents that didn't skipped the action
            agents_in_action = set()

            # Mask initialization
            action_mask = [[1 * (0 if action == 0 and not train_params.allow_no_op else 1)
                            for action in range(action_size)] for _ in range(env.get_num_agents())]

            # Collect and preprocess observations and fill action dictionary
            for agent in env.get_agent_handles():
                """
                Agents always enter in the if at least once in the episode so there is no further controls.
                When obs is absent because the agent has reached its final goal the observation remains the same.
                """
                preproc_timer.start()
                if obs[agent]:
                    agent_obs[agent] = normalize_observations.normalize_observation(obs[agent], env, agent,
                                                                                    env_wrapper.deadlocks, rail_obs)

                    # Action mask modification only if action masking is True
                    if train_params.action_masking:
                        for action in range(action_size):
                            if env.agents[agent].status != RailAgentStatus.READY_TO_DEPART:
                                _, cell_valid, _, _, transition_valid = env._check_action_on_agent(
                                    RailEnvActions(action),
                                    env.agents[agent])
                                if not all([cell_valid, transition_valid]):
                                    action_mask[agent][action] = 0

                preproc_timer.end()

                # Fill action dict
                # If an agent is in deadlock leave him learn
                if env_wrapper.deadlocks[agent]:
                    action_dict[agent] = \
                        ppo.policy_old.act(np.append(agent_obs[agent], [agent]), memory, action_mask[agent],
                                           action=torch.tensor(int(RailEnvActions.DO_NOTHING)).to(device))
                    agents_in_action.add(agent)
                # If can skip
                elif train_params.action_skipping \
                        and env.agents[agent].position is not None and env.rail.get_full_transitions(
                    env.agents[agent].position[0], env.agents[agent].position[1]) not in decision_cells:
                    # We always insert in memory the last time step
                    if step == max_steps - 1:
                        action_dict[agent] = \
                            ppo.policy_old.act(np.append(agent_obs[agent], [agent]), memory, action_mask[agent],
                                               action=torch.tensor(int(RailEnvActions.MOVE_FORWARD)).to(device))
                        agents_in_action.add(agent)
                    # Otherwise skip
                    else:
                        action_dict[agent] = int(RailEnvActions.MOVE_FORWARD)
                # Else
                elif info["status"][agent] in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                    action_dict[agent] = \
                        ppo.policy_old.act(np.append(agent_obs[agent], [agent]), memory, action_mask[agent],
                                           action=torch.tensor(int(RailEnvActions.DO_NOTHING)).to(device))
                    agents_in_action.add(agent)
                else:
                    action_dict[agent] = \
                        ppo.policy_old.act(np.append(agent_obs[agent], [agent]), memory, action_mask[agent])
                    agents_in_action.add(agent)

            # Update statistics
            for a in list(action_dict.values()):
                action_count[a] += 1

            # Environment step
            step_timer.start()
            obs, rewards, done, info, rewards_shaped = env_wrapper.step(action_dict, action_mask)
            step_timer.end()

            # Update score and compute total rewards equal to each agent
            score += np.sum(list(rewards.values()))
            total_timestep_reward_shaped = np.sum(list(rewards_shaped.values()))

            # Update dones and rewards for each agent that performed act()
            for a in agents_in_action:
                memory.rewards[a].append(total_timestep_reward_shaped)
                memory.dones[a].append(done["__all__"])

                # Set dones to True when the episode is finished because the maximum number of steps has been reached
                if step == max_steps - 1:
                    memory.dones[a][-1] = True

            for a in range(env.get_num_agents()):
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
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

            """
            if done["__all__"]:
                break
            """

        # Collection information about training
        normalized_score, tasks_finished, completion_percentage, deadlocks_percentage, action_probs = \
            stats.step(score, max_steps, env.get_num_agents(), info, env_wrapper.deadlocks, action_count)

        # Save checkpoints
        if train_params.checkpoint_interval is not None and episode % train_params.checkpoint_interval == 0:
            if train_params.save_model_path is not None:
                ppo.policy.save(train_params.save_model_path)
        # Rendering
        if train_params.render:
            env_renderer.close_window()

        print(
            "\rEpisode {}"
            "\tScore: {:.3f}"
            " Avg: {:.3f}"
            "\tDone: {:.2f}%"
            " Avg: {:.2f}%"
            "\tDeads: {:.2f}%"
            " Avg: {:.2f}%"
            "\tAction Probs: {}".format(
                episode,
                normalized_score,
                np.mean(stats.accumulated_normalized_score),
                100 * completion_percentage,
                100 * np.mean(stats.accumulated_completion),
                100 * deadlocks_percentage,
                100 * np.mean(stats.accumulated_deadlocks),
                format_action_prob(action_probs)
            ), end=" ")

        # Evaluation
        if train_params.checkpoint_interval is not None and episode % train_params.checkpoint_interval == 0:
            with torch.no_grad():
                scores, completions, deads = eval_policy(env, action_size, ppo, train_params, env_params,
                                                         train_params.eval_episodes, max_steps, normalize_observations)
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
            stats.accumulated_eval_normalized_score.append(np.mean(scores))
            stats.accumulated_eval_completion.append(np.mean(completions))
            stats.accumulated_eval_deads.append(np.mean(deads))
            writer.add_scalar("evaluation/accumulated_score", np.mean(stats.accumulated_eval_normalized_score), episode)
            writer.add_scalar("evaluation/accumulated_completion", np.mean(stats.accumulated_eval_completion), episode)
            writer.add_scalar("evaluation/accumulated_deadlocks", np.mean(stats.accumulated_eval_deads), episode)
        # Save logs to Tensorboard
        writer.add_scalar("training/score", normalized_score, episode)
        writer.add_scalar("training/accumulated_score", np.mean(stats.accumulated_normalized_score), episode)
        writer.add_scalar("training/completion", completion_percentage, episode)
        writer.add_scalar("training/accumulated_completion", np.mean(stats.accumulated_completion), episode)
        writer.add_scalar("training/deadlocks", deadlocks_percentage, episode)
        writer.add_scalar("training/accumulated_deadlocks", np.mean(stats.accumulated_deadlocks), episode)
        writer.add_histogram("actions/distribution", np.array(action_probs), episode)
        writer.add_scalar("actions/nothing", action_probs[RailEnvActions.DO_NOTHING], episode)
        writer.add_scalar("actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode)
        writer.add_scalar("actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode)
        writer.add_scalar("actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode)
        writer.add_scalar("actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode)
        writer.add_scalar("training/loss", ppo.loss, episode)
        writer.add_scalar("timer/reset", reset_timer.get(), episode)
        writer.add_scalar("timer/step", step_timer.get(), episode)
        writer.add_scalar("timer/learn", learn_timer.get(), episode)
        writer.add_scalar("timer/preproc", preproc_timer.get(), episode)
        writer.add_scalar("timer/total", training_timer.get_current(), episode)

    training_timer.end()


def eval_policy(env, action_size, ppo, train_params, env_params, n_eval_episodes, max_steps, normalize_observations):
    action_count = [1] * action_size
    scores = []
    completions = []
    deads = []

    env_wrapper = EnvWrapper(env,
                             env_params.invalid_action_penalty,
                             env_params.stop_penalty,
                             env_params.deadlock_penalty,
                             env_params.shortest_path_penalty_coefficient,
                             env_params.done_bonus)

    for episode in range(1, n_eval_episodes + 1):

        # Reset environment
        obs, info = env_wrapper.reset()
        decision_cells = env_wrapper.find_decision_cells()

        rail_obs = normalize_observations.reset_rail_obs(env)

        # Score of the episode as a sum of scores of each step for statistics
        score = 0.0

        # Observation related information
        agent_obs = [None] * env.get_num_agents()

        # Run episode
        for step in range(max_steps):
            # Action counter used for statistics
            action_dict = dict()

            # Set used to track agents that didn't skipped the action
            agents_in_action = set()

            # Mask initialization
            action_mask = [[1 * (0 if action == 0 and not train_params.allow_no_op else 1)
                            for action in range(action_size)] for _ in range(env.get_num_agents())]

            # Collect and preprocess observations and fill action dictionary
            for agent in env.get_agent_handles():
                """
                Agents always enter in the if at least once in the episode so there is no further controls.
                When obs is absent because the agent has reached its final goal the observation remains the same.
                """
                if obs[agent]:
                    agent_obs[agent] = normalize_observations.normalize_observation(obs[agent], env, agent, env_wrapper.deadlocks,
                                                                                    rail_obs)

                    # Action mask modification only if action masking is True
                    if train_params.action_masking:
                        for action in range(action_size):
                            if env.agents[agent].status != RailAgentStatus.READY_TO_DEPART:
                                _, cell_valid, _, _, transition_valid = env._check_action_on_agent(
                                    RailEnvActions(action),
                                    env.agents[agent])
                                if not all([cell_valid, transition_valid]):
                                    action_mask[agent][action] = RailEnvActions.DO_NOTHING

                # Fill action dict
                # If an agent is in deadlock leave him learn
                if env_wrapper.deadlocks[agent]:
                    action_dict[agent] = \
                        ppo.policy_old.act(np.append(agent_obs[agent], [agent]), None, action_mask[agent],
                                           action=torch.tensor(int(RailEnvActions.DO_NOTHING)).to(device))
                    agents_in_action.add(agent)
                # If can skip
                elif train_params.action_skipping \
                        and env.agents[agent].position is not None and env.rail.get_full_transitions(
                    env.agents[agent].position[0], env.agents[agent].position[1]) in decision_cells:
                    # We always insert in memory the last time step
                    if step == max_steps - 1:
                        action_dict[agent] = \
                            ppo.policy_old.act(np.append(agent_obs[agent], [agent]), None, action_mask[agent],
                                               action=torch.tensor(int(RailEnvActions.MOVE_FORWARD)).to(device))
                        agents_in_action.add(agent)
                    # Otherwise skip
                    else:
                        action_dict[agent] = int(RailEnvActions.MOVE_FORWARD)
                # Else
                elif info["status"][agent] in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                    action_dict[agent] = \
                        ppo.policy_old.act(np.append(agent_obs[agent], [agent]), None, action_mask[agent],
                                           action=torch.tensor(int(RailEnvActions.DO_NOTHING)).to(device))
                    agents_in_action.add(agent)
                else:
                    action_dict[agent] = \
                        ppo.policy_old.act(np.append(agent_obs[agent], [agent]), None, action_mask[agent])
                    agents_in_action.add(agent)

            # Update statistics
            for a in list(action_dict.values()):
                action_count[a] += 1

            # Environment step
            obs, rewards, done, info, rewards_shaped = env_wrapper.step(action_dict, action_mask)

            # Update deadlocks
            # Update score and compute total rewards equal to each agent
            score += np.sum(list(rewards.values()))

        scores.append(score / (max_steps * env.get_num_agents()))
        tasks_finished = sum(info["status"][a] in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]
                             for a in env.get_agent_handles())
        completions.append(tasks_finished / max(1, env.get_num_agents()))
        deads.append(sum(env_wrapper.deadlocks) / max(1, env.get_num_agents()))

    print("\t Eval: score {:.3f} done {:.1f} dead {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0,
                                                                  np.mean(deads) * 100.0))

    return scores, completions, deads


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


from datetime import datetime

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
        "horizon": 512,
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
        "checkpoint_interval": 1,
        "eval_episodes": 1,
        "use_gpu": False,
        "render": False,
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