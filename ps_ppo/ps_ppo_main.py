import random
from argparse import Namespace

import numpy as np
import torch
from flatland.envs.malfunction_generators import MalfunctionParameters, malfunction_from_params
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

from ps_ppo.algorithm import Memory, PsPPO
from ps_ppo.timer import Timer
from ps_ppo.observation_parsing import normalize_observation
from ps_ppo.reward_shaping import step_shaping


def train_multiple_agents(env_params, train_params):
    # Environment parameters
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city
    seed = env_params.seed

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_radius = env_params.observation_radius
    observation_max_path_depth = env_params.observation_max_path_depth

    # Training setup parameters
    n_episodes = train_params.n_episodes
    checkpoint_interval = train_params.checkpoint_interval
    horizon = train_params.horizon

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1. / 10000,  # Rate of malfunctions
        min_duration=15,  # Minimal duration
        max_duration=50  # Max duration
    )

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Fraction of train which each speed
    speed_profiles = {
        # Fast passenger train
        1.: 1.0,
        # Fast freight train
        1. / 2.: 0.0,
        # Slow commuter train
        1. / 3.: 0.0,
        # Slow freight train
        1. / 4.: 0.0
    }

    # Setup the environment
    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city,
            seed=seed
        ),
        schedule_generator=sparse_schedule_generator(speed_profiles),
        number_of_agents=n_agents,
        malfunction_generator_and_process_data=malfunction_from_params(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed
    )

    env.reset(regenerate_schedule=True, regenerate_rail=True)

    # Setup renderer

    if train_params.render:
        env_renderer = RenderTool(env, gl="PGL")
    else:
        env_renderer = None

    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes

    # The action space of flatland is 5 discrete actions
    action_size = env.action_space[0]

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))

    action_count = [0] * action_size
    smoothed_normalized_score = -1.0
    smoothed_completion = 0.0

    memory = Memory(n_agents)

    ppo = PsPPO(n_agents,
                # + 1 because also agent id is passed
                state_size + 1,
                action_size,
                train_params.shared,
                train_params.critic_mlp_width,
                train_params.critic_mlp_depth,
                train_params.last_critic_layer_scaling,
                train_params.actor_mlp_width,
                train_params.actor_mlp_depth,
                train_params.last_actor_layer_scaling,
                train_params.learning_rate,
                train_params.adam_eps,
                train_params.activation,
                train_params.discount_factor,
                train_params.epochs,
                train_params.batch_size,
                train_params.eps_clip,
                train_params.lmbda,
                train_params.advantage_estimator,
                train_params.value_loss_function,
                train_params.entropy_coefficient,
                train_params.value_loss_coefficient)

    """
    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_hparams(vars(train_params), {})
    writer.add_hparams(vars(env_params), {})
    """

    training_timer = Timer()
    training_timer.start()

    print("\nTraining {} trains on {}x{} grid for {} episodes. Update every {} timesteps.\n"
          .format(env.get_num_agents(), x_dim, y_dim, n_episodes, horizon))

    timestep = 0
    PATH = "model.pt"

    for episode in range(n_episodes + 1):

        # Timers
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()

        agent_obs = [None] * env.get_num_agents()
        # Reset environment
        reset_timer.start()
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        reset_timer.end()

        if train_params.render:
            env_renderer.set_new_rail()

        score = 0

        deadlocks = [False for agent in range(env.get_num_agents())]
        shortest_path = [obs.get(a)[6] if obs.get(a) is not None else 0 for a in range(env.get_num_agents())]

        # Run episode
        for step in range(max_steps):
            timestep += 1

            # Collect and preprocess observations
            for agent in env.get_agent_handles():
                # Agents always enter here at least once so there is no further controls
                # When obs is absent the agent has arrived and the observation remains the same
                if obs[agent]:
                    preproc_timer.start()
                    agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth,
                                                             observation_radius=observation_radius)
                    preproc_timer.end()

            # TODO try excluding completely arrived networks from changing policy
            action_dict = {a: ppo.policy_old.act(np.append(agent_obs[a], [a]), memory) if info['action_required'][a]
            else ppo.policy_old.act(np.append(agent_obs[a], [a]), memory, action=torch.tensor([0]))
                           for a in range(n_agents)}

            for a in list(action_dict.values()):
                action_count[a] += 1

            # Environment step
            """
            step_timer.start()
            obs, rewards, done, info = env.step(action_dict)
            step_timer.end()
            """

            obs, rewards, done, info, rewards_shaped, new_deadlocks, new_shortest_path = \
                step_shaping(env, action_dict, deadlocks, shortest_path)

            deadlocks = new_deadlocks
            shortest_path = new_shortest_path

            total_timestep_reward = np.sum(list(rewards.values()))
            score += total_timestep_reward
            total_timestep_reward_shaped = np.sum(list(rewards_shaped.values()))

            memory.rewards.append(total_timestep_reward_shaped)
            memory.dones.append(done['__all__'])

            # Set dones to True when the episode is finished because the maximum number of steps has been reached
            if step == max_steps - 1:
                memory.dones[-1] = True

            # Update
            if timestep % (horizon + 1) == 0:
                learn_timer.start()
                ppo.update(memory)
                learn_timer.end()

                """
                Set timestep to 1 because the batch includes an additional step which has not been considered in the 
                current trajectory (it has been inserted to compute the advantage) but must be considered in the next
                trajectory or is discarded.
                """
                memory.clear_memory_except_last()
                timestep = 1

            if train_params.render and episode % checkpoint_interval == 0:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )


        # Collection information about training
        tasks_finished = sum(info["status"][idx] == 2 or info["status"][idx] == 3 for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, n_agents)
        normalized_score = score / (max_steps * n_agents)
        action_probs = action_count / np.sum(action_count)
        action_count = [1] * action_size

        # Smoothed values for terminal display and for more stable hyper-parameter tuning
        smoothing = 0.99
        smoothed_normalized_score = smoothed_normalized_score * smoothing + normalized_score * (1.0 - smoothing)
        smoothed_completion = smoothed_completion * smoothing + completion * (1.0 - smoothing)

        # Print logs
        if episode % checkpoint_interval == 0:
            # TODO: Save network params as checkpoints
            print("..saving model..")
            # torch.save(ppo.policy.state_dict(), PATH)
            if train_params.render:
                env_renderer.close_window()

        print(
            '\rEpisode {}'
            '\tScore: {:.3f}'
            ' Avg: {:.3f}'
            '\tDone: {:.2f}%'
            ' Avg: {:.2f}%'
            '\tAction Probs: {}'.format(
                episode,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                format_action_prob(action_probs)
            ), end=" ")

        # TODO: Consider possible eval
        """
        if episode_idx % train_params.checkpoint_interval == 0:
            scores, completions, nb_steps_eval = eval_policy(env, policy, n_eval_episodes, max_steps)
            writer.add_scalar("evaluation/scores_min", np.min(scores), episode_idx)
            writer.add_scalar("evaluation/scores_max", np.max(scores), episode_idx)
            writer.add_scalar("evaluation/scores_mean", np.mean(scores), episode_idx)
            writer.add_scalar("evaluation/scores_std", np.std(scores), episode_idx)
            writer.add_histogram("evaluation/scores", np.array(scores), episode_idx)
            writer.add_scalar("evaluation/completions_min", np.min(completions), episode_idx)
            writer.add_scalar("evaluation/completions_max", np.max(completions), episode_idx)
            writer.add_scalar("evaluation/completions_mean", np.mean(completions), episode_idx)
            writer.add_scalar("evaluation/completions_std", np.std(completions), episode_idx)
            writer.add_histogram("evaluation/completions", np.array(completions), episode_idx)
            writer.add_scalar("evaluation/nb_steps_min", np.min(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_max", np.max(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_mean", np.mean(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_std", np.std(nb_steps_eval), episode_idx)
            writer.add_histogram("evaluation/nb_steps", np.array(nb_steps_eval), episode_idx)
            smoothing = 0.9
            smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(scores) * (1.0 - smoothing)
            smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)
            writer.add_scalar("evaluation/smoothed_score", smoothed_eval_normalized_score, episode_idx)
            writer.add_scalar("evaluation/smoothed_completion", smoothed_eval_completion, episode_idx)
        # Save logs to tensorboard
        writer.add_scalar("training/score", normalized_score, episode_idx)
        writer.add_scalar("training/smoothed_score", smoothed_normalized_score, episode_idx)
        writer.add_scalar("training/completion", np.mean(completion), episode_idx)
        writer.add_scalar("training/smoothed_completion", np.mean(smoothed_completion), episode_idx)
        writer.add_scalar("training/nb_steps", nb_steps, episode_idx)
        writer.add_histogram("actions/distribution", np.array(actions_taken), episode_idx)
        writer.add_scalar("actions/nothing", action_probs[RailEnvActions.DO_NOTHING], episode_idx)
        writer.add_scalar("actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode_idx)
        writer.add_scalar("actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode_idx)
        writer.add_scalar("actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode_idx)
        writer.add_scalar("actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode_idx)
        writer.add_scalar("training/epsilon", eps_start, episode_idx)
        writer.add_scalar("training/buffer_size", len(policy.memory), episode_idx)
        writer.add_scalar("training/loss", policy.loss, episode_idx)
        writer.add_scalar("timer/reset", reset_timer.get(), episode_idx)
        writer.add_scalar("timer/step", step_timer.get(), episode_idx)
        writer.add_scalar("timer/learn", learn_timer.get(), episode_idx)
        writer.add_scalar("timer/preproc", preproc_timer.get(), episode_idx)
        writer.add_scalar("timer/total", training_timer.get_current(), episode_idx)
        """


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


myseed = 19

environment_parameters = {
    # small_v0 config
    "n_agents": 3,
    "x_dim": 35,
    "y_dim": 35,
    "n_cities": 2,
    "max_rails_between_cities": 2,
    "max_rails_in_city": 3,

    "seed": myseed,
    "observation_tree_depth": 3,
    "observation_radius": 25,
    "observation_max_path_depth": 30
}

training_parameters = {
    "random_seed": myseed,
    # ====================
    # Network architecture
    # ====================
    # Shared actor-critic layer
    # If shared is True then the considered sizes are taken from the critic
    "shared": False,
    # Policy network
    "critic_mlp_width": 512,
    "critic_mlp_depth": 16,
    "last_critic_layer_scaling": 1.0,
    # Actor network
    "actor_mlp_width": 256,
    "actor_mlp_depth": 16,
    "last_actor_layer_scaling": 1.0,
    # Adam learning rate
    "learning_rate": 0.001,
    # Adam epsilon
    "adam_eps": 1e-5,
    # Activation
    "activation": "Tanh",
    "lmbda": 0.95,
    "entropy_coefficient": 0.01,
    # Called also baseline cost in shared setting (0.5)
    # (C54): {0.001, 0.1, 1.0, 10.0, 100.0}
    "value_loss_coefficient": 0.001,
    # ==============
    # Training setup
    # ==============
    "n_episodes": 2500,
    # 512, 1024, 2048, 4096
    "horizon": 4096,
    "epochs": 4,
    # Fixed trajectories, Shuffle trajectories, Shuffle transitions, Shuffle transitions (recompute advantages)
    # "batch_mode": None,
    # 64, 128, 256
    "batch_size": 1024,

    # ==========================
    # Normalization and clipping
    # ==========================
    # Discount factor (0.95, 0.97, 0.99, 0.999)
    "discount_factor": 0.99,

    # ====================
    # Advantage estimation
    # ====================
    # PPO-style value clipping
    "eps_clip": 0.25,
    # gae, n-steps
    "advantage_estimator": "gae",
    # huber or mse
    "value_loss_function": "mse",

    # ==========================
    # Optimization and rendering
    # ==========================
    "checkpoint_interval": 100,
    "use_gpu": False,
    "num_threads": 1,
    "render": True,
}

train_multiple_agents(Namespace(**environment_parameters), Namespace(**training_parameters))
