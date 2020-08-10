import random
from argparse import Namespace
from types import MappingProxyType

import torch
import numpy as np
from flatland.core.grid.grid4_utils import get_direction
from flatland.core.grid.grid_utils import distance_on_rail
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_rail_in_grid_map
from flatland.core.grid.grid4_utils import mirror
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import RailGenerator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d

from ps_ppo.algorithm import PsPPO, Memory
from ps_ppo.observation_parsing import normalize_observation
from ps_ppo.reward_shaping import step_shaping
from ps_ppo.timer import Timer


def complex_rail_generator(curriculum) -> RailGenerator:
    def generator(*args) -> RailGenerator:
        """
        Arguments are ignored and taken directly from the curriculum except the np_random: RandomState which is the last
        argument (args[-1])
        """

        if curriculum.get("num_agents") > curriculum.get("num_start_goal"):
            raise Exception("complex_rail_generator: num_agents > num_start_goal!")
        grid_map = GridTransitionMap(width=curriculum.get("width"), height=curriculum.get("height"),
                                     transitions=RailEnvTransitions())
        rail_array = grid_map.grid
        rail_array.fill(0)

        # generate rail array
        # step 1:
        # - generate a start and goal position
        #   - validate min/max distance allowed
        #   - validate that start/goals are not placed too close to other start/goals
        #   - draw a rail from [start,goal]
        #     - if rail crosses existing rail then validate new connection
        #     - possibility that this fails to create a path to goal
        #     - on failure generate new start/goal
        #
        # step 2:
        # - add more rails to map randomly between cells that have rails
        #   - validate all new rails, on failure don't add new rails
        #
        # step 3:
        # - return transition map + list of [start_pos, start_dir, goal_pos] points
        #

        rail_trans = grid_map.transitions
        start_goal = []
        start_dir = []
        nr_created = 0
        created_sanity = 0
        sanity_max = 9000

        free_cells = set([(r, c) for r, row in enumerate(rail_array) for c, col in enumerate(row) if col == 0])

        while nr_created < curriculum.get("num_start_goal") and created_sanity < sanity_max:
            all_ok = False

            if len(free_cells) == 0:
                break

            for _ in range(sanity_max):
                start = random.sample(free_cells, 1)[0]
                goal = random.sample(free_cells, 1)[0]

                # check min/max distance
                dist_sg = distance_on_rail(start, goal)
                if dist_sg < curriculum.get("min_dist"):
                    continue
                if dist_sg > curriculum.get("max_dist"):
                    continue
                # check distance to existing points
                sg_new = [start, goal]

                def check_all_dist():
                    """
                    Function to check the distance betweens start and goal
                    :param sg_new: start and goal tuple
                    :return: True if distance is larger than 2, False otherwise
                    """
                    for sg in start_goal:
                        for i in range(2):
                            for j in range(2):
                                dist = distance_on_rail(sg_new[i], sg[j])
                                if dist < 2:
                                    return False
                    return True

                if check_all_dist():
                    all_ok = True
                    free_cells.remove(start)
                    free_cells.remove(goal)
                    break

            if not all_ok:
                # we might as well give up at this point
                break

            new_path = connect_rail_in_grid_map(grid_map, start, goal, rail_trans, Vec2d.get_chebyshev_distance,
                                                flip_start_node_trans=True, flip_end_node_trans=True,
                                                respect_transition_validity=True, forbidden_cells=None)
            if len(new_path) >= 2:
                nr_created += 1
                start_goal.append([start, goal])
                start_dir.append(mirror(get_direction(new_path[0], new_path[1])))
            else:
                # after too many failures we will give up
                created_sanity += 1

        # add extra connections between existing rail
        created_sanity = 0
        nr_created = 0
        while nr_created < curriculum.get("num_extra") and created_sanity < sanity_max:
            if len(free_cells) == 0:
                break

            for _ in range(sanity_max):
                start = random.sample(free_cells, 1)[0]
                goal = random.sample(free_cells, 1)[0]

            new_path = connect_rail_in_grid_map(grid_map, start, goal, rail_trans, Vec2d.get_chebyshev_distance,
                                                flip_start_node_trans=True, flip_end_node_trans=True,
                                                respect_transition_validity=True, forbidden_cells=None)

            if len(new_path) >= 2:
                nr_created += 1
            else:
                # after too many failures we will give up
                created_sanity += 1

        return grid_map, {'agents_hints': {
            'start_goal': start_goal,
            'start_dir': start_dir
        }}

    return generator


class Curriculum:
    def __init__(self, level=0):
        self.level = level

    def update(self):
        raise NotImplementedError()

    def get(self, attribute):
        raise NotImplementedError()


class Manual_Curriculum(Curriculum):
    def __init__(self, configuration_file, level=0):
        super(Manual_Curriculum, self).__init__(level=level)
        self.file = configuration_file
        import yaml
        with open(configuration_file) as f:
            self.curriculum = yaml.safe_load(f)
        print("Manual curriculum: ", self.curriculum)
        self.num_levels = len(self.curriculum["curriculum"])

    def update(self):
        # Check rewards

        # Increase level
        self.level += 1

    def get(self, attribute):
        return self.curriculum["curriculum"][self.level][attribute]


def offset_curriculum_generator(num_levels,
                                initial_values,
                                offset=MappingProxyType({"width": lambda lvl: 0.2,
                                                         "height": lambda lvl: 0.2,
                                                         "num_agents": lambda lvl: 0.15,
                                                         "num_start_goal": lambda lvl: 0.15,
                                                         "num_extra": lambda lvl: 0.2,
                                                         "min_dist": lambda lvl: 0.3,
                                                         "max_dist": lambda lvl: 0.3}),
                                forget_every=None,
                                forget_intensity=None,
                                checkpoint_every=15,
                                checkpoint_recovery_levels=4):
    for k in initial_values:
        initial_values[k] = float(initial_values[k])

    checkpoint = initial_values
    checkpoint_counter = 0

    for level in range(1, num_levels + 1):

        if checkpoint_every is not None and checkpoint_recovery_levels is not None:
            # Save checkpoint
            if level % checkpoint_every and not checkpoint_counter > 0:
                checkpoint, initial_values = initial_values, checkpoint
                checkpoint_counter = checkpoint_recovery_levels

            # Solve checkpoints
            if checkpoint_counter > 0:
                initial_values = {k: (initial_values[k] + offset[k](level)) for k, v in initial_values.items()}
                checkpoint_counter -= 1

        for k, v in initial_values.items():
            if forget_every is not None and level % forget_every == 0 and forget_intensity is not None:
                initial_values[k] = v + offset[k](level) - forget_intensity
            else:
                initial_values[k] = v + offset[k](level)
        yield initial_values


class Semi_Auto_Curriculum(Curriculum):
    def __init__(self, generator, num_levels, level=0):
        super(Semi_Auto_Curriculum, self).__init__(level=level)
        self.generator = generator
        self.values = next(generator)
        self.num_levels = num_levels

    def update(self):
        """

        :return:

        :raise StopIteration
        """
        # Check rewards
        # Increase level in the iterator
        self.values = next(self.generator)

    def get(self, attribute):
        return int(self.values[attribute])


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def pretrain_multiple_agents(env_params, train_params):
    # Environment parameters
    n_agents = env_params.num_agents
    x_dim = env_params.width
    y_dim = env_params.height
    seed = env_params.seed
    num_start_goal = env_params.num_start_goal

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
    if seed is not None:
        torch.manual_seed(seed)

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=0,  # Rate of malfunctions
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

    mycurriculum = train_params.curriculum

    # Setup the environment
    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=complex_rail_generator(mycurriculum),
        schedule_generator=complex_schedule_generator(speed_profiles),
        number_of_agents=n_agents,
        malfunction_generator_and_process_data=malfunction_from_params(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed
    )

    env.reset(regenerate_schedule=True, regenerate_rail=True)

    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes

    # The action space of flatland is 5 discrete actions
    action_size = env.action_space[0]

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    max_steps = int(4 * 2 * (env.height + env.width + (n_agents / num_start_goal)))

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
                train_params.value_loss_coefficient,
                train_params.load_model_path)

    training_timer = Timer()
    training_timer.start()

    print("\nTraining {} trains on {}x{} grid for {} episodes. Update every {} timesteps.\n"
          .format(env.get_num_agents(), x_dim, y_dim, n_episodes, horizon))

    timestep = 0

    for episode in range(n_episodes + 1):

        # Setup renderer
        if train_params.render:
            env_renderer = RenderTool(env, gl="PGL")
        else:
            env_renderer = None

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

        deadlocks = [False for _ in range(env.get_num_agents())]
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

            action_dict = {a: ppo.policy_old.act(np.append(agent_obs[a], [a]), memory) if info['action_required'][a]
            else ppo.policy_old.act(np.append(agent_obs[a], [a]), memory, action=torch.tensor([0]))
                           for a in range(n_agents)}

            for a in list(action_dict.values()):
                action_count[a] += 1

            # Environment step
            step_timer.start()
            obs, rewards, done, info, rewards_shaped, new_deadlocks, new_shortest_path = \
                step_shaping(env, action_dict, deadlocks, shortest_path)
            step_timer.end()

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

    if train_params.save_model_path is not None:
        torch.save(ppo.policy.state_dict(), train_params.save_model_path)

    return completion


def curriculum_learning():
    myseed = None
    my_num_levels = 50
    """
    mycurriculum = Manual_Curriculum("curriculum/curriculum.yml")
    """

    mycurriculum = Semi_Auto_Curriculum(offset_curriculum_generator(my_num_levels, {"width": 8,
                                                                                    "height": 8,
                                                                                    "num_agents": 1,
                                                                                    "num_start_goal": 1,
                                                                                    "num_extra": 3,
                                                                                    "min_dist": 2,
                                                                                    "max_dist": 5}),
                                        my_num_levels)

    for level in range(my_num_levels):
        environment_parameters = {
            "num_agents": mycurriculum.get("num_agents"),
            "width": mycurriculum.get("width"),
            "height": mycurriculum.get("height"),
            "num_start_goal": mycurriculum.get("num_start_goal"),
            "num_extra": mycurriculum.get("num_extra"),
            "min_dist": mycurriculum.get("min_dist"),
            "max_dist": mycurriculum.get("max_dist"),

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
            "critic_mlp_width": 128,
            "critic_mlp_depth": 2,
            "last_critic_layer_scaling": 1.0,
            # Actor network
            "actor_mlp_width": 128,
            "actor_mlp_depth": 2,
            "last_actor_layer_scaling": 0.001,
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
            "n_episodes": 5,
            # 512, 1024, 2048, 4096
            "horizon": mycurriculum.get("max_dist"),
            "epochs": 2,
            # Fixed trajectories, Shuffle trajectories, Shuffle transitions, Shuffle transitions (recompute advantages)
            # "batch_mode": None,
            # 64, 128, 256
            "batch_size": mycurriculum.get("max_dist"),

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
            "checkpoint_interval": 1,
            "use_gpu": False,
            "num_threads": 1,
            "render": False,
            "save_model_path": "curr.pt",
            "load_model_path": "curr.pt",

            # ==========================
            # Curriculum
            # ==========================
            "curriculum": mycurriculum
        }

        try_outs = 0
        threshold = 0.5
        tests_counter = 2

        print("=" * 10)
        while tests_counter > 0 and try_outs < 10:
            print("Level %d try out number % d" % (level, try_outs))
            # Train
            done_percentage = pretrain_multiple_agents(Namespace(**environment_parameters),
                                                       Namespace(**training_parameters))
            try_outs += 1
            tests_counter -= 1 if done_percentage > threshold else 0

        # Update curriculum
        mycurriculum.update()


curriculum_learning()
