import random
from argparse import Namespace

import numpy as np
from flatland.core.grid.grid4_utils import get_direction
from flatland.core.grid.grid_utils import distance_on_rail
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_rail_in_grid_map
from flatland.core.grid.grid4_utils import mirror
from flatland.envs.malfunction_generators import MalfunctionParameters
from flatland.envs.rail_generators import RailGenerator
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d

from src.curriculum.curriculum import Semi_Auto_Curriculum, offset_curriculum_generator, Manual_Curriculum
from src.psppo.ps_ppo_flatland import train_multiple_agents


def complex_rail_generator(curriculum) -> RailGenerator:
    def generator() -> RailGenerator:
        """
        Arguments are ignored and taken directly from the curriculum except the np_random: RandomState which is the last
        argument (args[-1])
        """

        if curriculum.get("n_agents") > curriculum.get("n_cities"):
            raise Exception("complex_rail_generator: n_agents > n_cities!")
        grid_map = GridTransitionMap(width=curriculum.get("x_dim"), height=curriculum.get("y_size"),
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

        while nr_created < curriculum.get("n_cities") and created_sanity < sanity_max:
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
        while nr_created < curriculum.get("n_extra") and created_sanity < sanity_max:
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


def curriculum_learning():
    num_levels = 70
    num_episodes_in_env = 20
    model_saving_number = 0

    """
    curriculum = Manual_Curriculum("curriculum.yml")
    """

    curriculum = Semi_Auto_Curriculum(offset_curriculum_generator(num_levels, {"x_dim": 30,
                                                                               "y_dim": 30,
                                                                               "n_agents": 1.7,
                                                                               "n_cities": 3,
                                                                               # "n_extra": 3,
                                                                               # "min_dist": 2,
                                                                               # "max_dist": 5
                                                                               "max_rails_between_cities": 1,
                                                                               "max_rails_in_city": 1,
                                                                               "malfunction_level": 0,
                                                                               "speed_level": 0}),
                                      num_levels)

    try:
        myseed = 19
        level = 0
        threshold = 0.57

        while True:
            if curriculum.get("speed_level") < 1:
                speed_profiles = {1.: 1.0, 1. / 2.: 0.0, 1. / 3.: 0.0, 1. / 4.: 0.0}
            elif curriculum.get("speed_level") < 2:
                speed_profiles = {1.: 0.75, 1. / 2.: 0.25, 1. / 3.: 0.0, 1. / 4.: 0.0}
            elif curriculum.get("speed_level") < 3:
                speed_profiles = {1.: 0.4, 1. / 2.: 0.3, 1. / 3.: 0.2, 1. / 4.: 0.1}
            else:
                speed_profiles = {1.: 0.25, 1. / 2.: 0.25, 1. / 3.: 0.25, 1. / 4.: 0.25}

            if curriculum.get("malfunction_level") < 1:
                malfunction_rate = 0.0
            elif curriculum.get("malfunction_level") < 2:
                malfunction_rate = 0.1
            elif curriculum.get("malfunction_level") < 3:
                malfunction_rate = 0.2
            else:
                malfunction_rate = 0.3

            print("=" * 100)
            print("Running the level {}".format(level))
            print("=" * 100)

            try_outs = 0

            print("{}x{} grid, {} agents, {} cities, {} rails between cities, {} rails in cities, malfunctions rate {} "
                  "and speed {}. Completion threshold of {}".format(curriculum.get("x_dim"),
                                                                    curriculum.get("y_dim"),
                                                                    curriculum.get("n_agents"),
                                                                    curriculum.get("n_cities"),
                                                                    curriculum.get("max_rails_between_cities"),
                                                                    curriculum.get("max_rails_in_city"),
                                                                    malfunction_rate,
                                                                    speed_profiles,
                                                                    threshold))

            completion = 0

            while try_outs < 10 and completion < threshold:
                namefile = "curriculum_{}".format(model_saving_number)

                environment_parameters = {
                    "n_agents": curriculum.get("n_agents"),
                    "x_dim": curriculum.get("x_dim"),
                    "y_dim": curriculum.get("y_dim"),
                    "n_cities": curriculum.get("n_cities"),
                    # Old curriculum
                    # "n_extra": curriculum.get("n_extra"),
                    # "min_dist": curriculum.get("min_dist"),
                    # "max_dist": curriculum.get("max_dist"),
                    "max_rails_between_cities": curriculum.get("max_rails_between_cities"),
                    "max_rails_in_city": curriculum.get("max_rails_in_city"),
                    "seed": myseed,
                    "observation_tree_depth": 2,
                    "observation_radius": 10,
                    "observation_max_path_depth": 30,

                    # Malfunctions
                    "malfunction_parameters": MalfunctionParameters(
                        malfunction_rate=malfunction_rate,
                        min_duration=15,
                        max_duration=50),
                    # Speeds
                    "speed_profiles": speed_profiles,

                    # ============================
                    # Custom observations&rewards
                    # ============================
                    "custom_observations": False,

                    "reward_shaping": True,
                    "uniform_reward": True,
                    "stop_penalty": -0.0,
                    "invalid_action_penalty": -0.0,
                    "deadlock_penalty": -15.0,
                    # 1.0 for skipping
                    "shortest_path_penalty_coefficient": 1 + 1 / 15,
                    "done_bonus": 1 / 15,
                }

                training_parameters = {
                    # ============================
                    # Network architecture
                    # ============================
                    # Shared actor-critic layer
                    # If shared is True then the considered sizes are taken from the critic
                    "shared": False,
                    "shared_recurrent": False,
                    "linear_size": 128,
                    "hidden_size": 64,
                    # Policy network
                    "critic_mlp_width": 128,
                    "critic_mlp_depth": 3,
                    "last_critic_layer_scaling": 0.1,
                    # Actor network
                    "actor_mlp_width": 128,
                    "actor_mlp_depth": 3,
                    "last_actor_layer_scaling": 0.01,
                    "learning_rate": 0.001,
                    "adam_eps": 1e-5,
                    "activation": "Tanh",
                    "lmbda": 0.95,
                    "entropy_coefficient": 0.01,
                    "value_loss_coefficient": 0.001,

                    # ============================
                    # Training setup
                    # ============================
                    "n_episodes": num_episodes_in_env,
                    "horizon": 512,
                    "epochs": 8,
                    "batch_size": 32,
                    "batch_mode": "shuffle",

                    # ============================
                    # Normalization and clipping
                    # ============================
                    "discount_factor": 0.99,
                    "max_grad_norm": 0.5,
                    "eps_clip": 0.2,

                    # ============================
                    # Advantage estimation
                    # ============================
                    "advantage_estimator": "gae",

                    # ============================
                    # Optimization and rendering
                    # ============================
                    "checkpoint_interval": num_episodes_in_env,
                    "evaluation_mode": False,
                    "eval_episodes": None,
                    "use_gpu": False,
                    "render": False,
                    "print_stats": True,
                    "save_model_path": "/content/drive/My Drive/Colab Notebooks/models/" + namefile + ".pt",
                    "load_model_path": "/content/drive/My Drive/Colab Notebooks/models/" + "curriculum_{}".format(
                        model_saving_number - 1) + ".pt",
                    "tensorboard_path": "log_" + namefile + "/",

                    # ============================
                    # Action Masking / Skipping
                    # ============================
                    "action_masking": True,
                    "allow_no_op": False
                }

                print("Level %d try out number % d" % (level, try_outs))
                # Train
                _, completions, _, _ = train_multiple_agents(Namespace(**environment_parameters),
                                                             Namespace(**training_parameters))
                try_outs += 1
                model_saving_number += 1
                completion = np.mean(completions)
            print("\n" + "=" * 100)

            # Update curriculum
            curriculum.update()
            level += 1
    except StopIteration:
        return


if __name__ == "__main__":
    curriculum_learning()
