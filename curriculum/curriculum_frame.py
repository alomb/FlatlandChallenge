import time
from types import MappingProxyType

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


def complex_rail_generator(curriculum) -> RailGenerator:
    def generator(*args) -> RailGenerator:
        """
        Arguments are ignored and taken directly from the curriculum except the np_random: RandomState which is the last
        argument (args[-1])
        """

        if curriculum.get("num_agents") > curriculum.get("nr_start_goal"):
            raise Exception("complex_rail_generator: num_agents > nr_start_goal!")
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
        while nr_created < curriculum.get("nr_start_goal") and created_sanity < sanity_max:
            all_ok = False
            for _ in range(sanity_max):
                start = (args[-1].randint(0, curriculum.get("height")), args[-1].randint(0, curriculum.get("width")))
                goal = (args[-1].randint(0, curriculum.get("height")), args[-1].randint(0, curriculum.get("width")))

                # check to make sure start,goal pos is empty?
                if rail_array[goal] != 0 or rail_array[start] != 0:
                    continue
                # check min/max distance
                dist_sg = distance_on_rail(start, goal)
                if dist_sg < curriculum.get("min_dist"):
                    continue
                if dist_sg > curriculum.get("max_dist"):
                    continue
                # check distance to existing points
                sg_new = [start, goal]

                def check_all_dist(sg_new):
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

                if check_all_dist(sg_new):
                    all_ok = True
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
        while nr_created < curriculum.get("nr_extra") and created_sanity < sanity_max:
            all_ok = False
            for _ in range(sanity_max):
                start = (args[-1].randint(0, curriculum.get("height")), args[-1].randint(0, curriculum.get("width")))
                goal = (args[-1].randint(0, curriculum.get("height")), args[-1].randint(0, curriculum.get("width")))
                # check to make sure start,goal pos are not empty
                if rail_array[goal] == 0 or rail_array[start] == 0:
                    continue
                else:
                    all_ok = True
                    break
            if not all_ok:
                break
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

    def update(self, rewards):
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

    def update(self, rewards):
        # Check rewards

        # Increase level
        self.level += 1

    def get(self, attribute):
        return self.curriculum["curriculum"][self.level][attribute]


def curriculum_generator(num_levels,
                         initial_values,
                         offset=MappingProxyType({"width": 0.15,
                                                  "height": 0.15,
                                                  "num_agents": 0.1,
                                                  "nr_start_goal": 0.1,
                                                  "nr_extra": 0.15,
                                                  "min_dist": 0.3,
                                                  "max_dist": 0.3,
                                                  }),
                         noise=None):
    for k in initial_values:
        initial_values[k] = float(initial_values[k])

    for _ in range(num_levels):
        for k, v in initial_values.items():
            initial_values[k] = v + offset[k]
        yield initial_values


class Semi_Auto_Curriculum(Curriculum):
    def __init__(self, generator, num_levels, level=0):
        super(Semi_Auto_Curriculum, self).__init__(level=level)
        self.generator = generator
        self.values = next(generator)
        self.num_levels = num_levels

    def update(self, rewards):
        """

        :param rewards:
        :return:

        :raise StopIteration
        """
        # Check rewards
        # Increase level in the iterator
        self.values = next(self.generator)

    def get(self, attribute):
        return int(self.values[attribute])


render = True

observation_max_path_depth = 10
observation_tree_depth = 1
n_agents = 1
myseed = None

width = 10
height = 10

# Break agents from time to time
malfunction_parameters = MalfunctionParameters(
    # Rate of malfunctions
    malfunction_rate=0.0,
    # Minimal duration
    min_duration=15,
    # Max duration
    max_duration=50
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

my_num_levels = 50

"""
mycurriculum = Manual_Curriculum("curriculum/curriculum.yml")
"""
mycurriculum = Semi_Auto_Curriculum(curriculum_generator(my_num_levels, {"width": 6,
                                                                         "height": 6,
                                                                         "num_agents": 1,
                                                                         "nr_start_goal": 1,
                                                                         "nr_extra": 0,
                                                                         "min_dist": 2,
                                                                         "max_dist": 6}), my_num_levels)

for _ in range(30):
    # Create environment
    env = RailEnv(
        width=mycurriculum.get("width"),
        height=mycurriculum.get("height"),
        rail_generator=complex_rail_generator(mycurriculum),
        schedule_generator=complex_schedule_generator(speed_profiles),
        number_of_agents=mycurriculum.get("num_agents"),
        malfunction_generator_and_process_data=malfunction_from_params(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=myseed)

    env.reset(regenerate_rail=True, regenerate_schedule=True)

    # Step
    # Update curriculum
    mycurriculum.update(None)

    # Rendering
    if render:
        env_renderer = RenderTool(env, gl="PGL")
        env_renderer.set_new_rail()
        env_renderer.render_env(show=True, frames=False, show_observations=False, show_predictions=False)
        time.sleep(2)
        env_renderer.close_window()
