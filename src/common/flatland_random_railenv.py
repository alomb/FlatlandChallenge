from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from src.common.gym_env import FlatlandGymEnv
from src.common.rewards_wrapper import RewardsWrapper
from src.common.stats_wrapper import StatsWrapper


class FlatlandRandomRailEnv:
    def __init__(self,
                 train_params,
                 env_params,
                 observation):

        self._env = FlatlandGymEnv(
            rail_env=self._launch(env_params, observation),
            render=train_params.render
        )

        self._env = RewardsWrapper(self._env,
                                   env_params.invalid_action_penalty,
                                   env_params.stop_penalty,
                                   env_params.deadlock_penalty,
                                   env_params.shortest_path_penalty_coefficient,
                                   env_params.done_bonus)

        self._env = StatsWrapper(self._env,
                                 env_params)

    def _launch(self, env_params, observation):
        return RailEnv(
            width=env_params.x_dim,
            height=env_params.y_dim,
            rail_generator=sparse_rail_generator(
                max_num_cities=env_params.n_cities,
                grid_mode=False,
                max_rails_between_cities=env_params.max_rails_between_cities,
                max_rails_in_city=env_params.max_rails_in_city,
                seed=env_params.seed
            ),
            schedule_generator=sparse_schedule_generator(env_params.speed_profiles),
            number_of_agents=env_params.n_agents,
            malfunction_generator_and_process_data=malfunction_from_params(env_params.malfunction_parameters),
            obs_builder_object=observation,
            random_seed=env_params.seed
        )

    def reset(self):
        return self._env.reset()

    def step(self, action_dict):
        return self._env.step(action_dict)

    def get_rail_env(self):
        return self._env.rail_env
