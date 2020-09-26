import gym

from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

from src.common.deadlocks import DeadlocksDetector
from src.common.observation import NormalizeObservations
from src.common.wrappers import RewardsWrapper, StatsWrapper


class FlatlandRailEnv:
    """
    Flatland environment to deal with wrappers.
    """
    def __init__(self,
                 train_params,
                 env_params,
                 observation,
                 custom_observations,
                 reward_wrapper,
                 stats_wrapper):

        self.env = FlatlandGymEnv(self._launch(env_params, observation),
                                   custom_observations,
                                   env_params,
                                   render=train_params.render)

        self.state_size = self.env.state_size

        if reward_wrapper:
            self.env = RewardsWrapper(self.env,
                                       env_params.invalid_action_penalty,
                                       env_params.stop_penalty,
                                       env_params.deadlock_penalty,
                                       env_params.shortest_path_penalty_coefficient,
                                       env_params.done_bonus,
                                      env_params.uniform_reward)
        if stats_wrapper:
            self.env = StatsWrapper(self.env,
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
        return self.env.reset()

    def step(self, action_dict):
        return self.env.step(action_dict)

    def get_rail_env(self):
        return self.env.rail_env


class FlatlandGymEnv(gym.Env):
    """
    gym.Env wrapper of the Flatland environment providing deadlocks and observation normalization.
    """
    def __init__(self,
                 rail_env,
                 custom_observations,
                 env_params,
                 render=False,
                 regenerate_rail_on_reset=True,
                 regenerate_schedule_on_reset=True):

        self._regenerate_rail_on_reset = regenerate_rail_on_reset
        self._regenerate_schedule_on_reset = regenerate_schedule_on_reset
        self.rail_env = rail_env
        self.deadlocks_detector = DeadlocksDetector()

        self.observation_normalizer = NormalizeObservations(self.rail_env.obs_builder.observation_dim,
                                                            env_params.observation_tree_depth,
                                                            custom_observations,
                                                            self.rail_env.width,
                                                            self.rail_env.height,
                                                            env_params.observation_radius)

        self.state_size = self.observation_normalizer.state_size

        self.render = render
        self.env_renderer = None

    def reset(self):
        obs, info = self.rail_env.reset(regenerate_rail=self._regenerate_rail_on_reset,
                                        regenerate_schedule=self._regenerate_schedule_on_reset)
        # Reset rendering
        if self.render:
            self.env_renderer = RenderTool(self.rail_env, gl="PGL")
            self.env_renderer.set_new_rail()

        # Reset custom observations
        self.observation_normalizer.reset_custom_obs(self.rail_env)

        # Compute deadlocks
        self.deadlocks_detector.reset(self.rail_env.get_num_agents())
        info["deadlocks"] = {}

        for agent in range(self.rail_env.get_num_agents()):
            info["deadlocks"][agent] = self.deadlocks_detector.deadlocks[agent]

        # Normalization
        for agent in obs:
            if obs[agent] is not None:
                obs[agent] = self.observation_normalizer.normalize_observation(obs[agent], self.rail_env,
                                                                               agent, info["deadlocks"][agent])

        return obs, info

    def step(self, action_dict):
        """
        Normalize observations by default, update deadlocks and step.

        :param action_dict:
        :return:
        """
        obs, rewards, dones, info = self.rail_env.step(action_dict)

        # Compute deadlocks
        deadlocks = self.deadlocks_detector.step(self.rail_env)
        info["deadlocks"] = {}
        for agent in range(self.rail_env.get_num_agents()):
            info["deadlocks"][agent] = deadlocks[agent]

        # Normalization
        for agent in obs:
            if obs[agent] is not None:
                obs[agent] = self.observation_normalizer.normalize_observation(obs[agent], self.rail_env,
                                                                               agent, info["deadlocks"][agent])

        return obs, rewards, dones, info

    def show_render(self):
        """
        Open rendering window.

        :return:
        """
        if self.render:
            return self.env_renderer.render_env(
                show=True,
                frames=False,
                show_observations=False,
                show_predictions=False)

    def close(self):
        """
        Close rendering window.
        :return:
        """
        if self.render:
            return self.env_renderer.close_window()
