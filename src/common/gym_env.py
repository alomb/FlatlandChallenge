import gym
from flatland.utils.rendertools import RenderTool

from src.common.deadlocks import DeadlocksDetector
from src.common.observation import NormalizeObservations


class FlatlandGymEnv(gym.Env):
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
        obs, rewards, dones, info = self.rail_env.step(action_dict)

        # Compute deadlocks
        deadlocks = self.deadlocks_detector.step(self.rail_env)
        print(deadlocks)
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
        if self.render:
            return self.env_renderer.render_env(
                show=True,
                frames=False,
                show_observations=False,
                show_predictions=False)

    def close(self):
        if self.render:
            return self.env_renderer.close_window()
