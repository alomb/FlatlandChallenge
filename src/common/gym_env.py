from abc import ABC
from plistlib import Dict

import gym
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.utils.rendertools import RenderTool

from src.common.deadlocks import DeadlocksDetector


class FlatlandGymEnv(gym.Env, ABC):
    def __init__(self,
                 rail_env: RailEnv,
                 render: bool = False,
                 regenerate_rail_on_reset: bool = True,
                 regenerate_schedule_on_reset: bool = True,
                 ) -> None:

        self._regenerate_rail_on_reset = regenerate_rail_on_reset
        self._regenerate_schedule_on_reset = regenerate_schedule_on_reset
        self.rail_env = rail_env
        self.deadlocks_detector = DeadlocksDetector()
        self.render = render
        if render:
            self.env_renderer = RenderTool(self.rail_env, gl="PGL").set_new_rail()

    def reset(self):
        obs, info = self.rail_env.reset(regenerate_rail=self._regenerate_rail_on_reset,
                                         regenerate_schedule=self._regenerate_schedule_on_reset)
        self.deadlocks_detector.reset(self.rail_env.get_num_agents())
        info['deadlocks'] = {}

        for agent in range(self.rail_env.get_num_agents()):
            info['deadlocks'][agent] = self.deadlocks_detector.deadlocks[agent]

        return obs, info

    def step(self, action_dict):
        obs, rewards, dones, info = self.rail_env.step(action_dict)
        deadlocks = self.deadlocks_detector.step(self.rail_env, action_dict, dones)

        info["deadlocks"] = {}
        standard_rewards = {agent: {} for agent in range(self.rail_env.get_num_agents())}

        for agent in range(self.rail_env.get_num_agents()):
            info["deadlocks"][agent] = deadlocks[agent]
            standard_rewards[agent]["standard_rewards"] = rewards[agent]

        return obs, rewards, dones, info

    def show_render(self):
        if self.render:
            return self.env_renderer.render_env(
                show=True,
                frames=False,
                show_observations=False,
                show_predictions=False
            )

    def close(self):
        if self.render:
            return self.env_renderer.close_window()