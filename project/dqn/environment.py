from gym import Env
from gym.spaces import Discrete

"""
An OpenAI gym environment that wraps a Flatland one
"""


class SingleAgentEnvironment(Env):
    flatland_env = None
    renderer = None

    """
    Args:
        flatland_env:
        renderer: 
    """
    def __init__(self, flatland_env, renderer=None):
        self.flatland_env = flatland_env
        self.renderer = renderer

        self.reward_range = (-1, 1)
        self.action_space = Discrete(5)
        self.observation_space = Discrete(11)

    """
    Execute an action.
    Args:
        action: the action index to perform
    Return:
        new_observation:
        reward:
        done:
        info:
    """
    def step(self, action):
        new_observation, reward, done, info = self.flatland_env.step({0: action})
        return new_observation, reward, done, info

    """
    Reset the environment and return an observation
    Returns:
        observation:
    """
    def reset(self):
        observation, _ = self.flatland_env.reset(regenerate_rail=False,
                                                 regenerate_schedule=False,
                                                 random_seed=True)
        return observation

    # TODO: reset Render

    """
    """
    def render(self, mode='human'):
        # TODO:
        raise NotImplementedError
