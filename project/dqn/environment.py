import PIL
from IPython.core.display import display
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
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
        flatland_env: The Flatland environment
        renderer: The renderer
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
        new_observation: The new observation for each agent
        reward: The reward for each agent
        done: True if an agent has concluded
        info: Some info for each agent
    """

    def step(self, action):
        return self.flatland_env.step({0: action})

    """
    Reset the environment and return an observation
    Returns:
        observation: The new observation
    """

    def reset(self):
        observation, _ = self.flatland_env.reset(regenerate_rail=False,
                                                 regenerate_schedule=False,
                                                 random_seed=True)
        return observation
        # TODO: reset Render here or outside?

    """
        Render the environment
    """

    def render(self, mode='human'):
        # TODO: Merge both strategies (Jupyter vs .py)
        # In .py files
        # self.renderer.render_env(show=False, show_observations=False, show_predictions=False)
        # In Jupyter Notebooks
        env_renderer = RenderTool(self.flatland_env, gl="PILSVG")
        env_renderer.render_env()

        image = env_renderer.get_image()
        pil_image = PIL.Image.fromarray(image)
        display(pil_image)
        return image

    """
        Reset the renderer the environment
    """
    def reset_renderer(self):
        self.renderer = RenderTool(
            self.flatland_env,
            gl="PILSVG",
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
            show_debug=True,
            screen_height=700,
            screen_width=1300)

    def close_window(self):
        self.renderer.close_window()
