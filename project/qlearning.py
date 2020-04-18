from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.core.env_observation_builder import ObservationBuilder
import numpy as np
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
import matplotlib.pyplot as plt


class SingleAgentNavigationObs(ObservationBuilder):
    """
    An observation is a dictionary:
    {"observations": List[List[int]],
    "position": tuple}

    e.g. position = (12, 4) and observations = [[1 0 0], [0 1 0]] means: I'm in the position (12, 4) in the grid and
    I can perform two actions:
    - Move left;
    - Move forward
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get(self, handle: int = 0):
        agent = self.env.agents[handle]
        observations = []

        if agent.position:
            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
            position = agent.position
        else:
            possible_transitions = self.env.rail.get_transitions(*agent.initial_position, agent.direction)
            position = agent.initial_position

        num_transitions = np.count_nonzero(possible_transitions)

        if num_transitions == 1:
            observations.append([0, 1, 0])
        else:
            i = 0
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    observation = [0, 0, 0]
                    observation[i] = 1
                    observations.append(observation)
                i = i + 1

        return {"observations": observations, "state": position}


number_of_agents = 1                                                                    # In this script I can perform
action_to_direction = {0: 'no-op', 1: 'left', 2: 'forward', 3: 'right', 4: 'halt'}      # only 1, 2 and 3
LEARNING_RATE = 0.1     # Exploitation
DISCOUNT = 0.95         # Exploration
EPISODES = 50
SHOW_EVERY = 1          # Show the animation every 5 episodes
WIDTH = 40              # Number of rows of the environment
HEIGHT = 40             # Number of columns of the environment
rewards = []            # List for plotting the results

q_table = np.random.uniform(low=-2, high=0, size=(WIDTH, HEIGHT, len(action_to_direction)))     # q tensor

for episode in range(EPISODES):

    if episode % SHOW_EVERY == 0:
        render = True                               # This episode will be showed
        print("Show episode number: ", episode)
    else:
        render = False

    sparse_env = RailEnv(
        width=WIDTH,
        height=HEIGHT,
        rail_generator=sparse_rail_generator(
            max_num_cities=3,  # Number of cities (= train stations)
            grid_mode=False,  # Distribute the cities evenly in a grid
            max_rails_between_cities=1,  # Max number of rails connecting to a city
            max_rails_in_city=1  # Number of parallel tracks in cities
        ),
        obs_builder_object=SingleAgentNavigationObs(),
        number_of_agents=number_of_agents
    )

    env_renderer = RenderTool(sparse_env, gl="PILSVG",
                              agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                              show_debug=True,
                              screen_height=700,
                              screen_width=1300)

    observation, info = sparse_env.reset()          # e.g. of observation {0: {"state": (12, 4), observations: [[1 0 0],
    state = observation[0]["state"]                 # [0 1 0]]}}
    obs = observation[0]["observations"]
    done = {0: False}
    cost = 0                                        # cost of the path found (for plotting the results)

    while not done[0]:
        q_best = -np.inf                            # searching in the Q tensor the most promising action given the
        for single_obs in obs:                      # admissible actions (this information is stored in the "observations")
            index = np.argmax(single_obs)
            if q_table[state + (index,)] > q_best:
                q_best = q_table[state + (index,)]
                action = index + 1                  # action is setted to index + 1 because in sparse_env_step are represented
                                                    # in this way

        observation, all_rewards, done, _ = sparse_env.step({0: action})
        new_state = observation[0]["state"]
        obs = observation[0]["observations"]
        reward = all_rewards[0]
        cost += reward

        if render:
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        if not done[0]:                                     # https://it.wikipedia.org/wiki/Q-learning (view Q-formula)
            q_best = -np.inf
            for single_obs in obs:
                index = np.argmax(single_obs)
                if q_table[new_state + (index,)] > q_best:
                    q_best = q_table[new_state + (index,)]
                    new_action = index + 1
            current_q = q_table[state + (action - 1, )]
            max_future_q = q_table[new_state + (new_action - 1, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[state + (action - 1, )] = new_q
        else:
            q_table[state + (action - 1, )] = 0

        state = new_state

    if render:
        env_renderer.close_window()

    rewards.append(cost)

episodes = np.linspace(1, EPISODES, EPISODES)

fig, ax = plt.subplots()
plt.title("Total rewards vs episodes")
plt.plot(episodes, np.abs(rewards))
ax.set_xlabel("Episode")
ax.set_ylabel("Cost (total rewards)")
plt.show()




