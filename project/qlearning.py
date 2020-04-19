import matplotlib.pyplot as plt
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from obs.single import SingleAgentNavigationObs 

number_of_agents = 1
random_seed = 47                                                                  
# In this script I can perform only 1, 2 and 3  
action_to_direction = {0: 'no-op', 1: 'left', 2: 'forward', 3: 'right', 4: 'halt'}
# Exploitation
LEARNING_RATE = 0.1
# Exploration
DISCOUNT = 0.95
EPISODES = 50
# Show the animation every 5 episodes
SHOW_EVERY = 10
# Number of rows of the environment
WIDTH = 40
# Number of columns of the environment
HEIGHT = 40
# List for plotting the results
rewards = []

q_table = np.random.uniform(low=-2, high=0, size=(WIDTH, HEIGHT, len(action_to_direction)))     # q tensor



sparse_env = RailEnv(
    width=WIDTH,
    height=HEIGHT,
    rail_generator=sparse_rail_generator(
        # Number of cities (= train stations)
        max_num_cities=3,
        # Distribute the cities evenly in a grid
        grid_mode=False,
        # Max number of rails connecting to a city
        max_rails_between_cities=1,
        # Number of parallel tracks in cities
        max_rails_in_city=1),
    obs_builder_object=SingleAgentNavigationObs(),
    number_of_agents=number_of_agents)

observation, info = sparse_env.reset(
    regenerate_rail=False,
    # Change the final station
    regenerate_schedule=False,
    random_seed=random_seed)

for episode in range(EPISODES):

    if episode % SHOW_EVERY == 0:
        # This episode will be showed
        render = True
        print("Show episode number: ", episode)
    else:
        render = False

    env_renderer = RenderTool(
        sparse_env, 
        gl="PILSVG",
        agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
        show_debug=True,
        screen_height=700,
        screen_width=1300)

    observation, info = sparse_env.reset(
        regenerate_rail=False,
        regenerate_schedule=False,
        random_seed=random_seed)

    state = observation[0]["state"]                         
    # e.g. of observation {0: {"state": (12, 4), observations: [[1 0 0], [0 1 0]]}}
    obs = observation[0]["observations"]
    done = {0: False}
    cost = 0                                                
    # cost of the path found (for plotting the results)

    while not done[0]:
        q_best = -np.inf                                    
        # searching in the Q tensor the most promising action given the
        for single_obs in obs:                              
            # admissible actions (this information is stored in the "observations")
            index = np.argmax(single_obs)
            if q_table[state + (index,)] > q_best:
                q_best = q_table[state + (index,)]
                action = index + 1                          
                # action is setted to index + 1 because in sparse_env_step are represented in this way

        observation, all_rewards, done, _ = sparse_env.step({0: action})
        new_state = observation[0]["state"]
        obs = observation[0]["observations"]
        reward = all_rewards[0]
        cost += reward

        if render:
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        # https://it.wikipedia.org/wiki/Q-learning (view Q-formula)
        if not done[0]:
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




