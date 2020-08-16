import numpy as np
import matplotlib.pyplot as plt

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.rail_env_shortest_paths import get_shortest_paths
from tensorflow.python.keras.optimizer_v2.adam import Adam

from single.dqn.agent import SingleDQNAgent
from single.dqn.environment import SingleAgentEnvironment
from single.observations import SingleDQNAgentObs

"""
    Execution of the Deep Q-Learning algorithm for a single agent navigation
"""

# Render the environment
render = True
renderer = None
# Print stats within the episode
print_stats = True
# Print stats at the end of each episode
print_episode_stats = True
# Frequency of episodes to print
print_episode_stats_freq = 1

random_seed = 1
np.random.seed(random_seed)

action_to_direction = {0: 'no-op', 1: 'left', 2: 'forward', 3: 'right', 4: 'halt'}

WIDTH = 30
HEIGHT = 30

EPISODES = 5
# TODO: Change maximum timesteps (dependent to map size)
TIMESTEPS = 120

BATCH_SIZE = 10
UPDATE_TARGET_NETWORK = 5

env = RailEnv(
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
        max_rails_in_city=1,
        seed=random_seed),
    obs_builder_object=SingleDQNAgentObs(),
    number_of_agents=1,
    random_seed=random_seed)


environment = SingleAgentEnvironment(env)
agents = [SingleDQNAgent(environment, Adam(lr=0.01)) for i in range(env.number_of_agents)]

agents[0].q_network.summary()

"""
Transform observation dictionary to neural network input (numpy column)
Args:
    observation: the observation to change
"""


def reshape_observation(observations):
    for a in range(env.number_of_agents):
        observations[a]["possible_directions"].extend([observations[a]["position"][0], observations[a]["position"][1]])
        observations[a] = np.array(observations[a]["possible_directions"]).reshape((-1, len(observations[a]["possible_directions"])))

    return observations


# Dictionary agent -> action used in step
action_dict = dict()

# Stats for each episode
stats = []
shortest_paths_rewards = []

for episode in range(0, EPISODES):
    # Reset the environment
    old_observations, info = environment.reset()
    print(str(old_observations))
    old_observations = reshape_observation(old_observations)

    # Reset the renderer
    if render:
        env_renderer = RenderTool(env, gl="PGL")
        env_renderer.set_new_rail()

    # Shortest path = number of intermediate states = number of states - 2 (excluding the first and the last one)
    shortest_paths_rewards.append(-(len(get_shortest_paths(env.distance_map, max_depth=25, agent_handle=0)[0])-2))

    # Initialize variables
    episode_reward = 0
    terminated = False

    # Episode stats
    action_counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for time_step in range(TIMESTEPS):
        print(shortest_paths_rewards)
        if print_stats:
            print("Episode " + str(time_step) + " in episode " + str(episode + 1))

        # Initially False, remains False if no agent updates it
        update_values = False

        # Choose actions
        for a in range(env.number_of_agents):
            if info["action_required"][a]:
                update_values = True
                action = agents[a].act(old_observations[a])
                action_counter[action] += 1
            else:
                action = 0
            if print_stats:
                print("Agent " + str(a) + " performs action: " + str(action))
            action_dict.update({a: action})

        # Apply the chosen actions
        new_observations, reward, terminated, info = environment.step(action_dict)

        if print_stats:
            print("Step (obs, reward, terminated, info): ")
            print(new_observations)
            print(reward)
            print(terminated)
            print(info)
            print("_______________________________")

        for a in range(env.number_of_agents):
            # Episode reward is the mean
            episode_reward += reward[a] / env.number_of_agents

            if update_values or terminated[a]:
                # Reshape the observations to feed the network
                new_observations = reshape_observation(new_observations)

                # Store S A R S' for each agent
                agents[a].store(old_observations[a], action_dict[a], reward[a], new_observations[a], terminated[a])

                old_observations = new_observations

        if time_step % UPDATE_TARGET_NETWORK == 0:
            for a in range(env.number_of_agents):
                agents[a].update_target_model()

        if render:
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        # Termination causes the end of the episode
        if terminated["__all__"] or time_step == TIMESTEPS - 1:
            for a in range(env.number_of_agents):
                agents[a].update_target_model()
            if render:
                env_renderer.close_window()
            break

        # Retrain when the batch is ready
        for a in range(env.number_of_agents):
            if len(agents[a].replay_buffer) > BATCH_SIZE:
                agents[a].retrain(BATCH_SIZE)

    if (episode + 1) % print_episode_stats_freq == 0:
        if print_episode_stats:
            print("**********************************")
            print("Episode: {}".format(episode + 1))
            print("Action counter: " + str(action_counter))
            print("Final reward: " + str(episode_reward))
            print("**********************************")
        stats.append({"action_counter": action_counter, "episode_reward": episode_reward})

rewards = []
for i in range (0, len(stats)):
    rewards.append(stats[i]["episode_reward"])

episodes = np.linspace(1, EPISODES, EPISODES)
fig, ax = plt.subplots()
plt.title("Episode reward vs optimal path reward")
plt.plot(episodes, np.subtract(shortest_paths_rewards, rewards))
ax.set_xlabel("Episode")
ax.set_ylabel("shortest_path_reward - episode_reward")

# The plot shows the number of overstep compared to the optimal e.g. optimal = -8, reward_realized = -9 => -8-(-9) = 1
plt.show()
