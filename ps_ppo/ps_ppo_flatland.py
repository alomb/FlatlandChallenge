import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.actions = [[] for _ in range(num_agents)]
        self.states = [[] for _ in range(num_agents)]
        self.logs_of_action_prob = [[] for _ in range(num_agents)]

        self.rewards = []
        self.dones = []

    def clear_memory(self):
        self.__init__(self.num_agents)


class ActorCritic(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 critic_mlp_width,
                 critic_mlp_depth,
                 last_critic_layer_scaling,
                 actor_mlp_width,
                 actor_mlp_depth,
                 last_actor_layer_scaling,
                 activation):
        """
        :param state_size: The number of attributes of each state
        :param action_size: The number of available actions
        :param critic_mlp_width: The number of nodes in the critic's hidden network
        :param critic_mlp_depth: The number of hidden layers + the input one of the critic's network
        :param last_critic_layer_scaling: The scale applied at initialization on the last critic network's layer
        :param actor_mlp_width: The number of nodes in the actor's hidden network
        :param actor_mlp_depth: The number of hidden layers + the input one of the actor's network
        :param last_actor_layer_scaling: The scale applied at initialization on the last actor network's layer
        :param activation: the activation function (ReLU, Tanh)
        """

        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.activation = activation

        # Network creation
        self.critic_network = self._build_network(False, critic_mlp_depth, critic_mlp_width)
        self.actor_network = self._build_network(True, actor_mlp_depth, actor_mlp_width)

        # Network initialization
        # https://pytorch.org/docs/stable/nn.init.html#nn-init-doc
        def weights_init(submodule):
            # TODO
            raise NotImplementedError()

        # self.critic_network.apply(weights_init)
        # self.actor_network.apply(weights_init)

        # Last layer's weights rescaling
        with torch.no_grad():
            list(self.critic_network.children())[-1].weight.mul_(last_critic_layer_scaling)
            # -2 because actor contains softmax as last layer
            list(self.actor_network.children())[-2].weight.mul_(last_actor_layer_scaling)

    def _build_network(self, is_actor, nn_depth, nn_width):
        if nn_depth <= 0:
            raise Exception("Networks' depths must be greater than 0")

        network = nn.Sequential()
        output_size = self.action_size if is_actor else 1
        nn_type = "actor" if is_actor else "critic"

        # First layer
        network.add_module("%s_input" % nn_type, nn.Linear(self.state_size,
                                                           nn_width if nn_depth > 1 else output_size))
        # If it's not the last layer add the activation
        if nn_depth > 1:
            network.add_module("%s_input_activation(%s)" % (nn_type, self.activation), self._get_activation())

        # Add hidden and last layers
        for layer in range(1, nn_depth):
            layer_name = "%s_layer_%d" % (nn_type, layer)
            # Last layer
            if layer == nn_depth - 1:
                network.add_module(layer_name, nn.Linear(nn_width, output_size))
            # Hidden layer
            else:
                network.add_module(layer_name, nn.Linear(nn_width, nn_width))
                network.add_module(layer_name + ("_activation(%s)" % self.activation), self._get_activation())

        # Actor needs softmax
        if is_actor:
            network.add_module("%s_softmax", nn.Softmax(dim=-1))

        return network

    def _get_activation(self):
        if self.activation == "ReLU":
            return nn.ReLU()
        elif self.activation == "Tanh":
            return nn.Tanh()
        else:
            raise Exception("The specified activation function don't exists or is not available")

    def act(self, state, memory, action=None):
        """
        The method used by the agent as its own policy to obtain the action to perform in the given a state and update
        the memory.

        :param state: the observed state
        :param memory: the memory to update
        :param action:
        :return: the action to perform
        """

        # The agent name is appended at the state
        agent_id = int(state[-1])
        # Transform the state Numpy array to a Torch Tensor
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.actor_network(state)
        """
        From the paper: "The stochastic policy πθ can be represented by a categorical distribution when the actions of
        the agent are discrete and by a Gaussian distribution when the actions are continuous."
        """
        action_distribution = Categorical(action_probs)

        if not action:
            action = action_distribution.sample()

        # Memory is updated
        memory.states[agent_id].append(state)
        memory.actions[agent_id].append(action)
        memory.logs_of_action_prob[agent_id].append(action_distribution.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        """
        Evaluate the current policy obtaining useful information on the decided action's probability distribution.

        :param state: the observed state
        :param action: the performed action
        :return: the logarithm of action probability, the value predicted by the critic, the distribution entropy
        """
        action_distribution = Categorical(self.actor_network(state))

        return action_distribution.log_prob(action), self.critic_network(state), action_distribution.entropy()


class PsPPO:
    def __init__(self,
                 n_agents,
                 state_size,
                 action_size,
                 # shared,
                 critic_mlp_width,
                 critic_mlp_depth,
                 last_critic_layer_scaling,
                 actor_mlp_width,
                 actor_mlp_depth,
                 last_actor_layer_scaling,
                 learning_rate,
                 activation,
                 discount_factor,
                 epochs,
                 eps_clip,
                 # advantage_estimator,
                 # value_function_loss
                 ):
        """
        :param n_agents: The number of agents
        :param state_size: The number of attributes of each state
        :param action_size: The number of available actions
        :param critic_mlp_width: The number of nodes in the critic's hidden network
        :param critic_mlp_depth: The number of layers in the critic's network
        :param last_critic_layer_scaling: The scale applied at initialization on the last critic network's layer
        :param actor_mlp_width: The number of nodes in the actor's hidden network
        :param actor_mlp_depth: The number of layers in the actor's network
        :param last_actor_layer_scaling: The scale applied at initialization on the last actor network's layer
        :param learning_rate: The learning rate
        :param discount_factor: The discount factor
        :param epochs: The number of training epochs for each batch
        :param eps_clip: The offset used in the minimum and maximum values of the clipping function
        """

        self.n_agents = n_agents
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epochs = epochs
        self.eps_clip = eps_clip

        # The policy updated at each learning epoch
        self.policy = ActorCritic(state_size,
                                  action_size,
                                  critic_mlp_width,
                                  critic_mlp_depth,
                                  last_critic_layer_scaling,
                                  actor_mlp_width,
                                  actor_mlp_depth,
                                  last_actor_layer_scaling,
                                  activation).to(device)
        # TODO: Consider changing Adam or its hyperparameters
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # The policy updated at the end of the training epochs where is used as the old policy.
        # It is used also to obtain trajectories.
        self.policy_old = ActorCritic(state_size,
                                      action_size,
                                      critic_mlp_width,
                                      critic_mlp_depth,
                                      last_critic_layer_scaling,
                                      actor_mlp_width,
                                      actor_mlp_depth,
                                      last_actor_layer_scaling,
                                      activation).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()

    def update(self, memory):
        """

        :param memory:
        :return:
        """
        # TODO: Optim: Use an iterator, possibly combining with mini-batch gd
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(memory.rewards), reversed(memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.discount_factor * discounted_reward)
            rewards.append(discounted_reward)
        rewards = rewards[::-1]

        """
        from itertools import repeat
        # Optimization that seems not to work lol, try with bigger lists
        def myfunc(reward_done_tuple, discounted_reward):
            if reward_done_tuple[1]:
                discounted_reward[0] = 0
            discounted_reward[0] = reward_done_tuple[0] + (self.gamma * discounted_reward[0])
            return discounted_reward[0]

        rewards = list(map(myfunc, zip(reversed(memory.rewards), reversed(memory.dones)), repeat([0])))[::-1]
        """

        # TODO Why normalizing
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # For each agent train the policy and the value network on personal observations
        for a in range(self.n_agents):
            # TODO: Mini-batch
            # Convert lists to tensors
            old_states = torch.stack(memory.states[a]).to(device).detach()
            old_actions = torch.stack(memory.actions[a]).to(device).detach()
            old_logs_of_action_prob = torch.stack(memory.logs_of_action_prob[a]).to(device).detach()

            # Save functions as objects outside to optimize code
            policy_evaluate = self.policy.evaluate
            torch_clamp = torch.clamp
            torch_min = torch.min
            torch_exp = torch.exp

            # Optimize policy
            for _ in range(self.epochs):
                # Evaluating old actions and values
                log_of_action_prob, state_estimated_value, dist_entropy = policy_evaluate(old_states, old_actions)

                # Find the ratio (pi_theta / pi_theta__old)
                probs_ratio = torch_exp(log_of_action_prob - old_logs_of_action_prob.detach())
                # Find the "Surrogate Loss"
                advantage = rewards - state_estimated_value.detach()
                unclipped_objective = probs_ratio * advantage
                clipped_objective = torch_clamp(probs_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                """
                loss = -torch_min(unclipped_objective, clipped_objective) + \
                       0.5 * self.mse_loss(state_estimated_value, rewards) - 0.01 * dist_entropy
                """
                loss = -torch_min(unclipped_objective, clipped_objective)

                # Gradient descent
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


########################################################################################################################
########################################################################################################################


from flatland.envs.observations import TreeObsForRailEnv


def max_lt(seq, val):
    """
    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if val > seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_gt(seq, val):
    """
    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if val <= seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min


def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    """
    This function returns the difference between min and max value of an observation
    :param obs: Observation that should be normalized
    :param clip_min: min value where observation will be clipped
    :param clip_max: max value where observation will be clipped
    :return: returnes normalized and clipped observatoin
    """
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max_lt(obs, 1000)) + 1

    min_obs = 0  # min(max_obs, min_gt(obs, 0))
    if normalize_to_range:
        min_obs = min_gt(obs, 0)
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def _split_node_into_feature_groups(node: TreeObsForRailEnv.Node) -> (np.ndarray, np.ndarray, np.ndarray):
    data = np.zeros(6)
    distance = np.zeros(1)
    agent_data = np.zeros(4)

    data[0] = node.dist_own_target_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.speed_min_fractional

    return data, distance, agent_data


def _split_subtree_into_feature_groups(node: TreeObsForRailEnv.Node, current_tree_depth: int, max_tree_depth: int) -> (
        np.ndarray, np.ndarray, np.ndarray):
    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
        num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        return [-np.inf] * num_remaining_nodes * 6, [-np.inf] * num_remaining_nodes, [-np.inf] * num_remaining_nodes * 4

    data, distance, agent_data = _split_node_into_feature_groups(node)

    if not node.childs:
        return data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(
            node.childs[direction], current_tree_depth + 1, max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def split_tree_into_feature_groups(tree: TreeObsForRailEnv.Node, max_tree_depth: int) -> (
        np.ndarray, np.ndarray, np.ndarray):
    """
    This function splits the tree into three difference arrays of values
    """
    data, distance, agent_data = _split_node_into_feature_groups(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(tree.childs[direction], 1,
                                                                                    max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def normalize_observation(observation: TreeObsForRailEnv.Node, tree_depth: int, observation_radius=0):
    """
    This function normalizes the observation used by the RL algorithm
    """
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
    return normalized_obs


from timeit import default_timer


class Timer(object):
    def __init__(self):
        self.total_time = 0.0
        self.start_time = 0.0
        self.end_time = 0.0

    def start(self):
        self.start_time = default_timer()

    def end(self):
        self.total_time += default_timer() - self.start_time

    def get(self):
        return self.total_time

    def get_current(self):
        return default_timer() - self.start_time

    def reset(self):
        self.__init__()

    def __repr__(self):
        return self.get()


########################################################################################################################
########################################################################################################################

import random
from argparse import ArgumentParser, Namespace

from flatland.utils.rendertools import RenderTool
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

try:
    import wandb

    wandb.init(sync_tensorboard=True)
except ImportError:
    print("Install wandb to log to Weights & Biases")

SUPPRESS_OUTPUT = False

if SUPPRESS_OUTPUT:
    # ugly hack to be able to run hyperparameters sweeps with w&b
    # they currently have a bug which prevents runs that output emojis to run :(
    def print(*args, **kwargs):
        pass


def train_multiple_agents(env_params, train_params):
    # Environment parameters
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city
    seed = env_params.seed

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_radius = env_params.observation_radius
    observation_max_path_depth = env_params.observation_max_path_depth

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1. / 10000,  # Rate of malfunctions
        min_duration=15,  # Minimal duration
        max_duration=50  # Max duration
    )

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Fraction of train which each speed
    speed_profiles = {
        1.: 1.0,  # Fast passenger train
        1. / 2.: 0.0,  # Fast freight train
        1. / 3.: 0.0,  # Slow commuter train
        1. / 4.: 0.0  # Slow freight train
    }

    # Setup the environment
    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city,
            seed=seed
        ),
        schedule_generator=sparse_schedule_generator(speed_profiles),
        number_of_agents=n_agents,
        malfunction_generator_and_process_data=malfunction_from_params(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed
    )

    env.reset(regenerate_schedule=True, regenerate_rail=True)

    n_episodes = train_params.n_episodes
    checkpoint_interval = train_params.checkpoint_interval
    update_timestep = train_params.update_timestep
    # TODO: Mini-batch gd
    # batch_mode = train_params.batch_mode
    # batch_size = training_parameters.batch_size

    # Setup renderer
    if train_params.render:
        env_renderer = RenderTool(env, gl="PGL")

    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = 0
    for i in range(observation_tree_depth + 1):
        n_nodes += np.power(4, i)
    state_size = n_features_per_node * n_nodes

    # The action space of flatland is 5 discrete actions
    # TODO: from param
    action_size = 5

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    # TODO: from param
    max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))

    action_count = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    smoothed_normalized_score = -1.0
    smoothed_completion = 0.0

    memory = Memory(n_agents)

    ppo = PsPPO(n_agents,
                state_size + 1,
                action_size,
                # train_params.shared,
                train_params.critic_mlp_width,
                train_params.critic_mlp_depth,
                train_params.last_critic_layer_scaling,
                train_params.actor_mlp_width,
                train_params.actor_mlp_depth,
                train_params.last_actor_layer_scaling,
                train_params.learning_rate,
                train_params.activation,
                train_params.discount_factor,
                train_params.epochs,
                train_params.eps_clip
                # train_params.advantage_estimator,
                # train_params.value_function_loss
                )

    """
    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_hparams(vars(train_params), {})
    writer.add_hparams(vars(env_params), {})
    """

    training_timer = Timer()
    training_timer.start()

    print("\nTraining {} trains on {}x{} grid for {} episodes. Update every {} timesteps.\n"
          .format(env.get_num_agents(), x_dim, y_dim, n_episodes, update_timestep))

    timestep = 0
    for episode in range(n_episodes + 1):
        # Timers
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()

        # Reset environment
        reset_timer.start()
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        reset_timer.end()

        if train_params.render:
            env_renderer.set_new_rail()

        score = 0

        # Run episode
        for step in range(max_steps - 1):
            timestep += 1

            # Collect and preprocess observations
            for agent in env.get_agent_handles():
                # TODO: check if
                if obs[agent]:
                    preproc_timer.start()
                    agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth,
                                                             observation_radius=observation_radius)
                    preproc_timer.end()

            action_dict = {a: ppo.policy_old.act(np.append(agent_obs[a], [a]), memory) if info['action_required'][a]
            else ppo.policy_old.act(np.append(agent_obs[a], [a]), memory, action=0)
                           for a in range(n_agents)}

            for a in list(action_dict.values()):
                action_count[a] += 1

            # Environment step
            step_timer.start()
            obs, rewards, done, info = env.step(action_dict)
            step_timer.end()

            total_timestep_reward = np.sum(list(rewards.values()))
            score += total_timestep_reward
            memory.rewards.append(total_timestep_reward)
            memory.dones.append(done['__all__'])

            # Update
            if timestep % update_timestep == 0:
                # print("update")
                learn_timer.start()
                ppo.update(memory)
                learn_timer.end()
                memory.clear_memory()
                timestep = 0

            if train_params.render and episode % checkpoint_interval == 0:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

            if done['__all__']:
                break

        # Collection information about training
        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        normalized_score = score / (max_steps * env.get_num_agents())
        action_probs = action_count / np.sum(action_count)
        action_count = [1] * action_size

        # Smoothed values for terminal display and for more stable hyper-parameter tuning
        smoothing = 0.99
        smoothed_normalized_score = smoothed_normalized_score * smoothing + normalized_score * (1.0 - smoothing)
        smoothed_completion = smoothed_completion * smoothing + completion * (1.0 - smoothing)

        # Print logs
        if episode % checkpoint_interval == 0:
            # TODO: Save network params as checkpoints
            if train_params.render:
                env_renderer.close_window()

        print(
            '\rEpisode {}'
            '\tScore: {:.3f}'
            ' Avg: {:.3f}'
            '\tDone: {:.2f}%'
            ' Avg: {:.2f}%'
            '\tAction Probs: {}'.format(
                episode,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                format_action_prob(action_probs)
            ), end=" ")

        # TODO: Consider possible eval
        """
        if episode_idx % train_params.checkpoint_interval == 0:
            scores, completions, nb_steps_eval = eval_policy(env, policy, n_eval_episodes, max_steps)
            writer.add_scalar("evaluation/scores_min", np.min(scores), episode_idx)
            writer.add_scalar("evaluation/scores_max", np.max(scores), episode_idx)
            writer.add_scalar("evaluation/scores_mean", np.mean(scores), episode_idx)
            writer.add_scalar("evaluation/scores_std", np.std(scores), episode_idx)
            writer.add_histogram("evaluation/scores", np.array(scores), episode_idx)
            writer.add_scalar("evaluation/completions_min", np.min(completions), episode_idx)
            writer.add_scalar("evaluation/completions_max", np.max(completions), episode_idx)
            writer.add_scalar("evaluation/completions_mean", np.mean(completions), episode_idx)
            writer.add_scalar("evaluation/completions_std", np.std(completions), episode_idx)
            writer.add_histogram("evaluation/completions", np.array(completions), episode_idx)
            writer.add_scalar("evaluation/nb_steps_min", np.min(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_max", np.max(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_mean", np.mean(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_std", np.std(nb_steps_eval), episode_idx)
            writer.add_histogram("evaluation/nb_steps", np.array(nb_steps_eval), episode_idx)

            smoothing = 0.9
            smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(scores) * (1.0 - smoothing)
            smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)
            writer.add_scalar("evaluation/smoothed_score", smoothed_eval_normalized_score, episode_idx)
            writer.add_scalar("evaluation/smoothed_completion", smoothed_eval_completion, episode_idx)

        # Save logs to tensorboard
        writer.add_scalar("training/score", normalized_score, episode_idx)
        writer.add_scalar("training/smoothed_score", smoothed_normalized_score, episode_idx)
        writer.add_scalar("training/completion", np.mean(completion), episode_idx)
        writer.add_scalar("training/smoothed_completion", np.mean(smoothed_completion), episode_idx)
        writer.add_scalar("training/nb_steps", nb_steps, episode_idx)
        writer.add_histogram("actions/distribution", np.array(actions_taken), episode_idx)
        writer.add_scalar("actions/nothing", action_probs[RailEnvActions.DO_NOTHING], episode_idx)
        writer.add_scalar("actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode_idx)
        writer.add_scalar("actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode_idx)
        writer.add_scalar("actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode_idx)
        writer.add_scalar("actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode_idx)
        writer.add_scalar("training/epsilon", eps_start, episode_idx)
        writer.add_scalar("training/buffer_size", len(policy.memory), episode_idx)
        writer.add_scalar("training/loss", policy.loss, episode_idx)
        writer.add_scalar("timer/reset", reset_timer.get(), episode_idx)
        writer.add_scalar("timer/step", step_timer.get(), episode_idx)
        writer.add_scalar("timer/learn", learn_timer.get(), episode_idx)
        writer.add_scalar("timer/preproc", preproc_timer.get(), episode_idx)
        writer.add_scalar("timer/total", training_timer.get_current(), episode_idx)
        """


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


myseed = 19

environment_parameters = {
    # small_v0 config
    "n_agents": 2,
    "x_dim": 35,
    "y_dim": 35,
    "n_cities": 4,
    "max_rails_between_cities": 2,
    "max_rails_in_city": 3,

    "seed": myseed,
    "observation_tree_depth": 2,
    "observation_radius": 10,
    "observation_max_path_depth": 30
}

training_parameters = {
    "random_seed": myseed,
    # ====================
    # Network architecture
    # ====================
    # Shared actor-critic layer
    # "shared": False,
    # Policy network
    "critic_mlp_width": 64,
    "critic_mlp_depth": 2,
    "last_critic_layer_scaling": 0.01,
    # Actor network
    "actor_mlp_width": 64,
    "actor_mlp_depth": 2,
    "last_actor_layer_scaling": 1.0,
    # Adam learning rate
    "learning_rate": 0.001,
    # Activation
    "activation": "ReLU",

    # ==============
    # Training setup
    # ==============
    "n_episodes": 2500,
    "checkpoint_interval": 100,
    "update_timestep": 300,
    "epochs": 4,
    # Fixed trajectories, Shuffle trajectories, Shuffle transitions, Shuffle transitions (recompute advantages)
    # "batch_mode": None,
    # 64, 128, 256
    # "batch_size": 64,

    # ==========================
    # Normalization and clipping
    # ==========================
    # Discount factor (0.95, 0.97, 0.99, 0.999)
    "discount_factor": 0.99,

    # ====================
    # Advantage estimation
    # ====================
    # PPO-style value clipping
    "eps_clip": 0.2,
    # GAE, N-steps
    # "advantage_estimator": "n-steps",
    # huber, mse
    # "value_function_loss": "mse",

    # ==========================
    # Optimization and rendering
    # ==========================
    "use_gpu": False,
    "num_threads": 1,
    "render": False,
}

train_multiple_agents(Namespace(**environment_parameters), Namespace(**training_parameters))
