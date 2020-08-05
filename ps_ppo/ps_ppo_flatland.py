import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from collections import OrderedDict

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
                 shared,
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
        :param shared: The actor and critic hidden and first layers are shared
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
        critic_layers = self._build_network(False, critic_mlp_depth, critic_mlp_width)
        self.critic_network = nn.Sequential(critic_layers)
        if not shared:
            self.actor_network = nn.Sequential(self._build_network(True, actor_mlp_depth, actor_mlp_width))
        else:
            if critic_mlp_depth <= 1:
                raise Exception("Shared networks must have depth greater than 1")
            actor_layers = critic_layers.copy()
            actor_layers.popitem()
            actor_layers["actor_output_layer"] = nn.Linear(critic_mlp_width, action_size)
            actor_layers["actor_softmax"] = nn.Softmax(dim=-1)
            self.actor_network = nn.Sequential(actor_layers)

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

        network = OrderedDict()
        output_size = self.action_size if is_actor else 1
        nn_type = "actor" if is_actor else "critic"

        # First layer
        network["%s_input" % nn_type] = nn.Linear(self.state_size,
                                                  nn_width if nn_depth > 1 else output_size)
        # If it's not the last layer add the activation
        if nn_depth > 1:
            network["%s_input_activation(%s)" % (nn_type, self.activation)] = self._get_activation()

        # Add hidden and last layers
        for layer in range(1, nn_depth):
            layer_name = "%s_layer_%d" % (nn_type, layer)
            # Last layer
            if layer == nn_depth - 1:
                network[layer_name] = nn.Linear(nn_width, output_size)
            # Hidden layer
            else:
                network[layer_name] = nn.Linear(nn_width, nn_width)
                network[layer_name + ("_activation(%s)" % self.activation)] = self._get_activation()

        # Actor needs softmax
        if is_actor:
            network["%s_softmax" % nn_type] = nn.Softmax(dim=-1)

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
        agent_id = int(state)
        # Transform the state Numpy array to a Torch Tensor
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.actor_network(state[-1])
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
        action_distribution = Categorical(self.actor_network(state[-1]))

        return action_distribution.log_prob(action[-1]), self.critic_network(state), action_distribution.entropy()


class PsPPO:
    def __init__(self,
                 n_agents,
                 state_size,
                 action_size,
                 shared,
                 critic_mlp_width,
                 critic_mlp_depth,
                 last_critic_layer_scaling,
                 actor_mlp_width,
                 actor_mlp_depth,
                 last_actor_layer_scaling,
                 learning_rate,
                 adam_eps,
                 activation,
                 discount_factor,
                 epochs,
                 batch_size,
                 eps_clip,
                 lmbda,
                 # advantage_estimator,
                 value_function_loss,
                 entropy_coefficient=None,
                 value_loss_coefficient=None
                 ):
        """
        :param n_agents: The number of agents
        :param state_size: The number of attributes of each state
        :param action_size: The number of available actions
        :param shared: The actor and critic hidden and first layers are shared
        :param critic_mlp_width: The number of nodes in the critic's hidden network
        :param critic_mlp_depth: The number of layers in the critic's network
        :param last_critic_layer_scaling: The scale applied at initialization on the last critic network's layer
        :param actor_mlp_width: The number of nodes in the actor's hidden network
        :param actor_mlp_depth: The number of layers in the actor's network
        :param last_actor_layer_scaling: The scale applied at initialization on the last actor network's layer
        :param learning_rate: The learning rate
        :param adam_eps: Adam optimizer epsilon value
        :param discount_factor: The discount factor
        :param epochs: The number of training epochs for each batch
        :param batch_size: The size of data batches used for each agent in the training loop
        :param eps_clip: The offset used in the minimum and maximum values of the clipping function
        :param value_function_loss: The function used to compute the value loss mse of huber (L1 loss)
        :param entropy_coefficient: Coefficient multiplied by the entropy and used in the shared setting loss function
        :param value_loss_coefficient: Coefficient multiplied by the value loss and used in the loss function
        """

        self.n_agents = n_agents
        self.shared = shared
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epochs = epochs
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.value_loss_coefficient = value_loss_coefficient
        self.entropy_coefficient = entropy_coefficient

        # The policy updated at each learning epoch
        self.policy = ActorCritic(state_size,
                                  action_size,
                                  shared,
                                  critic_mlp_width,
                                  critic_mlp_depth,
                                  last_critic_layer_scaling,
                                  actor_mlp_width,
                                  actor_mlp_depth,
                                  last_actor_layer_scaling,
                                  activation).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, eps=adam_eps)

        # The policy updated at the end of the training epochs where is used as the old policy.
        # It is used also to obtain trajectories.
        self.policy_old = ActorCritic(state_size,
                                      action_size,
                                      shared,
                                      critic_mlp_width,
                                      critic_mlp_depth,
                                      last_critic_layer_scaling,
                                      actor_mlp_width,
                                      actor_mlp_depth,
                                      last_actor_layer_scaling,
                                      activation).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        if value_function_loss == "mse":
            self.value_loss_function = nn.MSELoss()
        elif value_function_loss == "huber":
            self.value_loss_function = nn.SmoothL1Loss()
        else:
            raise Exception("The provided value loss function is not available!")


    def _get_advantages(self, lmbda, gamma, state_estimated_value, rewards):
        returns = []
        gae = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * state_estimated_value[i + 1] - state_estimated_value[i]
            gae = delta + gamma * lmbda * gae
            returns.append(gae + state_estimated_value[i])

        returns = torch.tensor(returns[::-1], dtype=torch.float32).to(device)
        adv = returns - state_estimated_value[:-1]

        return adv

    def update(self, memory):
        """
        :param memory:
        :return:
        """
        """
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(memory.rewards), reversed(memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.discount_factor * discounted_reward)
            rewards.append(discounted_reward)
        rewards = rewards[::-1]
        """
        # Save functions as objects outside to optimize code
        epochs = self.epochs
        n_agents = self.n_agents
        batch_size = self.batch_size

        lmbda = self.lmbda
        discount_factor = self.discount_factor
        policy_evaluate = self.policy.evaluate
        get_advantages = self._get_advantages
        torch_clamp = torch.clamp
        torch_min = torch.min
        obj_eps = self.eps_clip
        torch_exp = torch.exp
        shared = self.shared
        ec = self.entropy_coefficient
        vlc = self.value_loss_coefficient
        value_loss_function = self.value_loss_function
        optimizer = self.optimizer

        last_reward = memory.rewards.pop()
        last_done = memory.dones.pop()

        # TODO Why normalizing? boolean hyperparam
        # Normalizing the rewards:
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # For each agent train the policy and the value network on personal observations
        for a in range(n_agents):
            last_state = memory.states[a].pop()
            last_action = memory.actions[a].pop()
            last_logs_action_of_prop = memory.logs_of_action_prob[a].pop()

            # Convert lists to tensors
            old_states = torch.stack(memory.states[a]).to(device).detach()
            old_actions = torch.stack(memory.actions[a]).to(device).detach()
            old_logs_of_action_prob = torch.stack(memory.logs_of_action_prob[a]).to(device).detach()

            # Optimize policy
            for _ in range(epochs):
                for batch_start in range(0, len(old_states), batch_size):
                    batch_end = batch_start + batch_size
                    if batch_end >= len(old_states):
                        # Evaluating old actions and values
                        # print("Old_states: ",old_states[batch_start:batch_end].shape)
                        # print("Last_state: ", torch.unsqueeze(last_state, 0).shape)
                        # print("Action: ", old_actions[batch_start:batch_end].shape)
                        # print("Last action", torch.unsqueeze(last_action, 0).shape)
                        log_of_action_prob, state_estimated_value, dist_entropy = \
                            policy_evaluate(
                                torch.cat((old_states[batch_start:batch_end], torch.unsqueeze(last_state, 0))),
                                torch.cat((old_actions[batch_start:batch_end], torch.unsqueeze(last_action, 0))))
                    else:
                        # Evaluating old actions and values
                        # print("Old_states: ",old_states[batch_start:batch_end].shape)
                        # print("Action: ", old_actions[batch_start:batch_end].shape)
                        log_of_action_prob, state_estimated_value, dist_entropy = \
                            policy_evaluate(old_states[batch_start:batch_end + 1],
                                            old_actions[batch_start:batch_end + 1])

                    # print(old_states[batch_start:batch_end])
                    # print(old_actions[batch_start:batch_end])
                    # print(old_logs_of_action_prob[batch_start:batch_end])
                    # print(rewards[batch_start:batch_end])

                    # Find the ratio (pi_theta / pi_theta__old)
                    probs_ratio = torch_exp(
                        log_of_action_prob - old_logs_of_action_prob[batch_start:batch_end].detach())
                    # Find the "Surrogate Loss"
                    advantage = get_advantages(lmbda, discount_factor, state_estimated_value.detach(),
                                                     memory.rewards[batch_start:batch_end])
                    # advantage = rewards[batch_start:batch_end] - state_estimated_value.detach()

                    # Advantage normalization
                    advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-10)

                    unclipped_objective = probs_ratio * advantage
                    clipped_objective = torch_clamp(probs_ratio, 1 - obj_eps, 1 + obj_eps) * advantage

                    loss = -torch_min(unclipped_objective,
                                      clipped_objective) + vlc * value_loss_function(
                        state_estimated_value[:-1].squeeze(),
                        torch.tensor(memory.rewards[batch_start:batch_end], dtype=torch.float32).to(device))

                    if shared:
                        loss -= ec * dist_entropy

                    # Gradient descent
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()

                    # To show graph
                    """
                    from datetime import datetime
                    from torchviz import make_dot
                    now = datetime.now()
                    make_dot(loss.mean()).render("attached" + now.strftime("%H-%M-%S"), format="png")
                    exit()
                    """
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
    horizon = train_params.horizon
    batch_size = train_params.batch_size

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
                train_params.shared,
                train_params.critic_mlp_width,
                train_params.critic_mlp_depth,
                train_params.last_critic_layer_scaling,
                train_params.actor_mlp_width,
                train_params.actor_mlp_depth,
                train_params.last_actor_layer_scaling,
                train_params.learning_rate,
                train_params.adam_eps,
                train_params.activation,
                train_params.discount_factor,
                train_params.epochs,
                train_params.batch_size,
                train_params.eps_clip,
                train_params.lmbda,
                train_params.value_loss_function,
                # train_params.advantage_estimator
                train_params.entropy_coefficient,
                train_params.value_loss_coefficient
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
          .format(env.get_num_agents(), x_dim, y_dim, n_episodes, horizon))

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
        for step in range(max_steps):
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
            if timestep % (horizon + 1) == 0:
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
        tasks_finished = sum(info["status"][idx] == 2 or info["status"][idx] == 3 for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, n_agents)
        normalized_score = score / (max_steps * n_agents)
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
    "n_agents": 5,
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
    # If shared is True then the considered sizes are taken from the critic
    "shared": False,
    # Policy network
    "critic_mlp_width": 128,
    "critic_mlp_depth": 4,
    "last_critic_layer_scaling": 1.0,
    # Actor network
    "actor_mlp_width": 256,
    "actor_mlp_depth": 4,
    "last_actor_layer_scaling": 0.01,
    # Adam learning rate
    "learning_rate": 0.001,
    # Adam epsilon
    "adam_eps": 1e-5,
    # Activation
    "activation": "Tanh",
    "lmbda": 0.95,
    "entropy_coefficient": 0.01,
    # Called also baseline cost in shared setting (0.5)
    # (C54): {0.001, 0.1, 1.0, 10.0, 100.0}
    "value_loss_coefficient": 0.001,
    # ==============
    # Training setup
    # ==============
    "n_episodes": 2500,
    # 512, 1024, 2048, 4096
    "horizon": 1024,
    "epochs": 4,
    # Fixed trajectories, Shuffle trajectories, Shuffle transitions, Shuffle transitions (recompute advantages)
    # "batch_mode": None,
    # 64, 128, 256
    "batch_size": 64,

    # ==========================
    # Normalization and clipping
    # ==========================
    # Discount factor (0.95, 0.97, 0.99, 0.999)
    "discount_factor": 0.99,

    # ====================
    # Advantage estimation
    # ====================
    # PPO-style value clipping
    "eps_clip": 0.25,
    # GAE, N-steps
    # "advantage_estimator": "n-steps",
    # huber or mse
    "value_loss_function": "huber",

    # ==========================
    # Optimization and rendering
    # ==========================
    "checkpoint_interval": 100,
    "use_gpu": False,
    "num_threads": 1,
    "render": False,
}

train_multiple_agents(Namespace(**environment_parameters), Namespace(**training_parameters))