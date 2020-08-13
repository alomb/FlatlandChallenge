import numpy as np
import torch
import torch.nn as nn
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.agent_utils import RailAgentStatus
from torch.distributions import Categorical
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.actions = [[] for _ in range(num_agents)]
        self.states = [[] for _ in range(num_agents)]
        self.logs_of_action_prob = [[] for _ in range(num_agents)]
        self.masks = [[] for _ in range(num_agents)]
        self.rewards = [[] for _ in range(num_agents)]
        self.dones = [[] for _ in range(num_agents)]

    def clear_memory(self):
        self.__init__(self.num_agents)

    def clear_memory_except_last(self, agent):
        self.actions[agent] = self.actions[agent][-1:]
        self.states[agent] = self.states[agent][-1:]
        self.logs_of_action_prob[agent] = self.logs_of_action_prob[agent][-1:]
        self.masks[agent] = self.masks[agent][-1:]
        self.rewards[agent] = self.rewards[agent][-1:]
        self.dones[agent] = self.dones[agent][-1:]


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
        self.softmax = nn.Softmax(dim=-1)

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
            list(self.actor_network.children())[-1].weight.mul_(last_actor_layer_scaling)

    def _build_network(self, is_actor, nn_depth, nn_width):
        """
        Creates the network.
        Actor is not completed with the final softmax layer

        :param is_actor:
        :param nn_depth:
        :param nn_width:
        :return:
        """
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

        return network

    def _get_activation(self):
        if self.activation == "ReLU":
            return nn.ReLU()
        elif self.activation == "Tanh":
            return nn.Tanh()
        else:
            raise Exception("The specified activation function don't exists or is not available")

    def act(self, state, memory, action_mask, action=None):
        """
        The method used by the agent as its own policy to obtain the action to perform in the given a state and update
        the memory.
        :param state: the observed state
        :param memory: the memory to update
        :param action_mask: a list of 0 and 1 where 0 indicates that the agent should be not sampled
        :param action: an action to perform decided by some external logic
        :return: the action to perform
        """

        # The agent name is appended at the state
        agent_id = int(state[-1])
        # Transform the state Numpy array to a Torch Tensor
        state = torch.from_numpy(state).float().to(device)
        action_logits = self.actor_network(state)

        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(device)

        # Action masking, default values are True, False are present only if masking is enabled.
        # If No op is not allowed it is masked even if masking is not active
        action_logits = torch.where(action_mask, action_logits, torch.tensor(-1e+8).to(device))

        action_probs = self.softmax(action_logits)

        """
        From the paper: "The stochastic policy πθ can be represented by a categorical distribution when the actions of
        the agent are discrete and by a Gaussian distribution when the actions are continuous."
        """
        action_distribution = Categorical(action_probs)

        if action is None:
            action = action_distribution.sample()

        # Memory is updated
        memory.states[agent_id].append(state)
        memory.actions[agent_id].append(action)
        memory.logs_of_action_prob[agent_id].append(action_distribution.log_prob(action))
        memory.masks[agent_id].append(action_mask)

        return action.item()

    def evaluate(self, state, action, action_mask):
        """
        Evaluate the current policy obtaining useful information on the decided action's probability distribution.
        :param state: the observed state
        :param action: the performed action
        :param action_mask: a list of 0 and 1 where 0 indicates that the agent should be not sampled
        :return: the logarithm of action probability, the value predicted by the critic, the distribution entropy
        """

        action_logits = self.actor_network(state[:-1])

        # Action masking, default values are True, False are present only if masking is enabled.
        # If No op is not allowed it is masked even if masking is not active
        action_logits = torch.where(action_mask[:-1], action_logits, torch.tensor(-1e+8).to(device))

        action_probs = self.softmax(action_logits)

        action_distribution = Categorical(action_probs)

        return action_distribution.log_prob(action[:-1]), self.critic_network(state), action_distribution.entropy()


class PsPPO:
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
                 learning_rate,
                 adam_eps,
                 activation,
                 discount_factor,
                 epochs,
                 batch_size,
                 eps_clip,
                 lmbda,
                 advantage_estimator,
                 value_function_loss,
                 entropy_coefficient=None,
                 value_loss_coefficient=None):
        """
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
        :param lmbda: Controls gae bias–variance trade-off
        :param advantage_estimator: The advantage estimation technique n-steps or gae (Generalized Advantage estimation)
        :param value_function_loss: The function used to compute the value loss mse of huber (L1 loss)
        :param entropy_coefficient: Coefficient multiplied by the entropy and used in the shared setting loss function
        :param value_loss_coefficient: Coefficient multiplied by the value loss and used in the loss function
        """

        self.shared = shared
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epochs = epochs
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.value_loss_coefficient = value_loss_coefficient
        self.entropy_coefficient = entropy_coefficient

        if advantage_estimator == "gae":
            self.gae = True
        elif advantage_estimator == "n-steps":
            self.gae = False
        else:
            raise Exception("Advantage estimator not available")

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

    def _get_advs(self, gae, rewards, dones, gamma, state_estimated_value, lmbda=None):
        """
        advantages = []
        import math

        for t in range(len(rewards)):
            if t == 0:
                adv = rewards[t] + gamma * state_estimated_value[t + 1] - state_estimated_value[t]
            else:
                adv = adv + math.pow(gamma, t) * rewards[t] +\
                      math.pow(gamma, t + 1) * state_estimated_value[t + 1] -\
                      math.pow(gamma, t) * state_estimated_value[t]
            advantages.append(adv * math.pow(lmbda, t))

        gae = (1 - lmbda) * torch.tensor(advantages, dtype=torch.float32).to(device)

        """

        if gae:
            assert len(rewards) + 1 == len(state_estimated_value)

            rewards = torch.tensor(rewards).to(device)
            gaes = torch.zeros_like(rewards)
            future_gae = torch.tensor(0.0, dtype=rewards.dtype).to(device)

            # to multiply with not_dones to handle episode boundary (last state has no V(s'))
            not_dones = 1 - torch.tensor(dones, dtype=torch.int).to(device)
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + gamma * state_estimated_value[t + 1] * not_dones[t] - state_estimated_value[t]
                gaes[t] = future_gae = delta + gamma * lmbda * not_dones[t] * future_gae

            return gaes
        else:
            rewards = torch.tensor(rewards).to(device)
            returns = torch.zeros_like(rewards)
            future_ret = state_estimated_value[-1]

            not_dones = 1 - torch.tensor(dones, dtype=torch.int).to(device)
            for t in reversed(range(len(rewards))):
                returns[t] = future_ret = rewards[t] + gamma * future_ret * not_dones[t]

            return returns - state_estimated_value[:-1]

    def update(self, memory, a):

        """
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(memory.rewards), reversed(memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.discount_factor * discounted_reward)
            rewards.append(discounted_reward)
        rewards = rewards[::-1]

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        """
        # Save functions as objects outside to optimize code
        epochs = self.epochs
        batch_size = self.batch_size

        lmbda = self.lmbda
        discount_factor = self.discount_factor
        policy_evaluate = self.policy.evaluate
        get_advantages = self._get_advs
        gae = self.gae
        torch_clamp = torch.clamp
        torch_min = torch.min
        obj_eps = self.eps_clip
        torch_exp = torch.exp
        shared = self.shared
        ec = self.entropy_coefficient
        vlc = self.value_loss_coefficient
        value_loss_function = self.value_loss_function
        optimizer = self.optimizer

        _ = memory.rewards[a].pop()
        _ = memory.dones[a].pop()

        last_state = memory.states[a].pop()
        last_action = memory.actions[a].pop()
        last_mask = memory.masks[a].pop()
        _ = memory.logs_of_action_prob[a].pop()

        # Convert lists to tensors
        old_states = torch.stack(memory.states[a]).to(device).detach()
        old_actions = torch.stack(memory.actions[a]).to(device)
        old_masks = torch.stack(memory.masks[a]).to(device)
        old_logs_of_action_prob = torch.stack(memory.logs_of_action_prob[a]).to(device).detach()

        # Optimize policy
        for _ in range(epochs):
            for batch_start in range(0, len(old_states), batch_size):
                batch_end = batch_start + batch_size
                if batch_end >= len(old_states):
                    # Evaluating old actions and values
                    # print("Old_states: ", old_states[batch_start:batch_end].shape)
                    # print("Last_state: ", torch.unsqueeze(last_state, 0).shape)
                    # print("Action: ", old_actions[batch_start:batch_end].shape)
                    # print("Last action", torch.unsqueeze(last_action, 0).shape)
                    log_of_action_prob, state_estimated_value, dist_entropy = \
                        policy_evaluate(
                            torch.cat((old_states[batch_start:batch_end], torch.unsqueeze(last_state, 0))),
                            torch.cat((old_actions[batch_start:batch_end], torch.unsqueeze(last_action, 0))),
                            torch.cat((old_masks[batch_start:batch_end], torch.unsqueeze(last_mask, 0))))
                    # torch.cat((old_actions[batch_start:batch_end], torch.tensor(last_action).reshape(1, 1))))
                else:
                    # Evaluating old actions and values
                    # print("Old_states: ",old_states[batch_start:batch_end + 1].shape)
                    # print("Action: ", old_actions[batch_start:batch_end + 1].shape)
                    log_of_action_prob, state_estimated_value, dist_entropy = \
                        policy_evaluate(old_states[batch_start:batch_end + 1],
                                        old_actions[batch_start:batch_end + 1],
                                        old_masks[batch_start:batch_end + 1])

                # print(old_states[batch_start:batch_end])
                # print(old_actions[batch_start:batch_end])
                # print(old_logs_of_action_prob[batch_start:batch_end])
                # print(log_of_action_prob)
                # print(rewards[batch_start:batch_end])

                # Find the ratio (pi_theta / pi_theta__old)
                probs_ratio = torch_exp(
                    log_of_action_prob - old_logs_of_action_prob[batch_start:batch_end].detach())
                # Find the "Surrogate Loss"

                advantage = get_advantages(
                    gae,
                    memory.rewards[a][batch_start:batch_end],
                    memory.dones[a][batch_start:batch_end],
                    discount_factor,
                    state_estimated_value.detach(),
                    lmbda)

                # advantage = rewards[a][batch_start:batch_end] - state_estimated_value.detach()

                """
                print("estimated value\t " + str(torch.mean(state_estimated_value).item()))
                print("reward\t " + str(
                    torch.mean(torch.tensor(memory.rewards[a][batch_start:batch_end]).to(device)).item()))
                print("advantage\t " + str(torch.mean(advantage).item()))
                print("probsratio\t " + str(torch.mean(probs_ratio).item()))
                """

                # Advantage normalization
                advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-10)

                unclipped_objective = probs_ratio * advantage
                clipped_objective = torch_clamp(probs_ratio, 1 - obj_eps, 1 + obj_eps) * advantage

                loss = -torch_min(unclipped_objective,
                                  clipped_objective) + vlc * value_loss_function(
                    state_estimated_value[:-1].squeeze(),
                    torch.tensor(memory.rewards[a][batch_start:batch_end], dtype=torch.float32).to(device))
                loss -= ec * torch.mean(dist_entropy)

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
    max_item = 0
    idx = len(seq) - 1
    while idx >= 0:
        if val > seq[idx] >= 0 and seq[idx] > max_item:
            max_item = seq[idx]
        idx -= 1
    return max_item


def min_gt(seq, val):
    """
    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min_item = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if val <= seq[idx] < min_item:
            min_item = seq[idx]
        idx -= 1
    return min_item


def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    """
    This function returns the difference between min and max value of an observation
    :param obs: Observation that should be normalized
    :param clip_min: min value where observation will be clipped
    :param clip_max: max value where observation will be clipped
    :param fixed_radius:
    :param normalize_to_range:
    :return: returns normalized and clipped observation
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
from argparse import Namespace

from flatland.utils.rendertools import RenderTool
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


def check_feasible_transitions(pos_a1, transitions, env):
    if transitions[0] == 1:
        position_check = (pos_a1[0] - 1, pos_a1[1])
        if not (env.cell_free(position_check)):
            for a2 in range(env.get_num_agents()):
                if env.agents[a2].position == position_check:
                    return a2

    if transitions[1] == 1:
        position_check = (pos_a1[0], pos_a1[1] + 1)
        if not (env.cell_free(position_check)):
            for a2 in range(env.get_num_agents()):
                if env.agents[a2].position == position_check:
                    return a2

    if transitions[2] == 1:
        position_check = (pos_a1[0] + 1, pos_a1[1])
        if not (env.cell_free(position_check)):
            for a2 in range(env.get_num_agents()):
                if env.agents[a2].position == position_check:
                    return a2

    if transitions[3] == 1:
        position_check = (pos_a1[0], pos_a1[1] - 1)
        if not (env.cell_free(position_check)):
            for a2 in range(env.get_num_agents()):
                if env.agents[a2].position == position_check:
                    return a2

    return None


def check_next_pos(a1, env):
    if env.agents[a1].position is not None:
        pos_a1 = env.agents[a1].position
        dir_a1 = env.agents[a1].direction
    else:
        pos_a1 = env.agents[a1].initial_position
        dir_a1 = env.agents[a1].initial_direction

    # NORTH
    if dir_a1 == 0:
        if env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1)[dir_a1] == 1:
            position_check = (pos_a1[0] - 1, pos_a1[1])
            if not (env.cell_free(position_check)):
                for a2 in range(env.get_num_agents()):
                    if env.agents[a2].position == position_check:
                        return a2
        else:
            return check_feasible_transitions(pos_a1, env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1), env)

    # EAST
    if dir_a1 == 1:
        if env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1)[dir_a1] == 1:
            position_check = (pos_a1[0], pos_a1[1] + 1)
            if not (env.cell_free(position_check)):
                for a2 in range(env.get_num_agents()):
                    if env.agents[a2].position == position_check:
                        return a2
        else:
            return check_feasible_transitions(pos_a1, env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1), env)

    # SOUTH
    if dir_a1 == 2:
        if env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1)[dir_a1] == 1:
            position_check = (pos_a1[0] + 1, pos_a1[1])
            if not (env.cell_free(position_check)):
                for a2 in range(env.get_num_agents()):
                    if env.agents[a2].position == position_check:
                        return a2
        else:
            return check_feasible_transitions(pos_a1, env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1), env)

    # WEST
    if dir_a1 == 3:
        if env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1)[dir_a1] == 1:
            position_check = (pos_a1[0], pos_a1[1] - 1)
            if not (env.cell_free(position_check)):
                for a2 in range(env.get_num_agents()):
                    if env.agents[a2].position == position_check:
                        return a2
        else:
            return check_feasible_transitions(pos_a1, env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1), env)

    return None


def check_deadlocks(a1, deadlocks, env):
    a2 = check_next_pos(a1[-1], env)

    if a2 is None:
        return False
    if deadlocks[a2] or a2 in a1:
        return True
    a1.append(a2)
    deadlocks[a2] = check_deadlocks(a1, deadlocks, env)
    if deadlocks[a2]:
        return True
    del a1[-1]
    return False


def check_invalid_transitions(action_dict, action_mask, invalid_action_penalty):
    return {a: invalid_action_penalty if a in action_dict and mask[action_dict[a]] == 0 else 0 for a, mask in
            enumerate(action_mask)}


def check_stop_transition(action_dict, rewards, stop_penalty):
    return {a: stop_penalty if action_dict[a] == 4 else rewards[a] for a in range(len(action_dict))}


def step_shaping(env, action_dict, deadlocks, shortest_path, action_mask, invalid_action_penalty,
                 stop_penalty, deadlock_penalty, shortest_path_penalty_coefficient, done_bonus):

    invalid_rewards_shaped = check_invalid_transitions(action_dict, action_mask, invalid_action_penalty)
    stop_rewards_shaped = check_stop_transition(action_dict, invalid_rewards_shaped, stop_penalty)

    # Environment step
    obs, rewards, done, info = env.step(action_dict)

    agents = []
    for a in range(env.get_num_agents()):
        if not done[a]:
            agents.append(a)
            if not deadlocks[a]:
                deadlocks[a] = check_deadlocks(agents, deadlocks, env)
            if not (deadlocks[a]):
                del agents[-1]

    new_shortest_path = [obs.get(a)[6] if obs.get(a) is not None else 0 for a in range(env.get_num_agents())]

    new_rewards_shaped = {
        a: rewards[a] if stop_rewards_shaped[a] == 0 else rewards[a] + stop_rewards_shaped[a]
        for a in range(env.get_num_agents())}

    rewards_shaped_shortest_path = {a: shortest_path_penalty_coefficient * new_rewards_shaped[a]
    if shortest_path[a] < new_shortest_path[a] else new_rewards_shaped[a] for a in range(env.get_num_agents())}

    rewards_shaped_deadlocks = {a: deadlock_penalty if deadlocks[a] and deadlock_penalty != 0
    else rewards_shaped_shortest_path[a] for a in range(env.get_num_agents())}

    rewards_shaped = {a: done_bonus if done[a] else rewards_shaped_deadlocks[a] for a in range(env.get_num_agents())}

    return obs, rewards, done, info, rewards_shaped, deadlocks, new_shortest_path


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

    # Custom observations&rewards
    custom_observations = env_params.custom_observations
    stop_penalty = env_params.stop_penalty
    invalid_action_penalty = env_params.invalid_action_penalty
    done_bonus = env_params.done_bonus
    deadlock_penalty = env_params.deadlock_penalty
    shortest_path_penalty_coefficient = env_params.shortest_path_penalty_coefficient

    # Training setup parameters
    n_episodes = train_params.n_episodes
    checkpoint_interval = train_params.checkpoint_interval
    horizon = train_params.horizon

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        # Fast passenger train
        1.: 1.0,
        # Fast freight train
        1. / 2.: 0.0,
        # Slow commuter train
        1. / 3.: 0.0,
        # Slow freight train
        1. / 4.: 0.0
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

    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes + custom_observations * 6

    # The action space of flatland is 5 discrete actions
    action_size = env.action_space[0]

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))

    action_count = [0] * action_size
    smoothed_normalized_score = -1.0
    smoothed_completion = 0.0

    memory = Memory(n_agents)

    ppo = PsPPO(state_size + 1,
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
                train_params.advantage_estimator,
                train_params.value_loss_function,
                train_params.entropy_coefficient,
                train_params.value_loss_coefficient)

    # TensorBoard writer
    """
    writer = SummaryWriter()
    writer.add_hparams(vars(train_params), {})
    writer.add_hparams(vars(env_params), {})
    """

    training_timer = Timer()
    training_timer.start()

    print("\nTraining {} trains on {}x{} grid for {} episodes. Update every {} timesteps.\n"
          .format(env.get_num_agents(), x_dim, y_dim, n_episodes, horizon))

    path = "model.pt"

    skip_cells = [int("1000000000100000", 2),
                  RailEnvTransitions().rotate_transition(int("1000000000100000", 2), 90),
                  int("0001001000000000", 2),
                  RailEnvTransitions().rotate_transition(int("0001001000000000", 2), 90),
                  RailEnvTransitions().rotate_transition(int("0001001000000000", 2), 180),
                  RailEnvTransitions().rotate_transition(int("0001001000000000", 2), 270)]

    for episode in range(n_episodes + 1):

        # Timers
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()

        agent_obs = [None] * env.get_num_agents()
        not_arrived_agents = set(range(n_agents))

        # Reset environment
        reset_timer.start()
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        reset_timer.end()

        # Setup renderer
        if train_params.render:
            env_renderer = RenderTool(env, gl="PGL")
        else:
            env_renderer = None

        if train_params.render:
            env_renderer.set_new_rail()

        score = 0

        deadlocks = [False for _ in range(env.get_num_agents())]
        shortest_path = [obs.get(a)[6] if obs.get(a) is not None else 0 for a in range(env.get_num_agents())]

        # Run episode
        for step in range(max_steps):

            action_mask = [[1 * (0 if action == 0 and not train_params.allow_no_op else 1)
                            for action in range(action_size)] for _ in not_arrived_agents]

            action_dict = dict()
            agents_in_action = set()

            # Collect and preprocess observations and fill action dictionary
            for agent in env.get_agent_handles():
                # Agents always enter here at least once so there is no further controls
                # When obs is absent the agent has arrived and the observation remains the same
                if obs[agent]:
                    preproc_timer.start()
                    agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth,
                                                             observation_radius=observation_radius)
                    if custom_observations:
                        if env.agents[agent].position is None:
                            pos_a_x = env.agents[agent].initial_position[0] / env.width
                            pos_a_y = env.agents[agent].initial_position[1] / env.height
                            a_direction = env.agents[agent].initial_direction / 4
                        else:
                            pos_a_x = env.agents[agent].position[0] / env.width
                            pos_a_y = env.agents[agent].position[1] / env.height
                            a_direction = env.agents[agent].direction / 4
                            """
                            rail_cell = env.rail.get_full_transitions(env.agents[agent].position[0],
                                                                      env.agents[agent].position[1])
                            print(rail_cell in skip_cells)
                            """

                        agent_obs[agent] = np.append(agent_obs[agent], [pos_a_x, pos_a_y, a_direction])
                        agent_obs[agent] = np.append(agent_obs[agent], [env.agents[agent].target[0] / env.width,
                                                                        env.agents[agent].target[1] / env.height])
                        if deadlocks[agent]:
                            agent_obs[agent] = np.append(agent_obs[agent], [1])
                        else:
                            agent_obs[agent] = np.append(agent_obs[agent], [0])

                    # Action mask modification only if action masking is True
                    if train_params.action_masking:
                        for action in range(action_size):
                            if env.agents[agent].status != RailAgentStatus.READY_TO_DEPART:
                                _, cell_valid, _, _, transition_valid = env._check_action_on_agent(
                                    RailEnvActions(action),
                                    env.agents[agent])
                                if not all([cell_valid, transition_valid]):
                                    action_mask[agent][action] = 0

                    preproc_timer.end()

                # Fill action dict
                # If an agent can skip the movement
                if deadlocks[agent] and custom_observations:
                    action_dict[agent] = \
                        ppo.policy_old.act(np.append(agent_obs[agent], [agent]), memory, action_mask[agent],
                                           action=torch.tensor(int(RailEnvActions.DO_NOTHING)).to(device))
                    agents_in_action.add(agent)
                elif train_params.action_skipping \
                        and env.agents[agent].position is not None and env.rail.get_full_transitions(
                    env.agents[agent].position[0], env.agents[agent].position[1]) in skip_cells:
                    # We always insert in memory the last time step
                    if step == max_steps - 1:
                        action_dict[agent] = \
                            ppo.policy_old.act(np.append(agent_obs[agent], [agent]), memory, action_mask[agent],
                                               action=torch.tensor(int(RailEnvActions.MOVE_FORWARD)).to(device))
                        agents_in_action.add(agent)
                    # Skip action
                    else:
                        action_dict[agent] = int(RailEnvActions.MOVE_FORWARD)
                # Else
                elif info["status"][agent] in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                    action_dict[agent] = \
                        ppo.policy_old.act(np.append(agent_obs[agent], [agent]), memory, action_mask[agent],
                                           action=torch.tensor(int(RailEnvActions.DO_NOTHING)).to(device))
                    agents_in_action.add(agent)
                else:
                    action_dict[agent] = \
                        ppo.policy_old.act(np.append(agent_obs[agent], [agent]), memory, action_mask[agent])
                    agents_in_action.add(agent)

            for a in list(action_dict.values()):
                action_count[a] += 1

            # Environment step
            """
            step_timer.start()
            obs, rewards, done, info = env.step(action_dict)
            step_timer.end()
            """
            step_timer.start()
            obs, rewards, done, info, rewards_shaped, new_deadlocks, new_shortest_path = \
                step_shaping(env, action_dict, deadlocks, shortest_path, action_mask, invalid_action_penalty,
                             stop_penalty, deadlock_penalty, shortest_path_penalty_coefficient, done_bonus)
            step_timer.end()

            # TODO update not_arrived
            # [not_arrived_agents.remove(a) if d and a != "__all__" else None for a, d in done.items()]

            deadlocks = new_deadlocks
            shortest_path = new_shortest_path

            total_timestep_reward = np.sum(list(rewards.values()))
            score += total_timestep_reward
            total_timestep_reward_shaped = np.sum(list(rewards_shaped.values()))

            # Update dones and rewards for each agent that performed act()
            for a in agents_in_action:
                memory.rewards[a].append(total_timestep_reward_shaped)
                memory.dones[a].append(done['__all__'])

                # Set dones to True when the episode is finished because the maximum number of steps has been reached
                if step == max_steps - 1:
                    memory.dones[a][-1] = True

            # For each agent
            for a in not_arrived_agents:
                # Update
                if len(memory.states[a]) % (horizon + 1) == 0:
                    learn_timer.start()
                    ppo.update(memory, a)
                    learn_timer.end()

                    """
                    Leave last memory unit because the batch includes an additional step which has not been considered 
                    in the current trajectory (it has been inserted to compute the advantage) but must be considered in 
                    the next trajectory or will be lost.
                    """
                    memory.clear_memory_except_last(a)

            if train_params.render:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

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
            print("..saving model..")
            # torch.save(ppo.policy.state_dict(), path)
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
        if episode % train_params.checkpoint_interval == 0:
            scores, completions, nb_steps_eval = eval_policy(env, policy, n_eval_episodes, max_steps)
            writer.add_scalar("evaluation/scores_min", np.min(scores), episode)
            writer.add_scalar("evaluation/scores_max", np.max(scores),episode)
            writer.add_scalar("evaluation/scores_mean", np.mean(scores), episode)
            writer.add_scalar("evaluation/scores_std", np.std(scores), episode)
            writer.add_histogram("evaluation/scores", np.array(scores), episode)
            writer.add_scalar("evaluation/completions_min", np.min(completions), episode)
            writer.add_scalar("evaluation/completions_max", np.max(completions), episode)
            writer.add_scalar("evaluation/completions_mean", np.mean(completions), episode)
            writer.add_scalar("evaluation/completions_std", np.std(completions), episode)
            writer.add_histogram("evaluation/completions", np.array(completions), episode)
            writer.add_scalar("evaluation/nb_steps_min", np.min(nb_steps_eval), episode)
            writer.add_scalar("evaluation/nb_steps_max", np.max(nb_steps_eval), episode)
            writer.add_scalar("evaluation/nb_steps_mean", np.mean(nb_steps_eval), episode)
            writer.add_scalar("evaluation/nb_steps_std", np.std(nb_steps_eval), episode)
            writer.add_histogram("evaluation/nb_steps", np.array(nb_steps_eval), episode)
            smoothing = 0.9
            smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(scores) * (1.0 - smoothing)
            smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)
            writer.add_scalar("evaluation/smoothed_score", smoothed_eval_normalized_score, episode)
            writer.add_scalar("evaluation/smoothed_completion", smoothed_eval_completion, episode)
        # Save logs to tensorboard
        writer.add_scalar("training/score", normalized_score, episode)
        writer.add_scalar("training/smoothed_score", smoothed_normalized_score, episode)
        writer.add_scalar("training/completion", np.mean(completion), episode)
        writer.add_scalar("training/smoothed_completion", np.mean(smoothed_completion), episode)
        writer.add_scalar("training/nb_steps", nb_steps, episode)
        writer.add_histogram("actions/distribution", np.array(actions_taken),episode)
        writer.add_scalar("actions/nothing", action_probs[RailEnvActions.DO_NOTHING], episode)
        writer.add_scalar("actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode)
        writer.add_scalar("actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode)
        writer.add_scalar("actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode)
        writer.add_scalar("actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode)
        writer.add_scalar("training/epsilon", eps_start, episode)
        writer.add_scalar("training/buffer_size", len(policy.memory), episode)
        writer.add_scalar("training/loss", policy.loss, episode)
        writer.add_scalar("timer/reset", reset_timer.get(), episode)
        writer.add_scalar("timer/step", step_timer.get(), episode)
        writer.add_scalar("timer/learn", learn_timer.get(), episode)
        writer.add_scalar("timer/preproc", preproc_timer.get(), episode)
        writer.add_scalar("timer/total", training_timer.get_current(), episode)
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
    "n_cities": 2,
    "max_rails_between_cities": 2,
    "max_rails_in_city": 3,

    "seed": myseed,
    "observation_tree_depth": 5,
    "observation_radius": 35,
    "observation_max_path_depth": 30,
    # ====================
    # Custom observations&rewards
    # ====================
    "custom_observations": False,

    "stop_penalty": -2.0,
    "invalid_action_penalty": -2.0,
    "deadlock_penalty": -5.0,
    "shortest_path_penalty_coefficient": 1.5,
    # 1.0 for skipping
    "done_bonus": 1.0,
}

training_parameters = {
    "random_seed": myseed,
    # ====================
    # Network architecture
    # ====================
    # Shared actor-critic layer
    # If shared is True then the considered sizes are taken from the critic
    "shared": True,
    # Policy network
    "critic_mlp_width": 1024,
    "critic_mlp_depth": 16,
    "last_critic_layer_scaling": 0.01,
    # Actor network
    "actor_mlp_width": 512,
    "actor_mlp_depth": 16,
    "last_actor_layer_scaling": 0.01,
    # Adam learning rate
    "learning_rate": 0.001,
    # Adam epsilon
    "adam_eps": 1e-5,
    # Activation
    "activation": "Tanh",
    "lmbda": 0.95,
    "entropy_coefficient": 0.1,
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
    # gae, n-steps
    "advantage_estimator": "gae",
    # huber or mse
    "value_loss_function": "mse",

    # ==========================
    # Optimization and rendering
    # ==========================
    "checkpoint_interval": 100,
    "use_gpu": False,
    "num_threads": 1,
    "render": False,

    # ==========================
    # Action Masking / Skipping
    # ==========================
    "action_masking": True,
    "allow_no_op": False,
    "action_skipping": True
}

train_multiple_agents(Namespace(**environment_parameters), Namespace(**training_parameters))
