import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class PsPPOPolicy(nn.Module):
    """
    The policy of the PS-PPO algorithm.
    """
    def __init__(self,
                 state_size,
                 action_size,
                 device,
                 train_params):
        """
        :param state_size: The number of attributes of each state
        :param action_size: The number of available actions
        :param train_params: Parameters to influence training
        """

        super(PsPPOPolicy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.activation = train_params.activation
        self.softmax = nn.Softmax(dim=-1)

        # Network creation
        critic_layers = self._build_network(False, train_params.critic_mlp_depth, train_params.critic_mlp_width)
        self.critic_network = nn.Sequential(critic_layers)
        if not train_params.shared:
            self.actor_network = nn.Sequential(self._build_network(True, train_params.actor_mlp_depth,
                                                                   train_params.actor_mlp_width))
        else:
            if train_params.critic_mlp_depth <= 1:
                raise Exception("Shared networks must have depth greater than 1")
            actor_layers = critic_layers.copy()
            actor_layers.popitem()
            actor_layers["actor_output_layer"] = nn.Linear(train_params.critic_mlp_width, action_size)
            self.actor_network = nn.Sequential(actor_layers)

        # Network orthogonal initialization
        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

        with torch.no_grad():
            self.critic_network.apply(weights_init)
            self.actor_network.apply(weights_init)

            # Last layer's weights rescaling
            list(self.critic_network.children())[-1].weight.mul_(train_params.last_critic_layer_scaling)
            list(self.actor_network.children())[-1].weight.mul_(train_params.last_actor_layer_scaling)

        # Load from file if available
        if train_params.load_model_path is not None:
            self.load(train_params.load_model_path)

    def _build_network(self, is_actor, nn_depth, nn_width):
        """
        Creates the network, including activation layers.
        The actor is not completed with the final softmax layer.

        :param is_actor: True if the resulting network will be used as the actor
        :param nn_depth: The number of layers included the first and last
        :param nn_width: The number of nodes in each hidden layer
        :return: an OrderedDict used to build the neural network
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
            print(self.activation)
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
        state = torch.from_numpy(state).float().to(self.device)
        action_logits = self.actor_network(state)

        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)

        # Action masking, default values are True, False are present only if masking is enabled.
        # If No op is not allowed it is masked even if masking is not active
        action_logits = torch.where(action_mask, action_logits, torch.tensor(-1e+8).to(self.device))

        action_probs = self.softmax(action_logits)

        """
        From the paper: "The stochastic policy πθ can be represented by a categorical distribution when the actions of
        the agent are discrete and by a Gaussian distribution when the actions are continuous."
        """
        action_distribution = Categorical(action_probs)

        if action is None:
            action = action_distribution.sample()

        # Memory is updated
        if memory is not None:
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
        action_logits = torch.where(action_mask[:-1], action_logits, torch.tensor(-1e+8).to(self.device))

        action_probs = self.softmax(action_logits)

        action_distribution = Categorical(action_probs)

        return action_distribution.log_prob(action[:-1]), self.critic_network(state), action_distribution.entropy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
        else:
            print("Loading file failed. File not found.")
