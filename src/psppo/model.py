import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn


class PsPPO(nn.Module):
    """
    The neural network for the PS-PPO algorithm.
    """
    def __init__(self,
                 state_size,
                 action_size,
                 masking_value,
                 train_params):
        """
        :param state_size: The number of attributes of each state
        :param action_size: The number of available actions
        :param train_params: Parameters to influence training
        """

        super(PsPPO, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.masking_value = masking_value
        self.activation = train_params.activation
        self.softmax = nn.Softmax(dim=-1)
        self.evaluation_mode = train_params.evaluation_mode
        self.recurrent_linear_size = train_params.linear_size
        self.is_recurrent = train_params.shared_recurrent

        # Network creation
        if self.is_recurrent:
            self.fc = nn.Linear(state_size, train_params.linear_size)
            self.fc_activation = self._get_activation()
            self.lstm = nn.LSTM(train_params.linear_size, train_params.hidden_size)
            self.fc_actor = nn.Linear(train_params.hidden_size, action_size)
            self.fc_critic = nn.Linear(train_params.hidden_size, 1)
        else:
            critic_layers = self._build_network(False, train_params.critic_mlp_depth, train_params.critic_mlp_width)
            self._critic_network = nn.Sequential(critic_layers)
            if not train_params.shared:
                self._actor_network = nn.Sequential(self._build_network(True, train_params.actor_mlp_depth,
                                                                        train_params.actor_mlp_width))
            else:
                if train_params.critic_mlp_depth <= 1:
                    raise Exception("Shared networks must have depth greater than 1")
                # Shallow layers copy
                actor_layers = critic_layers.copy()
                actor_layers.popitem()
                actor_layers["actor_output_layer"] = nn.Linear(train_params.critic_mlp_width, action_size)
                self._actor_network = nn.Sequential(actor_layers)

        # Network orthogonal initialization
        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, nn.init.calculate_gain(self.activation.lower()))
                torch.nn.init.zeros_(m.bias)

        with torch.no_grad():
            # Last layer's weights rescaling
            if self.is_recurrent:
                self.fc.apply(weights_init)
                self.fc_actor.apply(weights_init)
                self.fc_critic.apply(weights_init)

                self.fc_actor.weight.mul_(train_params.last_critic_layer_scaling)
                self.fc_critic.weight.mul_(train_params.last_actor_layer_scaling)
            else:
                self._critic_network.apply(weights_init)
                self._actor_network.apply(weights_init)

                list(self._critic_network.children())[-1].weight.mul_(train_params.last_critic_layer_scaling)
                list(self._actor_network.children())[-1].weight.mul_(train_params.last_actor_layer_scaling)

        # Load from file if available
        loading = False

        if train_params.load_model_path is not None:
            loading = self.load(train_params.load_model_path)
        if self.evaluation_mode and not loading:
            sys.exit()

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

    def act_forward(self, state, action_mask, hidden=None):
        """
        Method called to sample trajectories by the policy.

        :param state: the state to act on
        :param action_mask: a boolean tensor which indicates that an action should be not sampled
        :param hidden: the previous step hidden value
        :return: action probabilities and the hidden state if the architecture is recurrent
        """
        if self.is_recurrent:

            assert hidden is not None, "The network is recurrent but no hidden value has been passed, this should " \
                                       "not happen!"

            x = self.fc_activation(self.fc(state))
            # during act the sequence will contain only an element
            x = x.view(-1, 1, self.recurrent_linear_size)
            # lstm_hidden is a tuple containing hidden state (short-term memory) and cell state (long-term memory)
            x, lstm_hidden = self.lstm(x, hidden)
            x = self.fc_actor(x)
            x = x.squeeze()
            # action masking
            x = torch.where(action_mask, x, self.masking_value)

            return self.softmax(x), lstm_hidden
        else:
            action_logits = self._actor_network(state)
            # action masking
            action_logits = torch.where(action_mask, action_logits, self.masking_value)

            return self.softmax(action_logits), None

    def evaluate_forward(self, state, action_mask, hidden=None):
        """
        Method called by the policy to obtain state values estimations usually on a batch of experience.

        :param state: the states to evaluate.
        :param action_mask: a boolean tensor which indicates that an action should be not sampled
        :param hidden: the first hidden value to initialize the lstm
        :return: action probabilities and state value
        """
        if self.is_recurrent:
            x = self.fc_activation(self.fc(state))
            x = x.view(-1, 1, self.recurrent_linear_size)
            # lstm_hidden is a tuple containing  hidden state (short-term memory) and cell state (long-term memory)
            x, lstm_hidden = self.lstm(x, hidden)
            # Actor
            action_probs = self.fc_actor(x)
            action_probs = action_probs.squeeze()
            action_probs = torch.where(action_mask, action_probs, self.masking_value)
            action_probs = self.softmax(action_probs)
            # Critic
            value = self.fc_critic(x)
            value = value.squeeze()

            return action_probs, value
        else:
            action_logits = self._actor_network(state)

            # Action masking, default values are True, False are present only if masking is enabled.
            # If No op is not allowed it is masked even if masking is not active
            action_logits = torch.where(action_mask, action_logits, self.masking_value)

            return self.softmax(action_logits), self._critic_network(state)

    def _get_activation(self):
        """

        :return: the current activation function
        """
        if self.activation == "ReLU":
            return nn.ReLU()
        elif self.activation == "Tanh":
            return nn.Tanh()
        else:
            print(self.activation)
            raise Exception("The specified activation function don't exists or is not available")

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            return True
        else:
            print("Loading file failed. File not found.")
            return False
