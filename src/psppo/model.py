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
        self.is_shared = train_params.shared
        self.is_recurrent = train_params.shared_recurrent

        # Network creation
        if self.is_recurrent:
            # The recurrent case with shared linear and lstm cell
            self.fc = nn.Linear(state_size, train_params.linear_size)
            self.fc_activation = self._get_activation()
            self.lstm = nn.LSTM(train_params.linear_size, train_params.hidden_size)
            self.fc_actor = nn.Linear(train_params.hidden_size, action_size)
            self.fc_critic = nn.Linear(train_params.hidden_size, 1)
        else:
            # The separate network case
            if not train_params.shared:
                critic_layers = self._build_network(False, train_params.critic_mlp_depth, train_params.critic_mlp_width)
                critic_layers["critic_last_layer"] = nn.Linear(train_params.critic_mlp_width, 1)
                self._critic_network = nn.Sequential(critic_layers)

                actor_layers = self._build_network(True, train_params.actor_mlp_depth, train_params.actor_mlp_width)
                actor_layers["actor_last_layer"] = nn.Linear(train_params.actor_mlp_width, self.action_size)
                self._actor_network = nn.Sequential(actor_layers)
            else:
                # The shared case
                assert train_params.critic_mlp_depth == train_params.actor_mlp_depth, \
                    "The network is shared, critic and actor hidden layers depth should be the same!"
                assert train_params.critic_mlp_width == train_params.actor_mlp_width, \
                    "The network is shared, critic and actor hidden layers width should be the same!"

                base_layers = self._build_network(False, train_params.critic_mlp_depth, train_params.critic_mlp_width)
                self._base_network = nn.Sequential(base_layers)

                self.fc_critic = nn.Linear(train_params.critic_mlp_width, 1)
                self.fc_actor = nn.Linear(train_params.critic_mlp_width, self.action_size)

        # Network orthogonal initialization
        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, nn.init.calculate_gain(self.activation.lower()))
                torch.nn.init.zeros_(m.bias)

        with torch.no_grad():
            if self.is_recurrent:
                # Initialization
                self.fc.apply(weights_init)
                self.fc_actor.apply(weights_init)
                self.fc_critic.apply(weights_init)

                # Last layer's weights rescaling
                self.fc_actor.weight.mul_(train_params.last_critic_layer_scaling)
                self.fc_critic.weight.mul_(train_params.last_actor_layer_scaling)
            else:
                if not train_params.shared:
                    # Initialization
                    self._critic_network.apply(weights_init)
                    self._actor_network.apply(weights_init)

                    # Last layer's weights rescaling
                    list(self._critic_network.children())[-1].weight.mul_(train_params.last_critic_layer_scaling)
                    list(self._actor_network.children())[-1].weight.mul_(train_params.last_actor_layer_scaling)
                else:
                    # Initialization
                    self._base_network.apply(weights_init)
                    self.fc_actor.apply(weights_init)
                    self.fc_critic.apply(weights_init)

                    # Last layer's weights rescaling
                    self.fc_actor.weight.mul_(train_params.last_critic_layer_scaling)
                    self.fc_critic.weight.mul_(train_params.last_actor_layer_scaling)

        # Load from file if available
        loading = False

        if "load_model_path" in train_params and train_params.load_model_path is not None:
            loading = self.load(train_params.load_model_path)
        if self.evaluation_mode and not loading:
            # If the network is called in evaluation mode but there is no a model to load the execution is
            # terminated
            sys.exit()

    def _build_network(self, is_actor, nn_depth, nn_width):
        """
        Creates the hidden part of the network, including activation layers.
        The network must be completed with the final layers/softmax.

        :param is_actor: True if the resulting network will be used as the actor
        :param nn_depth: The number of layers included the first and last
        :param nn_width: The number of nodes in each hidden layer
        :return: an OrderedDict used to build the neural network
        """
        if nn_depth <= 1:
            raise Exception("Networks' depths must be greater than 1")

        network = OrderedDict()
        nn_type = "actor" if is_actor else "critic"

        # First layer
        network["%s_input" % nn_type] = nn.Linear(self.state_size, nn_width)
        network["%s_input_activation(%s)" % (nn_type, self.activation)] = self._get_activation()

        # Add hidden layers
        for layer in range(1, nn_depth - 1):
            layer_name = "%s_layer_%d" % (nn_type, layer)

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
            if self.is_shared:
                x = self._base_network(state)
                action_logits = self.fc_actor(x)
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
            if self.is_shared:
                x = self._base_network(state)
                action_logits = self.fc_actor(x)
                value = self.fc_critic(x)
            else:
                action_logits = self._actor_network(state)
                value = self._critic_network(state)

            # Action masking, default values are True, False are present only if masking is enabled.
            # If No op is not allowed it is masked even if masking is not active
            action_logits = torch.where(action_mask, action_logits, self.masking_value)

            return self.softmax(action_logits), value

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
        try:
            torch.save(self.state_dict(), path)
        except FileNotFoundError:
            print("\nCould not save the model because the desired path doesn't exist.")

    def load(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            return True
        else:
            print("Loading file failed. File not found.")
            return False
