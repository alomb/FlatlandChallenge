import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-network neural network.
    """

    def __init__(self, state_size, action_size, parameters):
        """
        :param state_size: The number of attributes of each state. It affects the size of the input of the nn.
        :param action_size: The number of available actions. It affects the size of the output of the nn.
        :param parameters: The set of parameters which affect the architecture of the nn.
        """

        super(DuelingQNetwork, self).__init__()
        self.shared = parameters.shared
        self.base_modules = nn.ModuleList([])
        self.value_modules = nn.ModuleList([])
        self.advantage_modules = nn.ModuleList([])

        # Shared network
        if parameters.shared:
            self.base_modules.append(nn.Linear(state_size, parameters.hidden_size))
            for i in range(parameters.hidden_layers):
                self.base_modules.append(nn.Linear(parameters.hidden_size, parameters.hidden_size))
        # Separated networks
        else:
            # First layers
            self.value_modules.append(nn.Linear(state_size, parameters.hidden_size))
            self.advantage_modules.append(nn.Linear(state_size, parameters.hidden_size))

            # Hidden layers
            for i in range(parameters.hidden_layers):
                self.value_modules.append(nn.Linear(parameters.hidden_size, parameters.hidden_size))
                self.advantage_modules.append(nn.Linear(parameters.hidden_size, parameters.hidden_size))

        # Last layers (both separated or two-head cases)
        self.value_modules.append(nn.Linear(parameters.hidden_size, 1))
        self.advantage_modules.append(nn.Linear(parameters.hidden_size, action_size))

    def forward(self, x):
        """
        :param x: input to the nn.

        :return set of Q-values the same size as action_size.
        """
        base = x
        for i in range(len(self.base_modules)):
            base = F.relu(self.base_modules[i](base))

        # Compute state value
        val = base
        for i in range(len(self.value_modules) - 1):
            val = self.value_modules[i](val)

        # Compute advantage
        adv = base
        for i in range(len(self.advantage_modules) - 1):
            adv = self.advantage_modules[i](adv)

        val = self.value_modules[-1](val)
        adv = self.advantage_modules[-1](adv)

        # Combine state value and advantage
        return val + adv - adv.mean()
