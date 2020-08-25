import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-network (https://arxiv.org/abs/1511.06581)
    """

    def __init__(self, state_size, action_size, parameters, evaluation_mode):
        super(DuelingQNetwork, self).__init__()
        self.shared = parameters.shared
        self.base_modules = nn.ModuleList([])
        self.critic_modules = nn.ModuleList([])
        self.actor_modules = nn.ModuleList([])

        if parameters.shared:
            self.base_modules.append(nn.Linear(state_size, parameters.hidden_size))
            for i in range(parameters.hidden_layers):
                self.base_modules.append(nn.Linear(parameters.hidden_size, parameters.hidden_size))
        else:
            self.critic_modules.append(nn.Linear(state_size, parameters.hidden_size))
            self.actor_modules.append(nn.Linear(state_size, parameters.hidden_size))

            for i in range(parameters.hidden_layers):
                self.critic_modules.append(nn.Linear(parameters.hidden_size, parameters.hidden_size))
                self.actor_modules.append(nn.Linear(parameters.hidden_size, parameters.hidden_size))

        # Last layer is always separated
        self.critic_modules.append(nn.Linear(parameters.hidden_size, 1))
        self.actor_modules.append(nn.Linear(parameters.hidden_size, action_size))

        loading = False

        if parameters.load_model_path is not None:
            loading = self.load(parameters.load_model_path)
        if evaluation_mode and not(loading):
            sys.exit()

    def forward(self, x):
        val = x
        adv = x

        for i in range(len(self.base_modules)):
            val = F.relu(self.base_modules[i](val))
            adv = F.relu(self.base_modules[i](adv))

        for i in range(len(self.critic_modules) - 1):
            val = self.critic_modules[i](val)

        for i in range(len(self.actor_modules) - 1):
            adv = self.actor_modules[i](adv)

        val = self.critic_modules[-1](val)
        adv = self.actor_modules[-1](adv)

        return val + adv - adv.mean()

    def load(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            return True
        else:
            print("Loading file failed. File not found.")
            return False
