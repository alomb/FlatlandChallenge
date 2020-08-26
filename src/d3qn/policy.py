import copy
import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.common.policy import Policy
from src.d3qn.memory import UniformExperienceReplay, PrioritisedExperienceReplay
from src.d3qn.model import DuelingQNetwork


class D3QNPolicy(Policy):
    """
    Dueling Double DQN policy
    """

    def __init__(self, state_size, action_size, parameters):
        self.evaluation_mode = parameters.evaluation_mode

        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = parameters.double_dqn
        self.hidsize = 1

        if not parameters.evaluation_mode:
            self.hidsize = parameters.hidden_size
            self.buffer_size = parameters.buffer_size
            self.batch_size = parameters.batch_size
            self.update_every = parameters.update_every
            self.learning_rate = parameters.learning_rate
            self.tau = parameters.tau
            self.gamma = parameters.gamma
            self.buffer_min_size = parameters.buffer_min_size

        # Device
        self.device = torch.device("cuda:0" if parameters.use_gpu and torch.cuda.is_available() else "cpu")

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, parameters, parameters.evaluation_mode).to(self.device)

        if not parameters.evaluation_mode:
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

            if parameters.memory_type.lower() == "uer":
                self.memory = UniformExperienceReplay(self.buffer_size, self.batch_size, self.device)
            elif parameters.memory_type.lower() == "per":
                self.memory = PrioritisedExperienceReplay(self.buffer_size, self.batch_size, self.device)
            else:
                raise Exception("Unknown experience replay \"{}\"".format(parameters.memory_type))

            self.t_step = 0
            self.loss = 0.0

    def act(self, state, action_mask=None, eps=0.):
        """

        :param state: the state to act on
        :param action_mask: a list of 0 and 1 where 0 indicates that the index's action should be not sampled
        :param eps: the epsilon-greedy factor to influence the exploration-exploitation tradeoff
        :return:


        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.qnetwork_local(state)

            if action_mask is not None:
                action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
                action_values = torch.where(action_mask, action_values, torch.tensor(-1e+8).to(self.device))

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Randomly choose an available action
            masked_actions = torch.nonzero(action_values > -1e+8, as_tuple=False)[:, 1]
            choice = torch.multinomial(masked_actions.float(), 1)
            return masked_actions[choice].item()

    def step(self, state, action, reward, next_state, done):
        """
        Perform a step if the evaluation mode is off, updating the memory and updating the networks if the agent has
        already self.update_every steps
        """
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        # Save experience in replay memory
        if type(self.memory) is UniformExperienceReplay:
            self.memory.add(state, action, reward, next_state, done)
        else:
            old_val = self.qnetwork_local(torch.tensor(state, dtype=torch.float32).to(self.device))[action]

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)

            if self.double_dqn:
                # Access at [1] because max returns values and indices, here indices correspond to actions
                q_best_action = self.qnetwork_local(next_state_tensor).max(0)[1]
                target_val = self.qnetwork_target(next_state_tensor)[q_best_action]
            else:
                target_val = self.qnetwork_target(next_state_tensor).max(0)[0]

            new_val = reward + self.gamma * target_val * (1 - done)

            error = abs(new_val - old_val).item()

            self.memory.add(state, action, reward, next_state, done, error)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
        if type(self.memory) is UniformExperienceReplay:
            experiences = self.memory.sample()
            indexes = None
            is_weights = None
        else:
            experiences, indexes, is_weights = self.memory.sample()

        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.double_dqn:
            # Access at [1] because max returns values and indices, here indices correspond to actions
            q_best_action = self.qnetwork_local(next_states).max(1)[1]
            q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        else:
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        if type(self.memory) is UniformExperienceReplay:
            # Compute loss
            self.loss = F.mse_loss(q_expected, q_targets)
        else:
            errors = torch.abs(q_expected - q_targets).data.numpy()

            # Update priority
            for i in range(self.batch_size):
                index = indexes[i]
                self.memory.update(index, errors[i])

            # Compute loss
            self.loss = (torch.tensor(is_weights, dtype=torch.float32).to(self.device) *
                         F.mse_loss(q_expected, q_targets)).mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # To show graph
        """
        from datetime import datetime
        from torchviz import make_dot
        now = datetime.now()
        make_dot(self.loss).render("attached" + now.strftime("%H-%M-%S"), format="png")
        exit()
        """

        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        """
        Save networks' params
        :param filename: the path where the file is saved
        """
        torch.save(self.qnetwork_local.state_dict(), "local_" + filename)
        torch.save(self.qnetwork_target.state_dict(), "target_" + filename)

    def load(self, filename):
        """
        Load networks' params
        :param filename: the path where the params are saved
        """
        if os.path.exists("local_" + filename):
            self.qnetwork_local.load_state_dict(torch.load("local_" + filename))
        if os.path.exists("target_" + filename):
            self.qnetwork_target.load_state_dict(torch.load("target_" + filename))
