import copy
import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.common.policy import Policy
from src.d3qn.memory import ReplayBuffer
from src.d3qn.model import DuelingQNetwork


class D3QNPolicy(Policy):
    """Dueling Double DQN policy"""

    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode

        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = True
        self.hidsize = 1

        if not evaluation_mode:
            self.hidsize = parameters.hidden_size
            self.buffer_size = parameters.buffer_size
            self.batch_size = parameters.batch_size
            self.update_every = parameters.update_every
            self.learning_rate = parameters.learning_rate
            self.tau = parameters.tau
            self.gamma = parameters.gamma
            self.buffer_min_size = parameters.buffer_min_size

        # Device
        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, hidsize1=self.hidsize, hidsize2=self.hidsize).to(
            self.device)

        if not evaluation_mode:
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
            self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)

            self.t_step = 0
            self.loss = 0.0

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.double_dqn:
            # Double DQN
            q_best_action = self.qnetwork_local(next_states).max(1)[1]
            q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        else:
            # DQN
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute loss
        self.loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def _soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), "local_" + filename)
        torch.save(self.qnetwork_target.state_dict(), "target_" + filename)

    def load(self, filename):
        if os.path.exists("local_" + filename):
            self.qnetwork_local.load_state_dict(torch.load("local_" + filename))
        if os.path.exists("target_" + filename):
            self.qnetwork_target.load_state_dict(torch.load("target_" + filename))
