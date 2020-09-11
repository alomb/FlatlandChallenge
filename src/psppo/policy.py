import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.common.policy import Policy
from src.psppo.memory import Memory
from src.psppo.model import PsPPO


class PsPPOPolicy(Policy):
    """
    The class responsible of some logics of the algorithm especially of the loss computation and updating of the policy.
    """

    def __init__(self,
                 state_size,
                 action_size,
                 train_params,
                 n_agents):
        """
        :param state_size: The number of attributes of each state
        :param action_size: The number of available actions
        :param train_params: Parameters to influence training
        :param n_agents: number of agents
        """

        super(PsPPOPolicy, self).__init__()
        # Device
        self.device = torch.device("cuda:0" if train_params.use_gpu and torch.cuda.is_available() else "cpu")
        self.n_agents = n_agents
        self.shared = train_params.shared
        self.learning_rate = train_params.learning_rate
        self.discount_factor = train_params.discount_factor
        self.epochs = train_params.epochs
        self.batch_size = train_params.batch_size
        self.batch_shuffle = True if train_params.batch_mode.lower() == "shuffle" else False
        self.horizon = train_params.horizon
        self.eps_clip = train_params.eps_clip
        self.max_grad_norm = train_params.max_grad_norm
        self.lmbda = train_params.lmbda
        self.value_loss_coefficient = train_params.value_loss_coefficient
        self.entropy_coefficient = train_params.entropy_coefficient

        if train_params.advantage_estimator.lower() == "gae":
            self.gae = True
        elif train_params.advantage_estimator.lower() == "n-steps":
            self.gae = False
        else:
            raise Exception("Advantage estimator not available")

        # The policy updated at each learning epoch
        self.policy = PsPPO(state_size,
                            action_size,
                            torch.tensor(-1e+8).to(self.device),
                            train_params).to(self.device)

        self.is_recurrent = train_params.shared_recurrent

        self.memory = Memory(n_agents, self.is_recurrent)

        self.hidden_dimension = [1, 1, train_params.hidden_size]
        self.next_hidden = None

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=train_params.learning_rate,
                                          eps=train_params.adam_eps)

        # The policy updated at the end of the training epochs where is used as the old policy.
        # It is used also to obtain trajectories.
        self.policy_old = PsPPO(state_size,
                                action_size,
                                torch.tensor(-1e+8).to(self.device),
                                train_params).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _get_advs(self, rewards, dones, state_estimated_value):
        rewards = torch.tensor(rewards).to(self.device)
        # to multiply with not_dones to handle episode boundary (last state has no V(s'))
        not_dones = 1 - torch.tensor(dones, dtype=torch.int).to(self.device)

        if self.gae:
            assert len(rewards) + 1 == len(state_estimated_value)

            gaes = torch.zeros_like(rewards)
            future_gae = torch.tensor(0.0, dtype=rewards.dtype).to(self.device)
            returns = []
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.discount_factor * state_estimated_value[t + 1] * not_dones[t] - \
                        state_estimated_value[t]
                gaes[t] = future_gae = delta + self.discount_factor * self.lmbda * not_dones[t] * future_gae
                returns.insert(0, gaes[t] + state_estimated_value[t])
            return torch.tensor(returns).to(self.device)
        else:
            returns = torch.zeros_like(rewards)
            future_ret = state_estimated_value[-1]

            for t in reversed(range(len(rewards))):
                returns[t] = future_ret = rewards[t] + self.discount_factor * future_ret * not_dones[t]

            return returns - state_estimated_value[:-1]

    def _learn(self, memory, a):
        # Save functions as objects outside to optimize code
        epochs = self.epochs
        batch_size = self.batch_size

        policy_evaluate = self._evaluate
        get_advantages = self._get_advs
        torch_clamp = torch.clamp
        torch_min = torch.min
        obj_eps = self.eps_clip
        torch_exp = torch.exp
        ec = self.entropy_coefficient
        vlc = self.value_loss_coefficient
        optimizer = self.optimizer
        is_recurrent = self.is_recurrent

        _ = memory.rewards[a].pop()
        _ = memory.dones[a].pop()

        last_state = memory.states[a].pop()
        last_action = memory.actions[a].pop()
        last_mask = memory.masks[a].pop()
        _ = memory.logs_of_action_prob[a].pop()

        # Convert lists to tensors
        old_states = torch.stack(memory.states[a]).to(self.device)
        if is_recurrent:
            old_hidden = memory.hidden_states[a]
        old_actions = torch.tensor(memory.actions[a]).to(self.device)
        old_masks = torch.stack(memory.masks[a]).to(self.device)
        old_logs_of_action_prob = torch.tensor(memory.logs_of_action_prob[a]).to(self.device)
        old_rewards = torch.tensor(memory.rewards[a]).to(self.device)

        with torch.no_grad():
            _, state_estimated_value, _ = policy_evaluate(torch.cat((old_states, torch.unsqueeze(last_state, 0))),
                                                          old_hidden[0] if is_recurrent else None,
                                                          torch.cat((old_actions, torch.unsqueeze(last_action, 0))),
                                                          torch.cat((old_masks, torch.unsqueeze(last_mask, 0))))

        returns = get_advantages(
            memory.rewards[a],
            memory.dones[a],
            state_estimated_value)

        # Find the "Surrogate Loss"
        advantage = returns - state_estimated_value[:-1].squeeze()

        # Optimize policy
        for _ in range(epochs):

            if self.batch_shuffle:
                perm = torch.randperm(len(old_states))
                old_states = old_states[perm]
                old_actions = old_actions[perm]
                old_masks = old_masks[perm]
                old_logs_of_action_prob = old_logs_of_action_prob[perm]
                old_rewards = old_rewards[perm]
                advantage = advantage[perm]

            for batch_start in range(0, len(old_states), batch_size):
                batch_end = batch_start + batch_size
                if batch_end >= len(old_states):
                    # Evaluating old actions and values
                    log_of_action_prob, state_estimated_value, dist_entropy = \
                        policy_evaluate(
                            torch.cat((old_states[batch_start:batch_end], torch.unsqueeze(last_state, 0))),
                            old_hidden[batch_start] if is_recurrent else None,
                            torch.cat((old_actions[batch_start:batch_end], torch.unsqueeze(last_action, 0))),
                            torch.cat((old_masks[batch_start:batch_end], torch.unsqueeze(last_mask, 0))))
                else:
                    # Evaluating old actions and values
                    log_of_action_prob, state_estimated_value, dist_entropy = \
                        policy_evaluate(old_states[batch_start:batch_end + 1],
                                        old_hidden[batch_start] if is_recurrent else None,
                                        old_actions[batch_start:batch_end + 1],
                                        old_masks[batch_start:batch_end + 1])

                batch_advantage = advantage[batch_start:batch_end]

                # Advantage normalization
                batch_advantage = (batch_advantage - torch.mean(batch_advantage)) / (torch.std(batch_advantage) + 1e-10)

                # Find the ratio (pi_theta / pi_theta__old)
                probs_ratio = torch_exp(log_of_action_prob - old_logs_of_action_prob[batch_start:batch_end])

                # Surrogate losses
                unclipped_objective = probs_ratio * batch_advantage
                clipped_objective = torch_clamp(probs_ratio, 1 - obj_eps, 1 + obj_eps) * batch_advantage

                # Policy loss
                policy_loss = torch_min(unclipped_objective, clipped_objective).mean()

                # Value loss
                value_loss = 0.5 * (old_rewards[batch_start:batch_end] -
                                    state_estimated_value[:-1].squeeze()).pow(2).mean()

                loss = -policy_loss + vlc * value_loss - ec * dist_entropy.mean()

                # Gradient descent
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                # Gradient clipping
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                optimizer.step()

                # Update training stats
                with torch.no_grad():
                    self.update_stat("state_estimated_value", state_estimated_value.mean())
                    self.update_stat("probs_ratio", probs_ratio.mean())
                    self.update_stat("advantage", advantage.mean())
                    self.update_stat("policy_loss", policy_loss)
                    self.update_stat("value_loss", vlc * value_loss)
                    self.update_stat("entropy_loss", ec * dist_entropy.mean())
                    self.update_stat("total_loss", loss)

                # To show graph
                """
                from datetime import datetime
                from torchviz import make_dot
                now = datetime.now()
                make_dot(loss).render("attached" + now.strftime("%H-%M-%S"), format="png")
                exit()
                """

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _evaluate(self, state, hidden, action, action_mask):
        """
        Evaluate the current policy obtaining useful information on the decided action's probability distribution.

        :param state: the observed state
        :param action: the performed action
        :param action_mask: a list of 0 and 1 where 0 indicates that the agent should be not sampled
        :return: the logarithm of action probability evaluated in the actions, the value predicted by the critic, the
        distribution entropy
        """

        action_probs, value = self.policy.evaluate_forward(state, action_mask, hidden=hidden)

        action_distribution = Categorical(action_probs[:-1])

        return action_distribution.log_prob(action[:-1]), value, action_distribution.entropy()

    def act(self, state, action_mask, agent_id=None):
        """
        The method used by the agent as its own policy to obtain the action to perform in the given a state and update
        the memory.

        :param state: the observed state
        :param action_mask: a list of 0 and 1 where 0 indicates that the index's action should be not sampled
        :param agent_id: the agent handle
        :return: the action to perform
        """

        assert agent_id is not None and type(agent_id) is int, "agent_id must be an integer and not None"

        # Transform the state Numpy array to a Torch Tensor
        state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            if self.policy.is_recurrent:
                if len(self.memory.hidden_states[agent_id]) >= 1:
                    prev_hidden = self.next_hidden
                else:
                    prev_hidden = (torch.zeros(self.hidden_dimension, dtype=torch.float),
                                        torch.zeros(self.hidden_dimension, dtype=torch.float))
            else:
                prev_hidden = None
            action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
            action_probs, hidden_state = self.policy_old.act_forward(state, action_mask, hidden=prev_hidden)

        if prev_hidden is not None:
            self.memory.hidden_states[agent_id].append(prev_hidden)
            self.next_hidden = hidden_state

        """
        From the paper: "The stochastic policy πθ can be represented by a categorical distribution when the actions of
        the agent are discrete and by a Gaussian distribution when the actions are continuous."
        """
        action_distribution = Categorical(action_probs)

        action = action_distribution.sample()

        # Memory is updated
        self.memory.states[agent_id].append(state)
        self.memory.actions[agent_id].append(action)
        self.memory.logs_of_action_prob[agent_id].append(action_distribution.log_prob(action))
        self.memory.masks[agent_id].append(action_mask)

        return action.item()

    def step(self, agent, agent_reward, agent_done):

        self.memory.rewards[agent].append(agent_reward)
        self.memory.dones[agent].append(agent_done)

        # Update if agent's horizon has been reached
        if self.memory.length(agent) % (self.horizon + 1) == 0:
            self._learn(self.memory, agent)

            """
            Leave last memory unit because the batch includes an additional step which has not been considered 
            in the current trajectory (it has been inserted to compute the advantage) but must be considered in 
            the next trajectory or will be lost.
            """
            self.memory.clear_memory_except_last(agent)

    def save(self, filename):
        self.policy.save(filename)

    def load(self, filename):
        self.policy.load(filename)
