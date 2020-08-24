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

        self.state_estimated_value_metric = 0
        self.probs_ratio_metric = 0
        self.advantage_metric = 0
        self.policy_loss_metric = 0
        self.value_loss_metric = 0
        self.entropy_loss_metric = 0
        self.loss_metric = 0

        self.memory = Memory(n_agents)

        if train_params.advantage_estimator.lower() == "gae":
            self.gae = True
        elif train_params.advantage_estimator.lower() == "n-steps":
            self.gae = False
        else:
            raise Exception("Advantage estimator not available")

        # The policy updated at each learning epoch
        self.policy = PsPPO(state_size,
                            action_size,
                            train_params).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=train_params.learning_rate,
                                          eps=train_params.adam_eps)

        # The policy updated at the end of the training epochs where is used as the old policy.
        # It is used also to obtain trajectories.
        self.policy_old = PsPPO(state_size,
                                action_size,
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

        _ = memory.rewards[a].pop()
        _ = memory.dones[a].pop()

        last_state = memory.states[a].pop()
        last_action = memory.actions[a].pop()
        last_mask = memory.masks[a].pop()
        _ = memory.logs_of_action_prob[a].pop()

        # Convert lists to tensors
        old_states = torch.stack(memory.states[a]).to(self.device)
        old_actions = torch.tensor(memory.actions[a]).to(self.device)
        old_masks = torch.stack(memory.masks[a]).to(self.device)
        old_logs_of_action_prob = torch.tensor(memory.logs_of_action_prob[a]).to(self.device)
        old_rewards = torch.tensor(memory.rewards[a]).to(self.device)

        with torch.no_grad():
            _, state_estimated_value, _ = policy_evaluate(torch.cat((old_states, torch.unsqueeze(last_state, 0))),
                                                          torch.cat((old_actions, torch.unsqueeze(last_action, 0))),
                                                          torch.cat((old_masks, torch.unsqueeze(last_mask, 0))))

        returns = get_advantages(
            memory.rewards[a],
            memory.dones[a],
            state_estimated_value)

        # Find the "Surrogate Loss"
        advantage = returns - state_estimated_value[:-1].squeeze()

        # Advantage normalization
        advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-10)

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
                            torch.cat((old_actions[batch_start:batch_end], torch.unsqueeze(last_action, 0))),
                            torch.cat((old_masks[batch_start:batch_end], torch.unsqueeze(last_mask, 0))))
                else:
                    # Evaluating old actions and values
                    log_of_action_prob, state_estimated_value, dist_entropy = \
                        policy_evaluate(old_states[batch_start:batch_end + 1],
                                        old_actions[batch_start:batch_end + 1],
                                        old_masks[batch_start:batch_end + 1])

                # Find the ratio (pi_theta / pi_theta__old)
                probs_ratio = torch_exp(
                    log_of_action_prob - old_logs_of_action_prob[batch_start:batch_end])

                # Surrogate losses
                unclipped_objective = probs_ratio * advantage[batch_start:batch_end]
                clipped_objective = torch_clamp(probs_ratio, 1 - obj_eps, 1 + obj_eps) * \
                                    advantage[batch_start:batch_end]

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

                with torch.no_grad():
                    self.state_estimated_value_metric = state_estimated_value.mean()
                    self.probs_ratio_metric = probs_ratio.mean()
                    self.advantage_metric = advantage.mean()
                    self.policy_loss_metric = policy_loss
                    self.value_loss_metric = vlc * value_loss
                    self.entropy_loss_metric = ec * dist_entropy.mean()
                    self.loss_metric = loss

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

    def _evaluate(self, state, action, action_mask):
        """
        Evaluate the current policy obtaining useful information on the decided action's probability distribution.

        :param state: the observed state
        :param action: the performed action
        :param action_mask: a list of 0 and 1 where 0 indicates that the agent should be not sampled
        :return: the logarithm of action probability, the value predicted by the critic, the distribution entropy
        """

        action_logits = self.policy.actor_network(state[:-1])

        # Action masking, default values are True, False are present only if masking is enabled.
        # If No op is not allowed it is masked even if masking is not active
        action_logits = torch.where(action_mask[:-1], action_logits, torch.tensor(-1e+8).to(self.device))

        action_probs = self.policy.softmax(action_logits)

        action_distribution = Categorical(action_probs)

        return action_distribution.log_prob(action[:-1]), self.policy.critic_network(state), \
               action_distribution.entropy()

    def act(self, state, action_mask=None, action=None):
        """
        The method used by the agent as its own policy to obtain the action to perform in the given a state and update
        the memory.

        :param state: the observed state
        :param action_mask: a list of 0 and 1 where 0 indicates that the agent should be not sampled
        :param action: an action to perform decided by some external logic
        :return: the action to perform
        """

        # The agent name is appended at the state
        agent_id = int(state[-1])
        # Transform the state Numpy array to a Torch Tensor
        state = torch.from_numpy(state).float().to(self.device)
        action_logits = self.policy_old.actor_network(state)

        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)

        # Action masking, default values are True, False are present only if masking is enabled.
        # If No op is not allowed it is masked even if masking is not active
        action_logits = torch.where(action_mask, action_logits, torch.tensor(-1e+8).to(self.device))

        action_probs = self.policy_old.softmax(action_logits)

        """
        From the paper: "The stochastic policy πθ can be represented by a categorical distribution when the actions of
        the agent are discrete and by a Gaussian distribution when the actions are continuous."
        """
        action_distribution = Categorical(action_probs)

        if action is None:
            action = action_distribution.sample()

        # Memory is updated
        self.memory.states[agent_id].append(state)
        self.memory.actions[agent_id].append(action)
        self.memory.logs_of_action_prob[agent_id].append(action_distribution.log_prob(action))
        self.memory.masks[agent_id].append(action_mask)

        return action.item()

    def step(self, agent, total_timestep_reward_shaped, done, last_step):

        self.memory.rewards[agent].append(total_timestep_reward_shaped)
        self.memory.dones[agent].append(done[agent])

        # Set dones to True when the episode is finished because the maximum number of steps has been reached
        if last_step:
            self.memory.dones[agent][-1] = True

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
