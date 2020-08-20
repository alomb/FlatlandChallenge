import torch
import torch.nn as nn

from src.psppo.policy import PsPPOPolicy


class PsPPO:
    """
    The class responsible of some logics of the algorithm especially of the loss computation and updating of the policy.
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

        self.device = device
        self.shared = train_params.shared
        self.learning_rate = train_params.learning_rate
        self.discount_factor = train_params.discount_factor
        self.epochs = train_params.epochs
        self.batch_size = train_params.batch_size
        self.eps_clip = train_params.eps_clip
        self.max_grad_norm = train_params.max_grad_norm
        self.lmbda = train_params.lmbda
        self.value_loss_coefficient = train_params.value_loss_coefficient
        self.entropy_coefficient = train_params.entropy_coefficient
        self.loss = 0

        if train_params.advantage_estimator == "gae":
            self.gae = True
        elif train_params.advantage_estimator == "n-steps":
            self.gae = False
        else:
            raise Exception("Advantage estimator not available")

        # The policy updated at each learning epoch
        self.policy = PsPPOPolicy(state_size,
                                  action_size,
                                  device,
                                  train_params).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=train_params.learning_rate,
                                          eps=train_params.adam_eps)

        # The policy updated at the end of the training epochs where is used as the old policy.
        # It is used also to obtain trajectories.
        self.policy_old = PsPPOPolicy(state_size,
                                      action_size,
                                      device,
                                      train_params).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _get_advs(self, rewards, dones, state_estimated_value):
        rewards = torch.tensor(rewards).to(self.device)
        # to multiply with not_dones to handle episode boundary (last state has no V(s'))
        not_dones = 1 - torch.tensor(dones, dtype=torch.int).to(self.device)

        if self.gae:
            assert len(rewards) + 1 == len(state_estimated_value)

            gaes = torch.zeros_like(rewards)
            future_gae = torch.tensor(0.0, dtype=rewards.dtype).to(self.device)

            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.discount_factor * state_estimated_value[t + 1] * not_dones[t] - \
                        state_estimated_value[t]
                gaes[t] = future_gae = delta + self.discount_factor * self.lmbda * not_dones[t] * future_gae
            return gaes
        else:
            returns = torch.zeros_like(rewards)
            future_ret = state_estimated_value[-1]

            for t in reversed(range(len(rewards))):
                returns[t] = future_ret = rewards[t] + self.discount_factor * future_ret * not_dones[t]

            return returns - state_estimated_value[:-1]

    def update(self, memory, a):
        # Save functions as objects outside to optimize code
        epochs = self.epochs
        batch_size = self.batch_size

        policy_evaluate = self.policy.evaluate
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
        old_states = torch.stack(memory.states[a]).to(self.device).detach()
        old_actions = torch.stack(memory.actions[a]).to(self.device)
        old_masks = torch.stack(memory.masks[a]).to(self.device)
        old_logs_of_action_prob = torch.stack(memory.logs_of_action_prob[a]).to(self.device).detach()

        # Optimize policy
        for _ in range(epochs):
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
                    log_of_action_prob - old_logs_of_action_prob[batch_start:batch_end].detach())

                # Find the "Surrogate Loss"
                advantage = get_advantages(
                    memory.rewards[a][batch_start:batch_end],
                    memory.dones[a][batch_start:batch_end],
                    state_estimated_value.detach())

                # Advantage normalization
                advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-10)

                # Surrogate losses
                unclipped_objective = probs_ratio * advantage
                clipped_objective = torch_clamp(probs_ratio, 1 - obj_eps, 1 + obj_eps) * advantage

                # Policy loss
                policy_loss = torch_min(unclipped_objective, clipped_objective).mean()

                # Value loss
                value_loss = 0.5 * (state_estimated_value[:-1].squeeze() -
                                    torch.tensor(memory.rewards[a][batch_start:batch_end],
                                                 dtype=torch.float32).to(self.device)).pow(2).mean()

                self.loss = -policy_loss + vlc * value_loss - ec * dist_entropy.mean()
                # Gradient descent
                optimizer.zero_grad()
                self.loss.backward(retain_graph=True)

                # Gradient clipping
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                optimizer.step()

                # To show graph
                """
                from datetime import datetime
                from torchviz import make_dot
                now = datetime.now()
                make_dot(self.loss).render("attached" + now.strftime("%H-%M-%S"), format="png")
                exit()
                """

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
