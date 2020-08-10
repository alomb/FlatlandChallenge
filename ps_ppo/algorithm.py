from collections import OrderedDict
import os.path

import torch
from torch import nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.actions = [[] for _ in range(num_agents)]
        self.states = [[] for _ in range(num_agents)]
        self.logs_of_action_prob = [[] for _ in range(num_agents)]

        self.rewards = []
        self.dones = []

    def clear_memory(self):
        self.__init__(self.num_agents)

    def clear_memory_except_last(self):
        self.actions = list(map(lambda l: l[-1:], self.actions))
        self.states = list(map(lambda l: l[-1:], self.states))
        self.logs_of_action_prob = list(map(lambda l: l[-1:], self.logs_of_action_prob))

        self.rewards = self.rewards[-1:]
        self.dones = self.dones[-1:]


class ActorCritic(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 shared,
                 critic_mlp_width,
                 critic_mlp_depth,
                 last_critic_layer_scaling,
                 actor_mlp_width,
                 actor_mlp_depth,
                 last_actor_layer_scaling,
                 activation):
        """
        :param state_size: The number of attributes of each state
        :param action_size: The number of available actions
        :param shared: The actor and critic hidden and first layers are shared
        :param critic_mlp_width: The number of nodes in the critic's hidden network
        :param critic_mlp_depth: The number of hidden layers + the input one of the critic's network
        :param last_critic_layer_scaling: The scale applied at initialization on the last critic network's layer
        :param actor_mlp_width: The number of nodes in the actor's hidden network
        :param actor_mlp_depth: The number of hidden layers + the input one of the actor's network
        :param last_actor_layer_scaling: The scale applied at initialization on the last actor network's layer
        :param activation: the activation function (ReLU, Tanh)
        """

        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.activation = activation

        # Network creation
        critic_layers = self._build_network(False, critic_mlp_depth, critic_mlp_width)
        self.critic_network = nn.Sequential(critic_layers)
        if not shared:
            self.actor_network = nn.Sequential(self._build_network(True, actor_mlp_depth, actor_mlp_width))
        else:
            if critic_mlp_depth <= 1:
                raise Exception("Shared networks must have depth greater than 1")
            actor_layers = critic_layers.copy()
            actor_layers.popitem()
            actor_layers["actor_output_layer"] = nn.Linear(critic_mlp_width, action_size)
            actor_layers["actor_softmax"] = nn.Softmax(dim=-1)
            self.actor_network = nn.Sequential(actor_layers)

        # Network initialization
        # https://pytorch.org/docs/stable/nn.init.html#nn-init-doc
        def weights_init(submodule):
            # TODO
            raise NotImplementedError()

        # self.critic_network.apply(weights_init)
        # self.actor_network.apply(weights_init)

        # Last layer's weights rescaling
        with torch.no_grad():
            list(self.critic_network.children())[-1].weight.mul_(last_critic_layer_scaling)
            # -2 because actor contains softmax as last layer
            list(self.actor_network.children())[-2].weight.mul_(last_actor_layer_scaling)

    def _build_network(self, is_actor, nn_depth, nn_width):
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

        # Actor needs softmax
        if is_actor:
            network["%s_softmax" % nn_type] = nn.Softmax(dim=-1)

        return network

    def _get_activation(self):
        if self.activation == "ReLU":
            return nn.ReLU()
        elif self.activation == "Tanh":
            return nn.Tanh()
        else:
            raise Exception("The specified activation function don't exists or is not available")

    def act(self, state, memory, action=None):
        """
        The method used by the agent as its own policy to obtain the action to perform in the given a state and update
        the memory.
        :param state: the observed state
        :param memory: the memory to update
        :param action:
        :return: the action to perform
        """

        # The agent name is appended at the state
        agent_id = int(state[-1])
        # Transform the state Numpy array to a Torch Tensor
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.actor_network(state)
        """
        From the paper: "The stochastic policy πθ can be represented by a categorical distribution when the actions of
        the agent are discrete and by a Gaussian distribution when the actions are continuous."
        """
        action_distribution = Categorical(action_probs)

        if not action:
            action = action_distribution.sample()

        # Memory is updated
        memory.states[agent_id].append(state)
        memory.actions[agent_id].append(action)
        memory.logs_of_action_prob[agent_id].append(action_distribution.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        """
        Evaluate the current policy obtaining useful information on the decided action's probability distribution.
        :param state: the observed state
        :param action: the performed action
        :return: the logarithm of action probability, the value predicted by the critic, the distribution entropy
        """

        action_distribution = Categorical(self.actor_network(state[:-1]))

        return action_distribution.log_prob(action[:-1]), self.critic_network(state), action_distribution.entropy()


class PsPPO:
    def __init__(self,
                 n_agents,
                 state_size,
                 action_size,
                 shared,
                 critic_mlp_width,
                 critic_mlp_depth,
                 last_critic_layer_scaling,
                 actor_mlp_width,
                 actor_mlp_depth,
                 last_actor_layer_scaling,
                 learning_rate,
                 adam_eps,
                 activation,
                 discount_factor,
                 epochs,
                 batch_size,
                 eps_clip,
                 lmbda,
                 advantage_estimator,
                 value_function_loss,
                 entropy_coefficient=None,
                 value_loss_coefficient=None,
                 load_model_path=None,
                 ):
        """
        :param n_agents: The number of agents
        :param state_size: The number of attributes of each state
        :param action_size: The number of available actions
        :param shared: The actor and critic hidden and first layers are shared
        :param critic_mlp_width: The number of nodes in the critic's hidden network
        :param critic_mlp_depth: The number of layers in the critic's network
        :param last_critic_layer_scaling: The scale applied at initialization on the last critic network's layer
        :param actor_mlp_width: The number of nodes in the actor's hidden network
        :param actor_mlp_depth: The number of layers in the actor's network
        :param last_actor_layer_scaling: The scale applied at initialization on the last actor network's layer
        :param learning_rate: The learning rate
        :param adam_eps: Adam optimizer epsilon value
        :param discount_factor: The discount factor
        :param epochs: The number of training epochs for each batch
        :param batch_size: The size of data batches used for each agent in the training loop
        :param eps_clip: The offset used in the minimum and maximum values of the clipping function
        :param lmbda: Controls gae bias–variance trade-off
        :param advantage_estimator: The advantage estimation technique n-steps or gae (Generalized Advantage estimation)
        :param value_function_loss: The function used to compute the value loss mse of huber (L1 loss)
        :param entropy_coefficient: Coefficient multiplied by the entropy and used in the shared setting loss function
        :param value_loss_coefficient: Coefficient multiplied by the value loss and used in the loss function
        :param load_model_path: The path containing the model to load
        """

        self.n_agents = n_agents
        self.shared = shared
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epochs = epochs
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.value_loss_coefficient = value_loss_coefficient
        self.entropy_coefficient = entropy_coefficient

        if advantage_estimator == "gae":
            self.gae = True
        elif advantage_estimator == "n-steps":
            self.gae = False
        else:
            raise Exception("Advantage estimator not available")

        if load_model_path is not None and not os.path.isfile(load_model_path):
            load_model_path = None

        # The policy updated at each learning epoch
        self.policy = ActorCritic(state_size,
                                  action_size,
                                  shared,
                                  critic_mlp_width,
                                  critic_mlp_depth,
                                  last_critic_layer_scaling,
                                  actor_mlp_width,
                                  actor_mlp_depth,
                                  last_actor_layer_scaling,
                                  activation).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, eps=adam_eps)

        # The policy updated at the end of the training epochs where is used as the old policy.
        # It is used also to obtain trajectories.
        self.policy_old = ActorCritic(state_size,
                                      action_size,
                                      shared,
                                      critic_mlp_width,
                                      critic_mlp_depth,
                                      last_critic_layer_scaling,
                                      actor_mlp_width,
                                      actor_mlp_depth,
                                      last_actor_layer_scaling,
                                      activation).to(device)

        if load_model_path is not None:
            self.policy.load_state_dict(torch.load(load_model_path))
            # self.policy.eval()
        self.policy_old.load_state_dict(self.policy.state_dict())
        # self.policy_old.eval()

        if value_function_loss == "mse":
            self.value_loss_function = nn.MSELoss()
        elif value_function_loss == "huber":
            self.value_loss_function = nn.SmoothL1Loss()
        else:
            raise Exception("The provided value loss function is not available!")

    def _get_advs(self, gae, rewards, dones, gamma, state_estimated_value):
        rewards = torch.tensor(rewards).to(device)
        # To handle episode ending
        not_dones = 1 - torch.tensor(dones, dtype=torch.int).to(device)

        if gae:
            assert len(rewards) + 1 == len(state_estimated_value)

            gaes = torch.zeros_like(rewards)
            future_gae = torch.tensor(0.0, dtype=rewards.dtype).to(device)

            for t in reversed(range(len(rewards))):
                delta = rewards[t] + gamma * state_estimated_value[t + 1] * not_dones[t] - state_estimated_value[t]
                gaes[t] = future_gae = delta + gamma * self.lmbda * not_dones[t] * future_gae

            return gaes
        else:
            returns = torch.zeros_like(rewards)
            future_ret = state_estimated_value[-1]

            for t in reversed(range(len(rewards))):
                returns[t] = future_ret = rewards[t] + gamma * future_ret * not_dones[t]

            return returns - state_estimated_value[:-1]

    def update(self, memory):
        """
        :param memory:
        :return:
        """

        # Save functions as objects outside to optimize code
        epochs = self.epochs
        n_agents = self.n_agents
        batch_size = self.batch_size

        discount_factor = self.discount_factor
        policy_evaluate = self.policy.evaluate
        get_advantages = self._get_advs
        gae = self.gae
        torch_clamp = torch.clamp
        torch_min = torch.min
        obj_eps = self.eps_clip
        torch_exp = torch.exp
        shared = self.shared
        ec = self.entropy_coefficient
        vlc = self.value_loss_coefficient
        value_loss_function = self.value_loss_function
        optimizer = self.optimizer

        _ = memory.rewards.pop()
        _ = memory.dones.pop()

        # For each agent train the policy and the value network on personal observations
        for a in range(n_agents):
            last_state = memory.states[a].pop()
            last_action = memory.actions[a].pop()
            _ = memory.logs_of_action_prob[a].pop()

            # Convert lists to tensors
            old_states = torch.stack(memory.states[a]).to(device).detach()
            old_actions = torch.stack(memory.actions[a]).to(device)
            old_logs_of_action_prob = torch.stack(memory.logs_of_action_prob[a]).to(device).detach()

            # Optimize policy
            for _ in range(epochs):
                for batch_start in range(0, len(old_states), batch_size):
                    batch_end = batch_start + batch_size
                    if batch_end >= len(old_states):
                        # Evaluating old actions and values
                        # print("Old_states: ", old_states[batch_start:batch_end].shape)
                        # print("Last_state: ", torch.unsqueeze(last_state, 0).shape)
                        # print("Action: ", old_actions[batch_start:batch_end].shape)
                        # print("Last action", torch.unsqueeze(last_action, 0).shape)
                        log_of_action_prob, state_estimated_value, dist_entropy = \
                            policy_evaluate(
                                torch.cat((old_states[batch_start:batch_end], torch.unsqueeze(last_state, 0))),
                                torch.cat((old_actions[batch_start:batch_end], torch.unsqueeze(last_action, 0))))
                    else:
                        # Evaluating old actions and values
                        # print("Old_states: ",old_states[batch_start:batch_end + 1].shape)
                        # print("Action: ", old_actions[batch_start:batch_end + 1].shape)
                        log_of_action_prob, state_estimated_value, dist_entropy = \
                            policy_evaluate(old_states[batch_start:batch_end + 1],
                                            old_actions[batch_start:batch_end + 1])

                    # print(old_states[batch_start:batch_end])
                    # print(old_actions[batch_start:batch_end])
                    # print(old_logs_of_action_prob[batch_start:batch_end])
                    # print(log_of_action_prob)
                    # print(rewards[batch_start:batch_end])

                    # Find the ratio (pi_theta / pi_theta__old)
                    probs_ratio = torch_exp(
                        log_of_action_prob - old_logs_of_action_prob[batch_start:batch_end].detach())

                    # Find the "Surrogate Loss"
                    advantage = get_advantages(
                        gae,
                        memory.rewards[batch_start:batch_end],
                        memory.dones[batch_start:batch_end],
                        discount_factor,
                        state_estimated_value.detach())

                    # print("estimated value\t " + str(torch.mean(state_estimated_value).item()))
                    # print("reward\t " + str(torch.mean(torch.tensor(memory.rewards[batch_start:batch_end]).to(device))
                    #                         .item()))
                    # print("advantage\t " + str(torch.mean(advantage).item()))
                    # print("probsratio\t " + str(torch.mean(probs_ratio).item()))

                    # Advantage normalization
                    advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-10)

                    unclipped_objective = probs_ratio * advantage
                    clipped_objective = torch_clamp(probs_ratio, 1 - obj_eps, 1 + obj_eps) * advantage

                    loss = -torch_min(unclipped_objective,
                                      clipped_objective) + vlc * value_loss_function(
                        state_estimated_value[:-1].squeeze(),
                        torch.tensor(memory.rewards[batch_start:batch_end], dtype=torch.float32).to(device))

                    if shared:
                        loss -= ec * dist_entropy

                    # Gradient descent
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()

                    # To show graph
                    """
                    from datetime import datetime
                    from torchviz import make_dot
                    now = datetime.now()
                    make_dot(loss.mean()).render("attached" + now.strftime("%H-%M-%S"), format="png")
                    exit()
                    """
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
