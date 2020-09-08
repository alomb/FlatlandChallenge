class Memory:
    """
    The class responsible of managing the collected experience.
    Experience is divided by type and each type is subdivided by the relative agent.
    """
    def __init__(self, num_agents, is_recurrent):
        """
        Initialize experience.

        :param num_agents: Number of agents
        """
        self.num_agents = num_agents
        self.actions = [[] for _ in range(num_agents)]
        self.states = [[] for _ in range(num_agents)]
        if is_recurrent:
            self.hidden_states = [[] for _ in range(num_agents)]
        else:
            self.hidden_states = []
        self.logs_of_action_prob = [[] for _ in range(num_agents)]
        self.masks = [[] for _ in range(num_agents)]
        self.rewards = [[] for _ in range(num_agents)]
        self.dones = [[] for _ in range(num_agents)]
        self.is_recurrent = is_recurrent

    def clear_memory(self):
        """
        Reset experience in its initial state
        :return:
        """
        self.__init__(self.num_agents, self.is_recurrent)

    def clear_memory_except_last(self, agent):
        """
        Remove the experience of a specific agent preserving only the last step.
        :param agent: The agent
        :return:
        """
        self.actions[agent] = self.actions[agent][-1:]
        self.states[agent] = self.states[agent][-1:]
        if not self.is_recurrent:
            assert self.hidden_states == [], "Hidden states should be empty!"
        else:
            self.hidden_states[agent] = self.hidden_states[agent][-1:]
        self.logs_of_action_prob[agent] = self.logs_of_action_prob[agent][-1:]
        self.masks[agent] = self.masks[agent][-1:]
        self.rewards[agent] = self.rewards[agent][-1:]
        self.dones[agent] = self.dones[agent][-1:]

    def length(self, agent):
        assert_len = len(self.actions[agent]) == len(self.states[agent]) == \
                     len(self.logs_of_action_prob[agent]) == len(self.masks[agent]) == len(self.rewards[agent]) == \
                     len(self.dones[agent])
        assert_message = "The lengths of the different parts of an agent's experience are not equal." \
                         " This should not happen!"
        if self.is_recurrent:
            assert assert_len and len(self.hidden_states) == len(self.actions), assert_message
        else:
            assert assert_len, assert_message

        return len(self.rewards[agent])
