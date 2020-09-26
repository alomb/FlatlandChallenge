from abc import ABC, abstractmethod


class Policy(ABC):
    """
    Policy common interface
    """

    def __init__(self):
        # A dictionary of stats containing lists with value and counter.
        self._training_stats_dict = dict()

    @abstractmethod
    def step(self, **args):
        """
        Update policy.
        :param args:
        :return:
        """
        pass

    @abstractmethod
    def act(self, state, action_mask, **args):
        """
        Provides a policy to the agent. Maps state to actions, considering the presence of an action mask.
        :param state:
        :param action_mask:
        :param args:
        :return:
        """
        pass

    @abstractmethod
    def save(self, filename):
        """
        Save network parameters in filename.

        :param filename:
        :return:
        """
        pass

    @abstractmethod
    def load(self, filename):
        """
        Load network parameters from filenam.

        :param filename:
        :return:
        """
        pass

    def reset_stats(self):
        """
        Reset stats and counters to zero
        """
        for s in self._training_stats_dict:
            self._training_stats_dict[s] = [0.0, 0]

    def get_stat(self, key):
        """

        :param key: the stat name
        :return: the stat average value
        """
        assert key in self._training_stats_dict, "Training stat not declared."
        return self._training_stats_dict[key][0] / self._training_stats_dict[key][1]

    def update_stat(self, key, value):
        """
        Update the stat assigning an initial value when there is no value yet and update counter.

        :param key: the stat name
        :param value: the value to sum to the current value
        """

        current_value = self._training_stats_dict.setdefault(key, [0.0, 0])[0]
        # Update value
        self._training_stats_dict[key][0] = current_value + value
        # Update counter
        self._training_stats_dict[key][1] = self._training_stats_dict[key][1] + 1

    def are_stats_ready(self):
        """

        :return: True when all the stats have been at least modified once and the statistics are present.
        """
        return self._training_stats_dict != {} and all(map(lambda x: x[1] != 0, self._training_stats_dict.values()))
