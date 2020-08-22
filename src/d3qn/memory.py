import random
from abc import ABC, abstractmethod
from collections import deque, namedtuple, Iterable

import torch
import numpy as np


class ExperienceReplay(ABC):

    def __init__(self):
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    @abstractmethod
    def add(self, state, action, reward, next_state, done, priority=None):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def v_stack_impr(self, data):
        """

        :param data: The list to transform.
        :return: The data in Numpy array, reshaped as (number of samples, size of single sample) considering possible
        scalar data of length 1.
        """

        sub_dim = len(data[0][0]) if isinstance(data[0], Iterable) else 1
        np_data = np.reshape(np.array(data), (len(data), sub_dim))
        return np_data


class UniformExperienceReplay(ExperienceReplay):
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        """
        super(UniformExperienceReplay, self).__init__()
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done, priority=None):
        """
        Add a new experience to memory.
        """

        e = self.experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(e)

    def sample(self):
        """
        :return: a random batch sample of experiences from memory.
        """

        batch = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(self.v_stack_impr([e.state for e in batch if e is not None])) \
            .float().to(self.device)
        actions = torch.from_numpy(self.v_stack_impr([e.action for e in batch if e is not None])) \
            .long().to(self.device)
        rewards = torch.from_numpy(self.v_stack_impr([e.reward for e in batch if e is not None])) \
            .float().to(self.device)
        next_states = torch.from_numpy(self.v_stack_impr([e.next_state for e in batch if e is not None])) \
            .float().to(self.device)
        dones = torch.from_numpy(self.v_stack_impr([e.done for e in batch if e is not None]).astype(np.uint8)) \
            .float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        :return: the current size of internal memory.
        """

        return len(self.memory)


class SumTree:
    """
    A binary tree data structure where the parent’s value is the sum of its children.
    It is used as an efficient data structure to perform weighted sampling by the Prioritized Experience Replay.
    The retrieval process is much more efficient than iterating through a cumulative sum array until the correct
    interval is found. It is also quite easy to update the weight values of the leaf nodes and propagate the changes.
    All that needs to be done is to take the difference of the change and then add that difference to all upstream node
    parents.

    for example the tree

                    11
            6               5
        2       4       3       2

    is

    [11, 6, 5, 2, 4, 3, 2]

    """

    def __init__(self, capacity):
        """

        :param capacity: Tree capacity, number of leaves and number of stored experiences
        """
        self.capacity = capacity
        # write indicates where the tree is updated
        self.write = 0
        # 2 * capacity - 1 is the number of nodes, -1 because in the root layer there is only one node
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        """
        Recursively update the parent nodes
        :param idx: starting index
        :param change: value to sum
        """
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        Recursively search for a sample on a leaf node
        :param idx: starting index
        :param s: random number sampled from an uniform distribution (0, root node) and used as a searching index.
        :return:
        """
        left = 2 * idx + 1
        right = left + 1

        # Leaves have been reached
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """

        :return: the sum of probabilities stored in the root node
        """
        return self.tree[0]

    def add(self, priority, data):
        """
        Store priority and sample.

        :param priority:
        :param data:
        """
        # Index of the leaf in the tree structure
        idx = self.write + self.capacity - 1

        # Add data update priorities
        self.data[self.write] = data
        self.update(idx, priority)

        # Update data related information
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        """
        Update priority

        :param idx: index to update
        :param priority: new priority
        """
        change = priority - self.tree[idx]

        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """
        Get priority and sample.

        :param s:
        :return: index, priority and data
        """
        idx = self._retrieve(0, s)
        # Index of the data from the index of the leaf in the tree
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]


class PrioritisedExperienceReplay(ExperienceReplay):
    """
    A version of experience replay which more frequently calls on those experiences of the agent where there is more
    learning value.
    """
    def __init__(self, capacity, batch_size, device, per_eps=0.01, per_alpha=0.6, per_beta=0.4,
                 per_beta_increment=0.001):
        """

        :param capacity: memory capacity
        :param per_eps: small constant added to the TD error to ensure that even samples with a low TD error still have
        a small chance of being selected for sampling
        :param per_alpha: a factor used to scale the prioritisation based on the TD error up or down. If per_alpha = 0
        then all of the terms go to 1 and every experience has the same chance of being selected, regardless of the TD
        error. Alternatively, if per_alpha = 0 then “full prioritisation” occurs i.e. every sample is randomly selected
        proportional to its TD error (plus the constant). A commonly used per_alpha value is 0.6 – so that
        prioritisation occurs but it is not absolute prioritisation. This promotes some exploration in addition to the
        PER process
        :param per_beta: the factor that increases the importance sampling weights. A value closer to 1 decreases the
        weights for high priority/probability samples and therefore corrects the bias more
        :param per_beta_increment: the value added to per_beta at each sampling until it is annealed to 1
        """
        super(PrioritisedExperienceReplay, self).__init__()
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.per_eps = per_eps
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment

    def _get_priority(self, td_error):
        """

        :param td_error:
        :return: the priority associated to the passed TD error
        """
        return (np.abs(td_error) + self.per_eps) ** self.per_alpha

    def add(self, state, action, reward, next_state, done, td_error=None):
        """
        Update the memory with new experience, the td_error here must be inserted to compute priorities.
        """
        if td_error is None:
            raise Exception("TD Error must be specified when adding experience in Prioritised Experience Replay!")

        self.tree.add(self._get_priority(td_error), self.experience(np.expand_dims(state, 0), action, reward,
                                                                    np.expand_dims(next_state, 0), done))

    def sample(self):
        """

        :return: batch, indexes of the data and importance sampling weights
        """
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.per_beta = np.min([1., self.per_beta + self.per_beta_increment])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            data = 0
            p = None
            idx = None

            while data == 0:
                s = random.uniform(a, b)
                idx, p, data = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        # Compute importance sampling weights
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.per_beta)
        # Rescaling weights from 0 to 1
        is_weight /= is_weight.max()

        states = torch.from_numpy(self.v_stack_impr([e.state for e in batch if e is not None])) \
            .float().to(self.device)
        actions = torch.from_numpy(self.v_stack_impr([e.action for e in batch if e is not None])) \
            .long().to(self.device)
        rewards = torch.from_numpy(self.v_stack_impr([e.reward for e in batch if e is not None])) \
            .float().to(self.device)
        next_states = torch.from_numpy(self.v_stack_impr([e.next_state for e in batch if e is not None])) \
            .float().to(self.device)
        dones = torch.from_numpy(self.v_stack_impr([e.done for e in batch if e is not None]).astype(np.uint8)) \
            .float().to(self.device)

        return (states, actions, rewards, next_states, dones), idxs, is_weight

    """
    def step(self):
        self.beta = np.min([1. - self.e, self.beta + self.beta_increment_per_sampling])
    """

    def update(self, idx, td_error):
        """
        Update priorities of the given indexes with the given error

        :param idx:
        :param td_error:
        """
        self.tree.update(idx, self._get_priority(td_error))

    def __len__(self):
        """
        :return: the current size of internal memory.
        """

        return self.tree.n_entries
