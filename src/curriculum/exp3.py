#@title Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("poster", font_scale=1.2)


# @title Code for EXP3
class TeacherExp3(object):
    """
    Teacher with Exponential-weight algorithm for Exploration and Exploitation.
    """

    def __init__(self, tasks, gamma=0.3):
        self._tasks = tasks
        self._n_tasks = len(self._tasks)
        self._gamma = gamma
        self._log_weights = np.zeros(self._n_tasks)

    @property
    def task_probabilities(self):
        weights = np.exp(self._log_weights - np.sum(self._log_weights))
        probs = ((1 - self._gamma) * weights / np.sum(weights) +
                 self._gamma / self._n_tasks)
        return probs

    def get_task(self):
        """Samples a task, according to current Exp3 belief.
        """
        task_i = np.random.choice(self._n_tasks, p=self.task_probabilities)
        return self._tasks[task_i]

    def update(self, task, reward):
        """ Updates the weight of task given current reward observed
        """
        task_i = self._tasks.index(task)
        reward_corrected = reward / self.task_probabilities[task_i]
        self._log_weights[task_i] += self._gamma * reward_corrected / self._n_tasks


# @title Example of EXP3 learning on 3 tasks {run: "auto"}

# Setup
p1 = 0.28  # @param {type: "slider", min:0, max:1, step:0.01}
p2 = 0.4  # @param {type: "slider", min:0, max:1, step:0.01}
p3 = 0.22  # @param {type: "slider", min:0, max:1, step:0.01}
gamma = 0.1  # @param {type: "slider", min:0, max:1, step:0.01}
T = 200  # @param {type:"integer"}

p_sum = p1 + p2 + p3
p1 /= p_sum
p2 /= p_sum
p3 /= p_sum
tasks_rewards = collections.OrderedDict((
    ('one', p1), ('two', p2), ('three', p3)))
tasks = tasks_rewards.keys()

# Train
teacher = TeacherExp3(tasks, gamma=gamma)
weights_all = np.empty((T, len(tasks)))
arm_probs_all = np.empty((T, len(tasks)))

for t in range(T):
    # pull an arm
    task = teacher.get_task()

    # Get reward
    reward = np.random.rand() < tasks_rewards[task]

    # update belief
    teacher.update(task, reward)

    weights_all[t] = teacher._log_weights
    arm_probs_all[t] = teacher.task_probabilities

# Plot
sns.set_style("darkgrid", {'figure.facecolor': 'none'})
with sns.color_palette("bright"):
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(arm_probs_all, linewidth=2)
    ax.legend(tasks, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Probability of choosing arm')