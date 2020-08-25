from timeit import default_timer

import numpy as np

from flatland.envs.rail_env import RailEnvActions
from torch.utils.tensorboard import SummaryWriter


class Timer(object):
    """
    Class to record elapsed time
    """

    def __init__(self):
        self.total_time = 0.0
        self.start_time = 0.0
        self.end_time = 0.0

    def start(self):
        """
        Start the timer

        """
        self.start_time = default_timer()

    def end(self):
        """
        Stop the timer

        """
        self.total_time += default_timer() - self.start_time

    def get(self):
        """

        :return: total time
        """
        return self.total_time

    def get_current(self):
        """

        :return elapsed time:
        """
        return default_timer() - self.start_time

    def reset(self):
        """
        Reset timer

        """
        self.__init__()

    def __repr__(self):
        """
        Used to print current elapsed time
        :return: elapsed time
        """
        return self.get()


class TensorBoardLogger:
    """
    Class to handle Tensorboard logging.
    """

    def __init__(self, tensorboard_path, env_params, train_params):
        """

        :param tensorboard_path: the path where logs are saved
        :param env_params: environment parameters
        :param train_params: training parameters
        """
        self.writer = SummaryWriter(tensorboard_path)

        self.writer.add_hparams(train_params, {})
        self.writer.add_hparams(env_params, {})

    def update_tensorboard(self, episode, env, policy_params, timers):
        """
        Save logs to Tensorboard

        :param episode: the current episode
        :param env: the environment used to extract some statistics
        :param policy_params: a dictionary of policy's statistics to record
        :param timers: a dictionary of timers to record
        """
        # Environment parameters
        self.writer.add_scalar("training/score", env.normalized_score, episode)
        self.writer.add_scalar("training/accumulated_score", np.mean(env.accumulated_normalized_score), episode)
        self.writer.add_scalar("training/completion", env.completion_percentage, episode)
        self.writer.add_scalar("training/accumulated_completion", np.mean(env.accumulated_completion), episode)
        self.writer.add_scalar("training/deadlocks", env.deadlocks_percentage, episode)
        self.writer.add_scalar("training/accumulated_deadlocks", np.mean(env.accumulated_deadlocks), episode)
        self.writer.add_histogram("actions/distribution", np.array(env.action_probs), episode)
        self.writer.add_scalar("actions/nothing", env.action_probs[RailEnvActions.DO_NOTHING], episode)
        self.writer.add_scalar("actions/left", env.action_probs[RailEnvActions.MOVE_LEFT], episode)
        self.writer.add_scalar("actions/forward", env.action_probs[RailEnvActions.MOVE_FORWARD], episode)
        self.writer.add_scalar("actions/right", env.action_probs[RailEnvActions.MOVE_RIGHT], episode)
        self.writer.add_scalar("actions/stop", env.action_probs[RailEnvActions.STOP_MOVING], episode)

        # Policy parameters
        for param_name, param in policy_params.items():
            assert type(param_name) is str, "Parameters names must be strings!"
            self.writer.add_scalar("training/" + param_name, param, episode)

        # Timers
        for timer_name, timer in timers.items():
            assert type(timer_name) is str and type(timer) is Timer, "A Timer object and its name (string) must be" \
                                                                     "passed!"
            self.writer.add_scalar("timer/" + timer_name, timer.get(), episode)
