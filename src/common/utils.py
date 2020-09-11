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

    def __init__(self, tensorboard_path, env_params=None, train_params=None):
        """

        :param tensorboard_path: the path where logs are saved
        :param env_params: environment parameters
        :param train_params: training parameters
        """
        self.writer = SummaryWriter(tensorboard_path)

        if train_params is not None:
            self.writer.add_hparams(train_params, {})
        if env_params is not None:
            self.writer.add_hparams(env_params, {})
        self.step = 0

    def update_tensorboard(self, env, train_params, timers):
        """
        Save logs to Tensorboard

        :param env: the environment used to extract some statistics
        :param train_params: a dictionary of training statistics to record
        :param timers: a dictionary of timers to record
        """
        # Environment parameters
        self.writer.add_scalar("metrics/score", env.normalized_score, self.step)
        self.writer.add_scalar("metrics/accumulated_score", np.mean(env.accumulated_normalized_score), self.step)
        self.writer.add_scalar("metrics/completion", env.completion_percentage, self.step)
        self.writer.add_scalar("metrics/accumulated_completion", np.mean(env.accumulated_completion), self.step)
        self.writer.add_scalar("metrics/deadlocks", env.deadlocks_percentage, self.step)
        self.writer.add_scalar("metrics/accumulated_deadlocks", np.mean(env.accumulated_deadlocks), self.step)
        self.writer.add_histogram("actions/distribution", np.array(env.action_probs), self.step)
        self.writer.add_scalar("actions/nothing", env.action_probs[RailEnvActions.DO_NOTHING], self.step)
        self.writer.add_scalar("actions/left", env.action_probs[RailEnvActions.MOVE_LEFT], self.step)
        self.writer.add_scalar("actions/forward", env.action_probs[RailEnvActions.MOVE_FORWARD], self.step)
        self.writer.add_scalar("actions/right", env.action_probs[RailEnvActions.MOVE_RIGHT], self.step)
        self.writer.add_scalar("actions/stop", env.action_probs[RailEnvActions.STOP_MOVING], self.step)

        # Training parameters
        for param_name, param in train_params.items():
            assert type(param_name) is str, "Parameters names must be strings!"
            self.writer.add_scalar("training/" + param_name, param, self.step)

        # Timers
        for timer_name, timer in timers.items():
            assert type(timer_name) is str and type(timer) is Timer, "A Timer object and its name (string) must be" \
                                                                     "passed!"
            self.writer.add_scalar("timers/" + timer_name, timer.get(), self.step)

        self.step += 1
