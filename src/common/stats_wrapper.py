import gym
from flatland.envs.agent_utils import RailAgentStatus
import numpy as np


class StatsWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 env_params):

        super().__init__(env)
        self.accumulated_normalized_score = []
        self.accumulated_completion = []
        self.accumulated_deadlocks = []
        # Evaluation statics
        self.accumulated_eval_normalized_score = []
        self.accumulated_eval_completion = []
        self.accumulated_eval_deads = []
        self.num_agents = env_params.n_agents
        self.episode = 0
        self.action_count = [0] * self.unwrapped.rail_env.action_space[0]
        self.max_steps = int(4 * 2 * (env_params.y_dim + env_params.x_dim + (env_params.n_agents / env_params.n_cities)))

    def reset(self):
        obs, info = self.env.reset()

        # Score of the episode as a sum of scores of each step for statistics
        self.score = 0
        self.timestep = 0

        return obs, info

    def step(self, action_dict):
        # Collection information about training
        for a in list(action_dict.values()):
            self.action_count[a] += 1

        obs, rewards, done, info = self.env.step(action_dict)

        self.timestep += 1
        self.score += np.sum(rewards[agent]["standard_rewards"] for agent in range(self.num_agents))

        if done["__all__"] or self.timestep >= self.max_steps:
            self._update_and_print_results(info)

        return obs, rewards, done, info

    def _update_and_print_results(self, info):

        self.normalized_score = self.score / (self.max_steps * self.num_agents)
        self.tasks_finished = sum(info["status"][a] in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]
                                  for a in range(self.num_agents))
        self.completion_percentage = self.tasks_finished / max(1, self.num_agents)
        self.deadlocks_percentage = sum(
            [info["deadlocks"][agent] for agent in range(self.num_agents)]) / self.num_agents
        self.action_probs = self.action_count / np.sum(self.action_count)

        # Mean values for terminal display and for more stable hyper-parameter tuning
        self.accumulated_normalized_score.append(self.normalized_score)
        self.accumulated_completion.append(self.completion_percentage)
        self.accumulated_deadlocks.append(self.deadlocks_percentage)

        self.episode += 1
        print(
            "\rEpisode {}"
            "\tScore: {:.3f}"
            " Avg: {:.3f}"
            "\tDone: {:.2f}%"
            " Avg: {:.2f}%"
            "\tDeads: {:.2f}%"
            " Avg: {:.2f}%"
            "\tAction Probs: {}".format(
                self.episode,
                self.normalized_score,
                np.mean(self.accumulated_normalized_score),
                100 * self.completion_percentage,
                100 * np.mean(self.accumulated_completion),
                100 * self.deadlocks_percentage,
                100 * np.mean(self.accumulated_deadlocks),
                self._format_action_prob()
            ), end=" ")

    def _format_action_prob(self):
        self.action_probs = np.round(self.action_probs, 3)
        actions = ["↻", "←", "↑", "→", "◼"]

        buffer = ""
        for action, action_prob in zip(actions, self.action_probs):
            buffer += action + " " + "{:.3f}".format(action_prob) + " "

        return buffer
