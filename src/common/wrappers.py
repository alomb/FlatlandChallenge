import gym
import numpy as np

from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions


class StatsWrapper(gym.Wrapper):
    def __init__(self, env, env_params):

        super().__init__(env)
        self.accumulated_normalized_score = []
        self.accumulated_completion = []
        self.accumulated_deadlocks = []
        self.num_agents = env_params.n_agents
        self.episode = 0
        self.action_count = [0] * self.unwrapped.rail_env.action_space[0]
        self.max_steps = int(4 * 2 * (env_params.y_dim + env_params.x_dim +
                                      (env_params.n_agents / env_params.n_cities)))
        self.score = 0
        self.timestep = 0

    def reset(self):
        obs, info = self.env.reset()

        # Score of the episode as a sum of scores of each step for statistics
        self.score = 0
        self.timestep = 0

        return obs, info

    def step(self, action_dict):

        # Update statistics
        for a in range(self.num_agents):
            if a not in action_dict:
                self.action_count[0] += 1
            else:
                self.action_count[action_dict[a]] += 1

        obs, rewards, done, info = self.env.step(action_dict)
        self.timestep += 1

        self.score += sum(rewards[agent] for agent in range(self.num_agents))

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


class RewardsWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 invalid_action_penalty,
                 stop_penalty,
                 deadlock_penalty,
                 shortest_path_penalty_coefficient,
                 done_bonus):

        super().__init__(env)
        self.shortest_path = []
        self.invalid_action_penalty = invalid_action_penalty
        self.stop_penalty = stop_penalty
        self.deadlock_penalty = deadlock_penalty
        self.shortest_path_penalty_coefficient = shortest_path_penalty_coefficient
        self.done_bonus = done_bonus

    def reset(self):
        obs, info = self.env.reset()
        self.shortest_path = [obs.get(a)[6] if obs.get(a) is not None else 0 for a in
                              range(self.unwrapped.rail_env.get_num_agents())]

        return obs, info

    def step(self, action_dict):

        num_agents = self.unwrapped.rail_env.get_num_agents()

        invalid_rewards_shaped = self._check_invalid_transitions(action_dict)
        invalid_stop_rewards_shaped = self._check_stop_transition(action_dict, invalid_rewards_shaped)

        # Environment step
        obs, rewards, done, info = self.env.step(action_dict)

        rewards_shaped2 = rewards.copy()
        for agent in range(num_agents):
            if done[agent]:
                rewards_shaped2[agent] = self.done_bonus
            else:
                # Sum stop and invalid penalties to the standard reward
                rewards_shaped2[agent] += invalid_stop_rewards_shaped[agent]

                # Shortest path penalty
                new_shortest_path = obs.get(agent)[6] if obs.get(agent) is not None else 0
                if self.shortest_path[agent] < new_shortest_path:
                    rewards_shaped2[agent] *= self.shortest_path_penalty_coefficient

                # Deadlocks penalty
                if "deadlocks" in info and info["deadlocks"][agent]:
                    rewards_shaped2[agent] += self.deadlock_penalty

        return obs, rewards, done, info

    def _check_invalid_transitions(self, action_dict):
        rewards = {}
        for agent in range(self.unwrapped.rail_env.get_num_agents()):
            if self.unwrapped.rail_env.agents[agent].status != RailAgentStatus.READY_TO_DEPART and \
                    self.unwrapped.rail_env.agents[agent].status != RailAgentStatus.DONE and \
                    self.unwrapped.rail_env.agents[agent].status != RailAgentStatus.DONE_REMOVED:
                _, cell_valid, _, _, transition_valid = self.unwrapped.rail_env._check_action_on_agent(
                    RailEnvActions(action_dict[agent]),
                    self.unwrapped.rail_env.agents[agent])
                if not all([cell_valid, transition_valid]):
                    rewards[agent] = self.invalid_action_penalty
                else:
                    rewards[agent] = 0.0
            else:
                rewards[agent] = 0.0

        return rewards

    def _check_stop_transition(self, action_dict, rewards):
        return {a: self.stop_penalty if a in action_dict and action_dict[a] == RailEnvActions.STOP_MOVING
                else rewards[a] for a in range(self.unwrapped.rail_env.get_num_agents())}
