from collections import defaultdict

import gym
import numpy as np
from flatland.core.grid.grid4_utils import get_new_position

from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions


class StatsWrapper(gym.Wrapper):
    """
    Wrapper to store and print training statistics.
    """

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
        """
        Reset the environment and the statistics

        :return: observation and info
        """
        obs, info = self.env.reset()

        # Score of the episode as a sum of scores of each step for statistics
        self.score = 0
        self.timestep = 0

        return obs, info

    def step(self, action_dict):
        """
        Update some statistics and print at the end of the episode

        :param action_dict: dictionary containing for each agent the decided action
        :return: the tuple obs, rewards, done and info
        """
        # Update statistics
        for a in range(self.num_agents):
            if a not in action_dict:
                self.action_count[0] += 1
            else:
                self.action_count[action_dict[a]] += 1

        obs, rewards, done, info = self.env.step(action_dict)
        self.timestep += 1

        # Update score and compute total rewards equal to each agent considering the rewards shaped or normal
        self.score += float(sum(rewards.values())) if "original_rewards" not in info \
            else float(sum(info["original_rewards"].values()))

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
    """
    Wrapper that changes the rewards based on some penalties and bonuses. Used to perform reward shaping.
    """

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
        """
        Reset the environment and the shortest path cache

        :return: observation and info
        """
        obs, info = self.env.reset()
        self.shortest_path = [obs.get(a)[6] if obs.get(a) is not None else 0 for a in
                              range(self.unwrapped.rail_env.get_num_agents())]

        return obs, info

    def step(self, action_dict):
        """
        Computes the shaped rewards

        :param action_dict: dictionary containing for each agent the decided action
        :return: the tuple obs, rewards, done and info, info contains a new key "original_rewards" containing the
        original rewards get from the previous layer
        """
        num_agents = self.unwrapped.rail_env.get_num_agents()

        invalid_rewards_shaped = self._check_invalid_transitions(action_dict)
        stop_rewards_shaped = self._check_stop_transition(action_dict)

        invalid_stop_rewards_shaped = {k: invalid_rewards_shaped.get(k, 0) + stop_rewards_shaped.get(k, 0)
                                       for k in set(invalid_rewards_shaped) & set(stop_rewards_shaped)}

        # Environment step
        obs, rewards, done, info = self.env.step(action_dict)

        info["original_rewards"] = rewards

        rewards_shaped = rewards.copy()
        for agent in range(num_agents):
            if done[agent]:
                rewards_shaped[agent] = self.done_bonus
            else:
                # Sum stop and invalid penalties to the standard reward
                rewards_shaped[agent] += invalid_stop_rewards_shaped[agent]

                # Shortest path penalty
                new_shortest_path = obs.get(agent)[6] if obs.get(agent) is not None else 0
                if self.shortest_path[agent] < new_shortest_path:
                    rewards_shaped[agent] *= self.shortest_path_penalty_coefficient

                # Deadlocks penalty
                if "deadlocks" in info and info["deadlocks"][agent]:
                    rewards_shaped[agent] += self.deadlock_penalty

        return obs, rewards_shaped, done, info

    def _check_invalid_transitions(self, action_dict):
        """

        :param action_dict: dictionary containing for each agent the decided action
        :return: the penalties based on attempted invalid transitions
        """
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

    def _check_stop_transition(self, action_dict):
        """

        :param action_dict: dictionary containing for each agent the decided action
        :return: the penalties based on decided STOPS
        """
        return {a: self.stop_penalty if a in action_dict and action_dict[a] == RailEnvActions.STOP_MOVING
                else 0.0 for a in range(self.unwrapped.rail_env.get_num_agents())}


class ActionSkippingWrapper(gym.Wrapper):

    def __init__(self, env, discounting):
        super().__init__(env)
        self._decision_cells = None
        self._discounting = discounting
        self._skipped_rewards = defaultdict(list)

    def _find_all_decision_cells(self):
        switches = []
        switches_neighbors = []
        directions = list(range(4))
        for h in range(self.unwrapped.rail_env.height):
            for w in range(self.unwrapped.rail_env.width):
                pos = (h, w)
                is_switch = False
                # Check for switch: if there is more than one outgoing transition
                for orientation in directions:
                    possible_transitions = self.unwrapped.rail_env.rail.get_transitions(*pos, orientation)
                    num_transitions = np.count_nonzero(possible_transitions)
                    if num_transitions > 1:
                        switches.append(pos)
                        is_switch = True
                        break
                if is_switch:
                    # Add all neighbouring rails, if pos is a switch
                    for orientation in directions:
                        possible_transitions = self.unwrapped.rail_env.rail.get_transitions(*pos, orientation)
                        for movement in directions:
                            if possible_transitions[movement]:
                                switches_neighbors.append(get_new_position(pos, movement))

        decision_cells = switches + switches_neighbors
        return tuple(map(set, (switches, switches_neighbors, decision_cells)))

    def _on_decision_cell(self, agent):
        """

        :param agent: an EnvAgent
        :return: True when the agent is on a decision cell
        """
        return agent.position is None or agent.position in self._decision_cells
        # or agent.position == agent.initial_position

    def step(self, action_dict):
        """

        :param action_dict: dictionary containing for each agent the decided action
        :return: the tuple obs, rewards, done and info
        """
        final_obs, final_rewards, final_done = {}, {}, {}

        # Repeat until at least one agent has performed the action
        while len(final_obs) == 0:
            state, reward, done, info = self.env.step(action_dict)
            for agent_id, agent_obs in state.items():
                # The agent choose
                if done[agent_id] or self._on_decision_cell(self.unwrapped.rail_env.agents[agent_id]):
                    final_obs[agent_id] = agent_obs
                    final_rewards[agent_id] = reward[agent_id]
                    final_done[agent_id] = done[agent_id]

                    # If rewards were accumulated now are collected
                    if self._discounting is not None:
                        discounted_skipped_reward = final_rewards[agent_id]

                        # Compute the discounted rewards
                        for skipped_reward in reversed(self._skipped_rewards[agent_id]):
                            discounted_skipped_reward = self._discounting * discounted_skipped_reward + skipped_reward

                        final_rewards[agent_id] = discounted_skipped_reward
                        self._skipped_rewards[agent_id] = []
                # The agent accumulate rewards
                elif self._discounting is not None:
                    self._skipped_rewards[agent_id].append(reward[agent_id])

            final_done['__all__'] = done['__all__']
            # action_dict = {}
        return final_obs, final_rewards, final_done, info

    def reset(self):
        """
        Reset the environment and the decision cells

        :return: observation and info
        """
        obs, info = self.env.reset()
        _, _, self._decision_cells = self._find_all_decision_cells()

        return obs, info
