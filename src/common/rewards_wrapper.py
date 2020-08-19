import gym
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions


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
        self.shortest_path = [obs.get(a)[6] if obs.get(a) is not None else 0 for a in range(self.unwrapped.rail_env.get_num_agents())]

        return obs, info

    def step(self, action_dict):

        num_agents = self.unwrapped.rail_env.get_num_agents()

        invalid_rewards_shaped = self._check_invalid_transitions(action_dict)
        stop_rewards_shaped = self._check_stop_transition(action_dict, invalid_rewards_shaped)

        # Environment step
        obs, rewards, done, info = self.env.step(action_dict)

        new_shortest_path = [obs.get(a)[6] if obs.get(a) is not None else 0 for a in range(num_agents)]

        new_rewards_shaped = {
            a: rewards[a] if stop_rewards_shaped[a] == 0 else rewards[a] + stop_rewards_shaped[a]
            for a in range(num_agents)}

        rewards_shaped_shortest_path = {a: self.shortest_path_penalty_coefficient * new_rewards_shaped[a]
        if self.shortest_path[a] < new_shortest_path[a] else new_rewards_shaped[a] for a in
                                        range(num_agents)}

        # If done it always get the done_bonus
        rewards_shaped = {a: self.done_bonus if done[a] else rewards_shaped_shortest_path[a] for a in
                          range(num_agents)}

        return_rewards = {agent: {} for agent in range(self.rail_env.get_num_agents())}

        for agent in range(num_agents):
            return_rewards[agent]["standard_rewards"] = rewards[agent]
            return_rewards[agent]["rewards_shaped"] = rewards_shaped[agent]

        return obs, return_rewards, done, info

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
        return {a: self.stop_penalty if action_dict[a] == RailEnvActions.STOP_MOVING else rewards[a]
                for a in range(len(action_dict))}
