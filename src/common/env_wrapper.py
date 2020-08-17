from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnvActions
import numpy as np

from src.common.deadlocks import DeadlocksDetector


def _check_invalid_transitions(action_dict, action_mask, invalid_action_penalty):
    return {a: invalid_action_penalty if a in action_dict and mask[action_dict[a]] == 0 else 0 for a, mask in
            enumerate(action_mask)}


def _check_stop_transition(action_dict, rewards, stop_penalty):
    return {a: stop_penalty if action_dict[a] == RailEnvActions.STOP_MOVING else rewards[a]
            for a in range(len(action_dict))}


class EnvWrapper:
    def __init__(self,
                 env,
                 invalid_action_penalty,
                 stop_penalty,
                 deadlock_penalty,
                 shortest_path_penalty_coefficient,
                 done_bonus):
        self.env = env
        self.deadlocks = []
        self.shortest_path = []
        self.invalid_action_penalty = invalid_action_penalty
        self.stop_penalty = stop_penalty
        self.deadlock_penalty = deadlock_penalty
        self.shortest_path_penalty_coefficient = shortest_path_penalty_coefficient
        self.done_bonus = done_bonus
        self.deadlock_detector = DeadlocksDetector()

    def reset(self):
        obs, info = self.env.reset(regenerate_rail=True, regenerate_schedule=True)
        self.deadlocks = [False for _ in range(self.env.get_num_agents())]
        self.shortest_path = [obs.get(a)[6] if obs.get(a) is not None else 0 for a in range(self.env.get_num_agents())]
        return obs, info

    def step(self, action_dict, action_mask):
        invalid_rewards_shaped = _check_invalid_transitions(action_dict, action_mask, self.invalid_action_penalty)
        stop_rewards_shaped = _check_stop_transition(action_dict, invalid_rewards_shaped, self.stop_penalty)

        # Environment step
        obs, rewards, done, info = self.env.step(action_dict)
        self.deadlocks = self.deadlock_detector.deadlocks_detection(self.env, action_dict, self.deadlocks, done)

        new_shortest_path = [obs.get(a)[6] if obs.get(a) is not None else 0 for a in range(self.env.get_num_agents())]

        new_rewards_shaped = {
            a: rewards[a] if stop_rewards_shaped[a] == 0 else rewards[a] + stop_rewards_shaped[a]
            for a in range(self.env.get_num_agents())}

        rewards_shaped_shortest_path = {a: self.shortest_path_penalty_coefficient * new_rewards_shaped[a]
        if self.shortest_path[a] < new_shortest_path[a] else new_rewards_shaped[a] for a in range(self.env.get_num_agents())}

        rewards_shaped_deadlocks = {a: self.deadlock_penalty if self.deadlocks[a] and self.deadlock_penalty != 0
        else rewards_shaped_shortest_path[a] for a in range(self.env.get_num_agents())}

        # If done it always get the done_bonus
        rewards_shaped = {a: self.done_bonus if done[a] else rewards_shaped_deadlocks[a] for a in
                          range(self.env.get_num_agents())}

        return obs, rewards, done, info, rewards_shaped

    def find_decision_cells(self):
        switches = []
        switches_neighbors = []
        directions = list(range(4))
        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                is_switch = False
                # Check for switch counting the outgoing transition
                for orientation in directions:
                    possible_transitions = self.env.rail.get_transitions(*pos, orientation)
                    num_transitions = np.count_nonzero(possible_transitions)
                    if num_transitions > 1:
                        switches.append(pos)
                        is_switch = True
                        break
                if is_switch:
                    # Add all neighbouring rails, if pos is a switch
                    for orientation in directions:
                        possible_transitions = self.env.rail.get_transitions(*pos, orientation)
                        for movement in directions:
                            if possible_transitions[movement]:
                                switches_neighbors.append(get_new_position(pos, movement))

        return set(switches).union(set(switches_neighbors))