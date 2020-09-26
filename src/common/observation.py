import numpy as np
from flatland.envs.agent_utils import RailAgentStatus

from flatland.envs.observations import TreeObsForRailEnv


class NormalizeObservations:
    """
    Class containing functions to normalize observations.
    """
    def __init__(self,
                 observation_dim,
                 observation_tree_depth,
                 custom_observations,
                 width,
                 height,
                 observation_radius):

        self.custom_observations = custom_observations
        self.custom_obs = np.zeros((height, width, 16))
        self.observation_tree_depth = observation_tree_depth
        self.observation_radius = observation_radius
        # Calculate the state size given the depth of the tree observation and the number of features
        self.n_features_per_node = observation_dim
        self.n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
        # State size depends on features per nodes in observations, custom observations and + 1 (agent id of PS-PPO)
        self.state_size = self.n_features_per_node * self.n_nodes + (custom_observations * (width * height * 23 + 1))

    def _get_custom_observations(self, env, handle, agent_obs, deadlock):
        agent = env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        obs_targets = np.zeros((env.height, env.width, 2))
        obs_agents_state = np.zeros((env.height, env.width, 5)) - 1

        obs_agents_state[:, :, 4] = 0

        obs_agents_state[agent_virtual_position][0] = agent.direction
        obs_targets[agent.target][0] = 1

        for i in range(len(env.agents)):
            other_agent = env.agents[i]

            # ignore other agents not in the grid any more
            if other_agent.status == RailAgentStatus.DONE_REMOVED:
                continue

            obs_targets[other_agent.target][1] = 1

            # second to fourth channel only if in the grid
            if other_agent.position is not None:
                # second channel only for other agents
                if i != handle:
                    obs_agents_state[other_agent.position][1] = other_agent.direction
                obs_agents_state[other_agent.position][2] = other_agent.malfunction_data['malfunction']
                obs_agents_state[other_agent.position][3] = other_agent.speed_data['speed']
            # fifth channel: all ready to depart on this position
            if other_agent.status == RailAgentStatus.READY_TO_DEPART:
                obs_agents_state[other_agent.initial_position][4] += 1

        agent_obs = np.append(agent_obs, np.clip(obs_targets, 0, 1))
        agent_obs = np.append(agent_obs, np.clip(obs_agents_state, 0, 1))
        agent_obs = np.append(agent_obs, self.custom_obs)

        if deadlock:
            agent_obs = np.append(agent_obs, [1])
        else:
            agent_obs = np.append(agent_obs, [0])

        return agent_obs

    def _split_node_into_feature_groups(self, node):
        data = np.zeros(6)
        distance = np.zeros(1)
        agent_data = np.zeros(4)

        data[0] = node.dist_own_target_encountered
        data[1] = node.dist_other_target_encountered
        data[2] = node.dist_other_agent_encountered
        data[3] = node.dist_potential_conflict
        data[4] = node.dist_unusable_switch
        data[5] = node.dist_to_next_branch

        distance[0] = node.dist_min_to_target

        agent_data[0] = node.num_agents_same_direction
        agent_data[1] = node.num_agents_opposite_direction
        agent_data[2] = node.num_agents_malfunctioning
        agent_data[3] = node.speed_min_fractional

        return data, distance, agent_data

    def _min_gt(self, seq, val):
        """
        Return smallest item in seq for which item > val applies.
        None is returned if seq was empty or all items in seq were >= val.
        """
        min_item = np.inf
        idx = len(seq) - 1
        while idx >= 0:
            if val <= seq[idx] < min_item:
                min_item = seq[idx]
            idx -= 1
        return min_item

    def _max_lt(self, seq, val):
        """
        Return greatest item in seq for which item < val applies.
        None is returned if seq was empty or all items in seq were >= val.
        """
        max_item = 0
        idx = len(seq) - 1
        while idx >= 0:
            if val > seq[idx] >= 0 and seq[idx] > max_item:
                max_item = seq[idx]
            idx -= 1
        return max_item

    def _norm_obs_clip(self, obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
        """
        This function returns the difference between min and max value of an observation
        :param obs: Observation that should be normalized
        :param clip_min: min value where observation will be clipped
        :param clip_max: max value where observation will be clipped
        :param fixed_radius:
        :param normalize_to_range:
        :return: returns normalized and clipped observation
        """
        if fixed_radius > 0:
            max_obs = fixed_radius
        else:
            max_obs = max(1, self._max_lt(obs, 1000)) + 1

        min_obs = 0  # min(max_obs, min_gt(obs, 0))
        if normalize_to_range:
            min_obs = self._min_gt(obs, 0)
        if min_obs > max_obs:
            min_obs = max_obs
        if max_obs == min_obs:
            return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
        norm = np.abs(max_obs - min_obs)
        return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)

    def _split_subtree_into_feature_groups(self, node, current_tree_depth, max_tree_depth):
        if node == -np.inf:
            remaining_depth = max_tree_depth - current_tree_depth
            # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
            num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
            return [-np.inf] * num_remaining_nodes * 6, [-np.inf] * num_remaining_nodes, [
                -np.inf] * num_remaining_nodes * 4

        data, distance, agent_data = self._split_node_into_feature_groups(node)

        if not node.childs:
            return data, distance, agent_data

        for direction in TreeObsForRailEnv.tree_explored_actions_char:
            sub_data, sub_distance, sub_agent_data = self._split_subtree_into_feature_groups(
                node.childs[direction], current_tree_depth + 1, max_tree_depth)
            data = np.concatenate((data, sub_data))
            distance = np.concatenate((distance, sub_distance))
            agent_data = np.concatenate((agent_data, sub_agent_data))

        return data, distance, agent_data

    def _split_tree_into_feature_groups(self, tree, max_tree_depth):
        """
        This function splits the tree into three difference arrays of values
        """
        data, distance, agent_data = self._split_node_into_feature_groups(tree)

        for direction in TreeObsForRailEnv.tree_explored_actions_char:
            sub_data, sub_distance, sub_agent_data = self._split_subtree_into_feature_groups(tree.childs[direction], 1,
                                                                                             max_tree_depth)
            data = np.concatenate((data, sub_data))
            distance = np.concatenate((distance, sub_distance))
            agent_data = np.concatenate((agent_data, sub_agent_data))

        return data, distance, agent_data

    def reset_custom_obs(self, env):
        if self.custom_observations:
            self.custom_obs = np.zeros((env.height, env.width, 16))
            for i in range(self.custom_obs.shape[0]):
                for j in range(self.custom_obs.shape[1]):
                    bitlist = [int(digit) for digit in bin(env.rail.get_full_transitions(i, j))[2:]]
                    bitlist = [0] * (16 - len(bitlist)) + bitlist
                    self.custom_obs[i, j] = np.array(bitlist)

    def normalize_observation(self, obs, env, agent, deadlocks):
        """
        This function normalizes the observation used by the RL algorithm
        """
        data, distance, agent_data = self._split_tree_into_feature_groups(obs, self.observation_tree_depth)

        data = self._norm_obs_clip(data, fixed_radius=self.observation_radius)
        distance = self._norm_obs_clip(distance, normalize_to_range=True)
        agent_data = np.clip(agent_data, -1, 1)
        normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))

        if self.custom_observations:
            return self._get_custom_observations(env, agent, normalized_obs, deadlocks)
        else:
            return normalized_obs
