class DeadlocksDetector:
    def __init__(self):
        self.directions = [
            # North
            (-1, 0),
            # East
            (0, 1),
            # South
            (1, 0),
            # West
            (0, -1)]

    def _check_deadlocks(self, agents, deadlocks, action_dict, env):
        a2 = None

        if env.agents[agents[-1]].position is not None:
            cell_free, new_cell_valid, _, new_position, transition_valid = \
                env._check_action_on_agent(action_dict[agents[-1]], env.agents[agents[-1]])

            if not cell_free and new_cell_valid and transition_valid:
                for a2_tmp in range(env.get_num_agents()):
                    if env.agents[a2_tmp].position == new_position:
                        a2 = a2_tmp
                        break

        if a2 is None:
            return False
        if deadlocks[a2] or a2 in agents:
            return True
        agents.append(a2)
        deadlocks[a2] = self._check_deadlocks(agents, deadlocks, action_dict, env)
        if deadlocks[a2]:
            return True
        del agents[-1]
        return False

    def deadlocks_detection(self, env, action_dict, deadlocks, done):
        agents = []
        for a in range(env.get_num_agents()):
            if not done[a]:
                agents.append(a)
                if not deadlocks[a]:
                    deadlocks[a] = self._check_deadlocks(agents, deadlocks, action_dict, env)
                if not (deadlocks[a]):
                    del agents[-1]

        return deadlocks