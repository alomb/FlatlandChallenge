import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus


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

        self.deadlocks = []

    def reset(self, num_agents):
        self.deadlocks = [False for _ in range(num_agents)]

    def step(self, env):
        agents = []
        for a in range(env.get_num_agents()):
            if env.agents[a].status not in [RailAgentStatus.DONE_REMOVED, RailAgentStatus.READY_TO_DEPART, RailAgentStatus.DONE]:
                agents.append(a)
                if not self.deadlocks[a]:
                    self.deadlocks[a] = self._check_deadlocks(agents, self.deadlocks, env)
                if not (self.deadlocks[a]):
                    del agents[-1]
            else:
                self.deadlocks[a] = False

        return self.deadlocks

    def _check_feasible_transitions(self, pos_a1, transitions, env):
        for direction, values in enumerate(self.directions):
            if transitions[direction] == 1:
                position_check = (pos_a1[0] + values[0], pos_a1[1] + values[1])
                if not (env.cell_free(position_check)):
                    for a2 in range(env.get_num_agents()):
                        if env.agents[a2].position == position_check:
                            return a2

        return None

    def _check_next_pos(self, a1, env):

        pos_a1 = env.agents[a1].position
        dir_a1 = env.agents[a1].direction

        if env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1)[dir_a1] == 1:
            position_check = (pos_a1[0] + self.directions[dir_a1][0], pos_a1[1] + self.directions[dir_a1][1])
            if not (env.cell_free(position_check)):
                for a2 in range(env.get_num_agents()):
                    if env.agents[a2].position == position_check:
                        return a2
        else:
            return self._check_feasible_transitions(pos_a1, env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1), env)

    def _check_deadlocks(self, a1, deadlocks, env):
        a2 = self._check_next_pos(a1[-1], env)

        if a2 is None:
            return False
        if deadlocks[a2] or a2 in a1:
            return True
        a1.append(a2)
        deadlocks[a2] = self._check_deadlocks(a1, deadlocks, env)
        if deadlocks[a2]:
            return True
        del a1[-1]
        return False

    """
    def _check_deadlock(self, rail_env):
        # For each active and not deadlocked agent
        apple = list(filter(lambda a: a.status == RailAgentStatus.ACTIVE and not self.deadlocks[a.handle],
                                 rail_env.agents))
        for agent in list(filter(lambda a: a.status == RailAgentStatus.ACTIVE and not self.deadlocks[a.handle],
                                 rail_env.agents)):

            if agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED:
                print("EH!")

            position = agent.position
            direction = agent.direction

            while position is not None:
                possible_transitions = rail_env.rail.get_transitions(*position, direction)
                num_transitions = np.count_nonzero(possible_transitions)
                # If the agent can only move towards one direction
                if num_transitions == 1:
                    new_direction_me = np.argmax(possible_transitions)
                    new_cell_me = get_new_position(position, new_direction_me)
                    opp_agent = rail_env.agent_positions[new_cell_me]
                    # and the next cell contains an agent
                    if opp_agent != -1:
                        # collect information about the movement of the opposite agent, its next position, its next
                        # direction and its number of transitions
                        opp_position = rail_env.agents[opp_agent].position
                        opp_direction = rail_env.agents[opp_agent].direction
                        opp_possible_transitions = rail_env.rail.get_transitions(*opp_position, opp_direction)
                        opp_num_transitions = np.count_nonzero(opp_possible_transitions)
                        # If only one movement is valid
                        if opp_num_transitions == 1:
                            # Check if there is not an head to back collision
                            if opp_direction != direction:
                                self.deadlocks[agent.handle] = True
                                position = None
                            # Else check opposite agent to discover chains of collisions
                            else:
                                position = new_cell_me
                                direction = new_direction_me
                        # If the opposite agent can move away from the collision also its alternative movements must be
                        # controlled
                        else:
                            position = new_cell_me
                            direction = new_direction_me
                    else:
                        position = None
                else:
                    position = None
        return self.deadlocks
    
    def step(self, env):
        deads = self._check_deadlock(env)
        return deads
    
    """

