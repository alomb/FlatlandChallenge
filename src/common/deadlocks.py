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

    def _check_deadlock(self, rail_env):
        # For each active and not deadlocked agent
        for agent in list(filter(lambda a: a.status == RailAgentStatus.ACTIVE and not self.deadlocks[a.handle],
                                 rail_env.agents)):

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
        return self._check_deadlock(env)

