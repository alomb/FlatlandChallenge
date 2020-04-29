import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder

"""
Observation for a Q-learning based single agent.

An observation is a dictionary of the form:
{"observations": List[List[int]], "position": tuple}

e.g. position = (12, 4) and observations = [[1 0 0], [0 1 0]] means: I'm in the position (12, 4) in the grid and
I can perform two actions:
- Move left;
- Move forward
"""


class SingleAgentNavigationObs(ObservationBuilder):

    """
    """
    def __init__(self):
        super().__init__()

    """
    """
    def reset(self):
        # TODO
        pass

    """
    Args:
        handle: the agent index
    """
    def get(self, handle=0):
        agent = self.env.agents[handle]
        observations = []

        """
        Possible transitions are tuples of four ints (N, E, S, W). 
        The directions are always absolute, and relative to the environment not to the agent, like a compass.
        e.g.
            (1,0,0,1) you can only go N or W
        """
        if agent.position:
            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
            position = agent.position
        else:
            possible_transitions = self.env.rail.get_transitions(*agent.initial_position, agent.direction)
            position = agent.initial_position

        num_transitions = np.count_nonzero(possible_transitions)

        if num_transitions == 1:
            observations.append([0, 1, 0])
        else:
            i = 0
            """
            The avaiable movement directions are 3/4 because the train cannot move in opposite directions, but only 
            forward, left or right.

            e.g.
                If agent.direction is at N, we can move E (+1), N (+0) or W (-1)
            """
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    observation = [0, 0, 0]
                    observation[i] = 1
                    observations.append(observation)
                i = i + 1
        return {"observations": observations, "state": position}


"""
Observation for a DQN based single agent.

An observation is a dictionary of the form:
{"observations": List[List[int],List[int],List[int]], "position": tuple}

Similar to the Q-learning version but returns always the same structure, even impossible transitions in the form 
[0,0,0].
"""


class SingleDQNAgentObs(ObservationBuilder):

    def __init__(self):
        super().__init__()

    """
    """
    def reset(self):
        # TODO
        pass

    """
    Args:
        handle: the agent index
    """
    def get(self, handle=0):
        agent = self.env.agents[handle]
        observations = []

        if agent.position:
            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
            position = agent.position
        else:
            possible_transitions = self.env.rail.get_transitions(*agent.initial_position, agent.direction)
            position = agent.initial_position

        num_transitions = np.count_nonzero(possible_transitions)

        if num_transitions == 1:
            observations = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        else:
            i = 0
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                observation = [0, 0, 0]
                if possible_transitions[direction]:
                    observation[i] = 1
                observations.append(observation)
                i = i + 1

        return {"observations": observations, "state": position}
