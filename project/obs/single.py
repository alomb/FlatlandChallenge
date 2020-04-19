import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder

class SingleAgentNavigationObs(ObservationBuilder):
    """
    An observation is a dictionary:
    {"observations": List[List[int]],
    "position": tuple}

    e.g. position = (12, 4) and observations = [[1 0 0], [0 1 0]] means: I'm in the position (12, 4) in the grid and
    I can perform two actions:
    - Move left;
    - Move forward
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get(self, handle: int = 0):
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
            observations.append([0, 1, 0])
        else:
            i = 0
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    observation = [0, 0, 0]
                    observation[i] = 1
                    observations.append(observation)
                i = i + 1

        return {"observations": observations, "state": position}