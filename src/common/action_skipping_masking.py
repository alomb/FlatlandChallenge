import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnvActions


def find_decision_cells(env):
    """

    :param env: The RailEnv to inspect
    :return: A set containing decision cells, made by switches and their neighbors
    """

    switches = []
    switches_neighbors = []
    directions = list(range(4))
    for h in range(env.height):
        for w in range(env.width):
            pos = (h, w)
            is_switch = False
            # Check for switch counting the outgoing transition
            for orientation in directions:
                possible_transitions = env.rail.get_transitions(*pos, orientation)
                num_transitions = np.count_nonzero(possible_transitions)
                if num_transitions > 1:
                    switches.append(pos)
                    is_switch = True
                    break
            if is_switch:
                # Add all neighbouring rails, if pos is a switch
                for orientation in directions:
                    possible_transitions = env.rail.get_transitions(*pos, orientation)
                    for movement in directions:
                        if possible_transitions[movement]:
                            switches_neighbors.append(get_new_position(pos, movement))

    return set(switches).union(set(switches_neighbors))


def get_action_masking(env, agent, action_size, train_params):
    """

    :param env: the environment
    :param agent: the agent index/handler
    :param action_size: the environment's number of available actions
    :param train_params: training parameters to customize the mask
    :return: the action mask for the passed agent
    """

    # Mask initialization
    action_mask = [1 * (0 if action == RailEnvActions.DO_NOTHING and not train_params.allow_no_op else 1)
                   for action in range(action_size)]

    # Mask filling
    if train_params.action_masking:
        for action in range(action_size):
            """
            Control if the agent is in the scene has a position, excluding when it has been arrived and removed
            and when has not already started. In these cases the action masks is the initial one.
            """
            if env.get_rail_env().agents[agent].position is not None:

                _, cell_valid, _, _, transition_valid = env.get_rail_env()._check_action_on_agent(
                    RailEnvActions(action),
                    env.get_rail_env().agents[agent])

                if not all([cell_valid, transition_valid]):
                    action_mask[action] = 0

    return action_mask
