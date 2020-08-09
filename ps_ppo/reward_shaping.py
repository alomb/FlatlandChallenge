from flatland.envs.rail_env import RailEnvActions
import numpy as np


def check_feasible_transitions(pos_a1, transitions, env):
    if transitions[0] == 1:
        position_check = (pos_a1[0] - 1, pos_a1[1])
        if not (env.cell_free(position_check)):
            for a2 in range(env.get_num_agents()):
                if env.agents[a2].position == position_check:
                    return a2

    if transitions[1] == 1:
        position_check = (pos_a1[0], pos_a1[1] + 1)
        if not (env.cell_free(position_check)):
            for a2 in range(env.get_num_agents()):
                if env.agents[a2].position == position_check:
                    return a2

    if transitions[2] == 1:
        position_check = (pos_a1[0] + 1, pos_a1[1])
        if not (env.cell_free(position_check)):
            for a2 in range(env.get_num_agents()):
                if env.agents[a2].position == position_check:
                    return a2

    if transitions[3] == 1:
        position_check = (pos_a1[0], pos_a1[1] - 1)
        if not (env.cell_free(position_check)):
            for a2 in range(env.get_num_agents()):
                if env.agents[a2].position == position_check:
                    return a2

    return None


def check_next_pos(a1, env):
    if env.agents[a1].position is not None:
        pos_a1 = env.agents[a1].position
        dir_a1 = env.agents[a1].direction
    else:
        pos_a1 = env.agents[a1].initial_position
        dir_a1 = env.agents[a1].initial_direction

    # NORTH
    if dir_a1 == 0:
        if env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1)[dir_a1] == 1:
            position_check = (pos_a1[0] - 1, pos_a1[1])
            if not (env.cell_free(position_check)):
                for a2 in range(env.get_num_agents()):
                    if env.agents[a2].position == position_check:
                        return a2
        else:
            return check_feasible_transitions(pos_a1, env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1), env)

    # EAST
    if dir_a1 == 1:
        if env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1)[dir_a1] == 1:
            position_check = (pos_a1[0], pos_a1[1] + 1)
            if not (env.cell_free(position_check)):
                for a2 in range(env.get_num_agents()):
                    if env.agents[a2].position == position_check:
                        return a2
        else:
            return check_feasible_transitions(pos_a1, env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1), env)

    # SOUTH
    if dir_a1 == 2:
        if env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1)[dir_a1] == 1:
            position_check = (pos_a1[0] + 1, pos_a1[1])
            if not (env.cell_free(position_check)):
                for a2 in range(env.get_num_agents()):
                    if env.agents[a2].position == position_check:
                        return a2
        else:
            return check_feasible_transitions(pos_a1, env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1), env)

    # WEST
    if dir_a1 == 3:
        if env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1)[dir_a1] == 1:
            position_check = (pos_a1[0], pos_a1[1] - 1)
            if not (env.cell_free(position_check)):
                for a2 in range(env.get_num_agents()):
                    if env.agents[a2].position == position_check:
                        return a2
        else:
            return check_feasible_transitions(pos_a1, env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1), env)

    return None


def check_deadlocks(a1, deadlocks, env):
    a2 = check_next_pos(a1[-1], env)

    if a2 is None:
        return False
    if deadlocks[a2] or a2 in a1:
        return True
    a1.append(a2)
    deadlocks[a2] = check_deadlocks(a1, deadlocks, env)
    if deadlocks[a2]:
        return True
    del a1[-1]
    return False


def check_invalid_transitions(env, action_dict):
    invalid_rewards_shaped = {a: 0 for a in range(env.get_num_agents())}
    moving = [env.agents[a].moving for a in range(env.get_num_agents())]
    agent_speed_data = [env.agents[a].speed_data['transition_action_on_cellexit'] for a in range(env.get_num_agents())]
    action_dict_app = action_dict

    for a in range(env.get_num_agents()):
        if np.isclose(env.agents[a].speed_data['position_fraction'], 0.0, rtol=1e-03):
            if action_dict_app[a] is None:
                action_dict_app[a] = 0
            if action_dict_app[a] < 0:
                action_dict_app[a] = 0
            if action_dict_app[a] == 0 and moving[a]:
                action_dict_app[a] = 2
            if action_dict_app[a] == 4 and moving[a]:
                moving[a] = False
                invalid_rewards_shaped[a] -= 1
            if not moving[a] and not (action_dict_app[a] == 0 and action_dict_app[a] == 4):
                moving[a] = False
                invalid_rewards_shaped[a] -= 1  # Start penalty
            if moving[a]:
                action_stored = False
                _, new_cell_valid, new_direction, new_position, transition_valid = \
                    env._check_action_on_agent(RailEnvActions.MOVE_FORWARD, env.agents[a])
                if all([new_cell_valid, transition_valid]):
                    agent_speed_data[a] = action_dict[a]
                    action_stored = True
                if not action_stored:
                    # If the agent cannot move due to an invalid transition, we set its state to not moving
                    invalid_rewards_shaped[a] -= 2  # Invalid action penalty
                    invalid_rewards_shaped[a] -= 1  # Stop penalty

    return invalid_rewards_shaped


def step_shaping(env, action_dict, deadlocks, shortest_path):
    invalid_rewards_shaped = check_invalid_transitions(env, action_dict)
    # Environment step
    obs, rewards, done, info = env.step(action_dict)

    agents = []
    for a in range(env.get_num_agents()):
        if not done[a]:
            agents.append(a)
            if not deadlocks[a]:
                deadlocks[a] = check_deadlocks(agents, deadlocks, env)
            if not (deadlocks[a]):
                del agents[-1]

    new_shortest_path = [obs.get(a)[6] if obs.get(a) is not None else 0 for a in range(env.get_num_agents())]

    invalid_rewards_shaped = {a: invalid_rewards_shaped[a] + rewards[a] for a in range(env.get_num_agents())}

    rewards_shaped_shortest_path = {a: 2.0 * invalid_rewards_shaped[a] if shortest_path[a] < new_shortest_path[a]
                                    else invalid_rewards_shaped[a] for a in range(env.get_num_agents())}

    rewards_shaped_deadlocks = {a: -5.0 if deadlocks[a] else rewards_shaped_shortest_path[a]
                                for a in range(env.get_num_agents())}

    return obs, rewards, done, info, rewards_shaped_deadlocks, deadlocks, new_shortest_path
