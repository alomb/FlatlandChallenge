from flatland.envs.agent_utils import RailAgentStatus
import numpy as np


class Stats:
    def __init__(self):
        self.accumulated_normalized_score = []
        self.accumulated_completion = []
        self.accumulated_deadlocks = []
        # Evaluation statics
        self.accumulated_eval_normalized_score = []
        self.accumulated_eval_completion = []
        self.accumulated_eval_deads = []

    def step(self, score, max_steps, num_agents, info, deadlocks, action_count):
        # Collection information about training
        normalized_score = score / (max_steps * num_agents)
        tasks_finished = sum(info["status"][a] in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]
                             for a in range(num_agents))
        completion_percentage = tasks_finished / max(1, num_agents)
        deadlocks_percentage = sum(deadlocks) / num_agents
        action_probs = action_count / np.sum(action_count)

        # Mean values for terminal display and for more stable hyper-parameter tuning
        self.accumulated_normalized_score.append(normalized_score)
        self.accumulated_completion.append(completion_percentage)
        self.accumulated_deadlocks.append(deadlocks_percentage)

        return normalized_score, tasks_finished, completion_percentage, deadlocks_percentage, action_probs
