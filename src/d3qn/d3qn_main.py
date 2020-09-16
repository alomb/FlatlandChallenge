from argparse import Namespace
from datetime import datetime

from flatland.envs.malfunction_generators import MalfunctionParameters

from src.d3qn.d3qn_flatland import train_multiple_agents, FingerprintType
from src.d3qn.eval_d3qn import eval_policy


def train():
    seed = 14

    namefile = "d3qn_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    print("Running {}".format(namefile))

    environment_parameters = {
        "n_agents": 5,
        "x_dim": 16 * 3,
        "y_dim": 9 * 3,
        "n_cities": 5,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "seed": seed,
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 30,
        # Malfunctions
        "malfunction_parameters": MalfunctionParameters(
            malfunction_rate=0.0075,
            min_duration=15,
            max_duration=50),
        # Speeds
        "speed_profiles": {
            1.: 0.25,
            1. / 2.: 0.25,
            1. / 3.: 0.25,
            1. / 4.: 0.25},

        # ============================
        # Custom observations&rewards
        # ============================
        "custom_observations": False,

        "reward_shaping": True,
        "uniform_reward": True,
        "stop_penalty": -0.2,
        "invalid_action_penalty": -0.0,
        "deadlock_penalty": -5.0,
        # 1.0 for skipping
        "shortest_path_penalty_coefficient": 1.2,
        "done_bonus": 0.2,
    }

    training_parameters = {
        # ============================
        # Network architecture
        # ============================
        "double_dqn": True,
        "shared": True,
        "hidden_size": 256,
        "hidden_layers": 3,
        "update_every": 16,
        "type": 1,

        # epsilon greedy decay regulators
        "eps_decay": 0.99739538258046,
        "eps_start": 1.0,
        "eps_end": 0.02,

        "learning_rate": 0.52e-4,
        # To compute q targets
        "gamma": 0.99,
        # To compute target network soft update
        "tau": 1e-3,

        # ============================
        # Training setup
        # ============================
        "n_episodes": 2500,
        "batch_size": 128,
        # Minimum number of samples to start learning
        "buffer_min_size": 0,
        "fingerprints": True,
        # If not set the default value is the standard FingerprintType.EPSILON_STEP
        "fingerprint_type": FingerprintType.EPSILON_STEP,

        # ============================
        # Memory
        # ============================
        # Memory maximum size
        "buffer_size": int(1e6),
        # memory type uer or per
        "memory_type": "uer",


        # ============================
        # Saving and rendering
        # ============================
        "checkpoint_interval": 100,
        "evaluation_mode": False,
        "eval_episodes": 25,
        "use_gpu": False,
        "render": False,
        "print_stats": True,
        "wandb_project": "flatland-challenge-d3qn-er",
        "wandb_entity": "lomb",
        "wandb_tag": "d3qn",
        "save_model_path": namefile + ".pt",
        "load_model_path": namefile + ".pt",
        "tensorboard_path": "log_" + namefile + "/",

        # ============================
        # Action Masking / Skipping
        # ============================
        "action_masking": True,
        "allow_no_op": False
    }

    if training_parameters["evaluation_mode"]:
        eval_policy(Namespace(**environment_parameters), Namespace(**training_parameters))
    else:
        train_multiple_agents(Namespace(**environment_parameters), Namespace(**training_parameters))


if __name__ == "__main__":
    train()
