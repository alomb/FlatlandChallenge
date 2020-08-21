from argparse import Namespace
from datetime import datetime

from flatland.envs.malfunction_generators import MalfunctionParameters

from src.psppo.ps_ppo_flatland import train_multiple_agents

if __name__ == "__main__":
    myseed = 14

    # namefile = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    namefile = "psppo_naive"
    print("Filename: {}".format(namefile))

    environment_parameters = {
        "n_agents": 3,
        "x_dim": 16 * 3,
        "y_dim": 9 * 3,
        "n_cities": 5,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "seed": myseed,
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 30,
        # Malfunctions
        "malfunction_parameters": MalfunctionParameters(
            malfunction_rate=0,
            min_duration=15,
            max_duration=50),
        # Speeds
        "speed_profiles": {
            1.: 1.0,
            1. / 2.: 0.0,
            1. / 3.: 0.0,
            1. / 4.: 0.0},

        # ============================
        # Custom observations&rewards
        # ============================
        "custom_observations": True,

        "stop_penalty": 0.0,
        "invalid_action_penalty": 0.0,
        "deadlock_penalty": 0.0,
        "shortest_path_penalty_coefficient": 1.0,
        # 1.0 for skipping
        "done_bonus": 0.0,
    }

    training_parameters = {
        # ============================
        # Network architecture
        # ============================
        # Shared actor-critic layer
        # If shared is True then the considered sizes are taken from the critic
        "shared": True,
        # Policy network
        "critic_mlp_width": 128,
        "critic_mlp_depth": 4,
        "last_critic_layer_scaling": 0.1,
        # Actor network
        "actor_mlp_width": 128,
        "actor_mlp_depth": 4,
        "last_actor_layer_scaling": 0.01,
        # Adam learning rate
        "learning_rate": 0.001,
        # Adam epsilon
        "adam_eps": 1e-5,
        # Activation
        "activation": "Tanh",
        "lmbda": 0.95,
        "entropy_coefficient": 0.1,
        # Called also baseline cost in shared setting (0.5)
        # (C54): {0.001, 0.1, 1.0, 10.0, 100.0}
        "value_loss_coefficient": 0.001,

        # ============================
        # Training setup
        # ============================
        "n_episodes": 2500,
        # 512, 1024, 2048, 4096
        "horizon": 1024,
        "epochs": 4,
        # 64, 128, 256
        "batch_size": 64,

        # ============================
        # Normalization and clipping
        # ============================
        # Discount factor (0.95, 0.97, 0.99, 0.999)
        "discount_factor": 0.99,
        "max_grad_norm": 0.5,
        # PPO-style value clipping
        "eps_clip": 0.25,

        # ============================
        # Advantage estimation
        # ============================
        # gae or n-steps
        "advantage_estimator": "gae",

        # ============================
        # Optimization and rendering
        # ============================
        # Save and evaluate interval
        "checkpoint_interval": 75,
        "eval_episodes": None,
        "use_gpu": False,
        "render": False,
        "save_model_path": "checkpoint.pt",
        "load_model_path": "checkpoint.pt",
        "tensorboard_path": "log/",

        # ============================
        # Action Masking / Skipping
        # ============================
        "action_masking": False,
        "allow_no_op": True,
        "action_skipping": False
    }

    """
    # Save on Google Drive on Colab
    "save_model_path": "/content/drive/My Drive/Colab Notebooks/models/" + namefile + ".pt",
    "load_model_path": "/content/drive/My Drive/Colab Notebooks/models/todo.pt",
    "tensorboard_path": "/content/drive/My Drive/Colab Notebooks/logs/logs" + namefile + "/",
    """

    """
    # Mount Drive on Colab
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)

    # Show Tensorboard on Colab
    import tensorflow
    %load_ext tensorboard
    % tensorboard --logdir "/content/drive/My Drive/Colab Notebooks/logs_todo"
    """

    train_multiple_agents(Namespace(**environment_parameters), Namespace(**training_parameters))
