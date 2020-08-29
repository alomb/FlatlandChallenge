import wandb

from src.psppo.ps_ppo_main import train

if __name__ == "__main__":
    sweep_config = {
        "name": "ps-ppo-sweeps",
        "method": "random",  # grid, random
        "metric": {
          "name": "metrics/accumulated_score",
          "goal": "maximize"
        },
        "parameters": {
            "shared": {
                "values": [False, True]
            },
            "critic_mlp_width": {
                "values": [128, 256]
            },
            "activation": {
                "values": ["Tanh", "ReLU"]
            },
            "learning_rate": {
                "values": [0.001, 0.0003]
            },
            "value_loss_coefficient": {
                "values": [0.001, 0.1]
            },

        }
    }

    sweep_id = wandb.sweep(sweep_config, entity="lomb", project="flatland-challenge-lorem-ipsum-dolor-sit-amet")

    wandb.agent(sweep_id, train)
