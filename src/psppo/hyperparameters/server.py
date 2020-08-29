import wandb

from src.psppo.ps_ppo_main import train

if __name__ == "__main__":

    server = False
    if server:
        sweep_config = {
            "name": "ps-ppo-sweeps",
            "method": "grid",
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
                "learning_rate": {
                    "values": [0.002, 0.0002, 0.00002]
                },
                "entropy coefficient": {
                    "values": [0.005, 0.02]
                },
                "value_loss_coefficient": {
                    "values": [0.0005, 0.002]
                },
                "horizon": {
                    "values": [1024, 2048]
                },
                "batch_size": {
                    "values": [256, 512]
                },
                "eps": {
                    "values": [0.2, 0.3]
                },
            }
        }

        sweep_id = wandb.sweep(sweep_config, entity="lomb", project="flatland-challenge-lorem-ipsum-dolor-sit-amet")
        print("Sweep id: ", sweep_id)

    sweep_id = "5nhm3coj"
    wandb.agent(sweep_id, function=train, entity="lomb", project="flatland-challenge-lorem-ipsum-dolor-sit-amet")
