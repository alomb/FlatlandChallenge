import wandb

from src.d3qn.d3qn_main import train

if __name__ == "__main__":

    server = False
    if server:
        sweep_config = {
            "name": "d3qn-sweeps",
            "method": "grid",
            "metric": {
              "name": "metrics/accumulated_score",
              "goal": "maximize"
            },
            "parameters": {
                "double_dqn": {
                    "values": [False, True]
                },
                "shared": {
                    "values": [False, True]
                },
                "hidden_size": {
                    "values": [256, 128]
                },
                "hidden_layers": {
                    "values": [2, 3]
                },
                "update_every": {
                    "values": [8, 16, 32]
                },
                "type": {
                    "values": [1, 2]
                },
                # "eps_decay": {
                #    "values": [0.99, 0.98]
                # },
                "learning_rate": {
                    "values": [0.52e-4, 0.52e-3]
                },
                "batch_size": {
                    "values": [32, 128]
                },
                # "memory_type": {
                #    "values": ["per", "uer"]
                # },
            }
        }

        sweep_id = wandb.sweep(sweep_config, entity="fiorenzoparascandolo", project="flatland-challenge-d3qn")
        print("Sweep id: ", sweep_id)

    sweep_id = "3gdemc4a"
    wandb.agent(sweep_id, function=train, entity="fiorenzoparascandolo", project="flatland-challenge-d3qn")
