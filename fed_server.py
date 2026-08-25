"""
Week 2, Day 1-2 — Real Federated Training Server

Same FedAvg strategy as Week 1, but now:
- Initializes rounds from your actual U-Net's starting weights (instead of
  clients starting from random weights each time)
- Aggregates REAL Dice scores reported by each hospital after local training
- Runs more rounds since real training actually improves the model each round

Run this FIRST, then run fed_client.py three times to simulate 3
hospitals with real local training:

    python fed_client.py --client_id 1
    python fed_client.py --client_id 2
    python fed_client.py --client_id 3

Run:
    python fed_server.py
"""

import sys
import os
import flwr as fl
from flwr.common import ndarrays_to_parameters

sys.path.append(os.path.dirname(__file__))
from fed_utils import build_model, get_model_parameters

MIN_CLIENTS = 3
NUM_ROUNDS = 5  # a few more rounds now that training is real


def weighted_average(metrics):
    """Aggregates the 'dice' metric across clients, weighted by each client's dataset size."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_dice = sum(num_examples * m["dice"] for num_examples, m in metrics) / total_examples
    return {"dice": weighted_dice}


def main():
    # Start from the model's actual initial weights, not random per-client weights
    initial_model = build_model()
    initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"Starting REAL federated training server — waiting for {MIN_CLIENTS} hospital(s)...")
    print("Run `python fed_client.py --client_id N` (N=1,2,3) in 3 separate terminals now.")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
