"""
Day 5 — Flower Server

Starts a central Flower server that coordinates federated learning rounds
using the FedAvg (Federated Averaging) strategy.

For Day 5, this just proves the networking/framework skeleton works --
clients will connect and exchange DUMMY weights, not real trained ones.
Real training logic gets wired in during Week 2.

Run this FIRST, then run client.py in 3 separate terminals (Day 6).

Run:
    python server.py
"""

import flwr as fl

# Minimum number of clients that must connect before a round starts.
# For Day 5 testing, keep this low (1) so you can test with a single client
# before scaling to 3 "hospitals" in Day 6.
MIN_CLIENTS = 1
NUM_ROUNDS = 3  # how many federated learning rounds to run


def main():
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=MIN_CLIENTS,       # min clients needed for training
        min_evaluate_clients=MIN_CLIENTS,  # min clients needed for evaluation
        min_available_clients=MIN_CLIENTS, # min clients that must be connected at all
    )

    print(f"Starting Flower server — waiting for at least {MIN_CLIENTS} client(s) to connect...")
    print("Run `python client.py` in another terminal now.")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
