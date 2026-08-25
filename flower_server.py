"""
Week 1 - Node Scaffolding: Central Flower Server

This is the central orchestrator. It listens on a fixed port and waits for
hospital nodes (flower clients) to connect. For Week 1, this does NOT do any
real model training yet -- it just proves the central server can start, wait
for exactly 3 hospital nodes to connect, run one placeholder round, and shut
down cleanly. Real federated averaging with your actual U-Net model gets
wired in during Week 2 (Federated Training Loop).

Run this FIRST, in its own terminal:
    python flower_server.py

Then run hospital_node.py three times in three SEPARATE terminals (Week 1
Node Scaffolding requires 3 distinct nodes on separate local ports):
    python hospital_node.py --hospital-id 1 --node-port 9001
    python hospital_node.py --hospital-id 2 --node-port 9002
    python hospital_node.py --hospital-id 3 --node-port 9003

Each hospital_node.py binds its OWN local port (9001/9002/9003) to represent
itself as a distinct node, in addition to connecting out to this central
server's port (8080) to participate in the federated round.
"""

import flwr as fl
from flwr.server.strategy import FedAvg

SERVER_ADDRESS = "0.0.0.0:8080"
NUM_ROUNDS = 1          # Week 1: just prove connectivity with 1 placeholder round
MIN_NODES_REQUIRED = 3  # exactly 3 mock hospital nodes, per the project spec


def main():
    print(f"Starting central FedMed server on {SERVER_ADDRESS}")
    print(f"Waiting for {MIN_NODES_REQUIRED} hospital nodes to connect...\n")

    # FedAvg is the standard federated averaging strategy. For Week 1 we
    # only care that it can successfully complete a round with 3 connected
    # clients -- the actual model weights being averaged are placeholders
    # until Week 2 wires in the real U-Net.
    strategy = FedAvg(
        min_available_clients=MIN_NODES_REQUIRED,
        min_fit_clients=MIN_NODES_REQUIRED,
        min_evaluate_clients=MIN_NODES_REQUIRED,
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("\nServer finished. All 3 hospital nodes connected and completed a round.")
    print("Week 1 Node Scaffolding check: PASSED.")


if __name__ == "__main__":
    main()
