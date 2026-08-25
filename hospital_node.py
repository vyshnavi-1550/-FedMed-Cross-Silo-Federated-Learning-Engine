"""
Week 1 - Node Scaffolding: Mock Hospital Node

Represents ONE hospital in the FedMed network. Run this script 3 times
(in 3 separate terminals) with different --hospital-id and --node-port
values to simulate 3 distinct hospitals.

- Binds its OWN local port (e.g. 9001, 9002, 9003) to represent itself as a
  distinct, independently-running node, satisfying the "3 distinct mock
  Hospital Nodes running on separate local ports" requirement.
- ALSO connects out to the central Flower server (flower_server.py, on
  port 8080) to participate in the federated round.

For Week 1, this node does NOT train on real patient data yet -- it just
returns placeholder ("dummy") weights so we can confirm the full pipeline
(3 nodes -> central server -> aggregate -> back) works end-to-end. Real
local training on each hospital's data partition is wired in during
Week 2 (Federated Training Loop).

Run (three separate terminals):
    python hospital_node.py --hospital-id 1 --node-port 9001
    python hospital_node.py --hospital-id 2 --node-port 9002
    python hospital_node.py --hospital-id 3 --node-port 9003
"""

import argparse
import socket
import threading

import numpy as np
import flwr as fl

SERVER_ADDRESS = "127.0.0.1:8080"


def hold_local_port(node_port: int, hospital_id: int):
    """
    Binds this node's own local port so it exists as a distinct, addressable
    node on the network (as the plan requires), independent of the outbound
    connection this node also makes to the central server.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", node_port))
    sock.listen(1)
    print(f"[Hospital {hospital_id}] Listening on local port {node_port} (node identity)")

    def serve_forever():
        while True:
            try:
                conn, _ = sock.accept()
                conn.close()
            except OSError:
                break

    t = threading.Thread(target=serve_forever, daemon=True)
    t.start()
    return sock


class MockHospitalClient(fl.client.NumPyClient):
    """
    Placeholder Flower client for Week 1. Returns dummy weights so the
    connect -> broadcast -> aggregate loop can be verified without needing
    the real U-Net or real patient data yet.
    """

    def __init__(self, hospital_id: int):
        self.hospital_id = hospital_id
        # Dummy parameters standing in for real model weights (Week 2 will
        # replace these with the actual U-Net's state_dict tensors).
        self.dummy_weights = [np.zeros((4,), dtype=np.float32)]

    def get_parameters(self, config):
        return self.dummy_weights

    def fit(self, parameters, config):
        print(f"[Hospital {self.hospital_id}] Received global weights from server. "
              f"(Local training on real data starts in Week 2.)")
        # No real training yet -- just echo back to prove the round completes.
        return self.dummy_weights, 1, {}

    def evaluate(self, parameters, config):
        # No real Dice evaluation yet -- placeholder loss/metric.
        return 0.0, 1, {"dice": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hospital-id", type=int, required=True, help="1, 2, or 3")
    parser.add_argument("--node-port", type=int, required=True, help="e.g. 9001, 9002, 9003")
    args = parser.parse_args()

    hold_local_port(args.node_port, args.hospital_id)

    print(f"[Hospital {args.hospital_id}] Connecting to central server at {SERVER_ADDRESS}...")

    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=MockHospitalClient(args.hospital_id).to_client(),
    )

    print(f"[Hospital {args.hospital_id}] Round complete. Disconnected from server.")


if __name__ == "__main__":
    main()
