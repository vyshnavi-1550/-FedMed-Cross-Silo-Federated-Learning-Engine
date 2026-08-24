"""
Day 6 — Flower Client (parameterized for multiple hospital nodes)

Same as Day 5's client, but now accepts a --client_id argument so you can
launch 3 separate copies of this script to simulate 3 distinct "hospital"
nodes, all connecting to the same central server.

Run server.py FIRST in one terminal (it now expects 3 clients -- see the
updated MIN_CLIENTS in server.py), then run this script in 3 SEPARATE
terminals with different IDs:

    python client.py --client_id 1
    python client.py --client_id 2
    python client.py --client_id 3
"""

import argparse
import numpy as np
import flwr as fl

DUMMY_PARAM_SHAPES = [(10, 10), (10,), (5, 10), (5,)]


def get_dummy_parameters():
    return [np.random.randn(*shape).astype(np.float32) for shape in DUMMY_PARAM_SHAPES]


class DummyHospitalClient(fl.client.NumPyClient):
    """A stub client representing one hospital node. Implements the 3
    methods Flower requires: get_parameters, fit, evaluate."""

    def __init__(self, client_id):
        self.client_id = client_id
        self.parameters = get_dummy_parameters()

    def get_parameters(self, config):
        print(f"[Hospital {self.client_id}] Server requested current parameters.")
        return self.parameters

    def fit(self, parameters, config):
        # Real version (Week 2): load `parameters` into the U-Net, train
        # locally on THIS hospital's own private data, return updated weights.
        print(f"[Hospital {self.client_id}] Received global parameters. Simulating local training...")
        self.parameters = [p + np.random.randn(*p.shape).astype(np.float32) * 0.01 for p in parameters]
        num_examples = 42  # stand-in for len(this hospital's local dataset)
        return self.parameters, num_examples, {}

    def evaluate(self, parameters, config):
        # Real version (Week 2): run the U-Net on this hospital's local
        # validation data, return a real loss/Dice metric.
        print(f"[Hospital {self.client_id}] Received global parameters for evaluation.")
        dummy_loss = float(np.random.uniform(0.3, 0.7))
        num_examples = 10
        return dummy_loss, num_examples, {"dice": float(np.random.uniform(0.5, 0.9))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, default=1, help="Which mock hospital this represents (1, 2, or 3)")
    args = parser.parse_args()

    print(f"Starting Hospital {args.client_id} client, connecting to server at localhost:8080...")
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=DummyHospitalClient(client_id=args.client_id),
    )


if __name__ == "__main__":
    main()

