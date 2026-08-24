"""
Day 5 — Flower Client (dummy, for connectivity testing)

Simulates ONE hospital node connecting to the central server. For Day 5,
this returns random/dummy weights instead of actually training on real
data -- the goal is just to prove the client can connect, participate in
a round, and exchange weights with the server. Real per-hospital training
logic (on your MONAI U-Net) gets wired in during Week 2.

Run server.py FIRST in one terminal, then run this in another terminal.
In Day 6, you'll run 3 copies of this (or a parameterized version) to
simulate 3 hospitals.

Run:
    python client.py
"""

import numpy as np
import flwr as fl

# Number of dummy "model parameters" — stand-in for U-Net weights.
# Keep this small for Day 5; real U-Net weight arrays come in Week 2.
DUMMY_PARAM_SHAPES = [(10, 10), (10,), (5, 10), (5,)]


def get_dummy_parameters():
    """Random arrays standing in for model weights."""
    return [np.random.randn(*shape).astype(np.float32) for shape in DUMMY_PARAM_SHAPES]


class DummyHospitalClient(fl.client.NumPyClient):
    """A stub client representing one hospital node. Implements the 3
    methods Flower requires: get_parameters, fit, evaluate."""

    def __init__(self):
        self.parameters = get_dummy_parameters()

    def get_parameters(self, config):
        print("[Client] Server requested current parameters.")
        return self.parameters

    def fit(self, parameters, config):
        # Real version (Week 2): load `parameters` into the U-Net, train
        # locally on this hospital's private data, return updated weights.
        print("[Client] Received global parameters. Simulating local training...")
        self.parameters = [p + np.random.randn(*p.shape).astype(np.float32) * 0.01 for p in parameters]
        num_examples = 42  # stand-in for len(local_dataset)
        return self.parameters, num_examples, {}

    def evaluate(self, parameters, config):
        # Real version (Week 2): run the U-Net on this hospital's local
        # validation data, return a real loss/Dice metric.
        print("[Client] Received global parameters for evaluation.")
        dummy_loss = float(np.random.uniform(0.3, 0.7))
        num_examples = 10
        return dummy_loss, num_examples, {"dice": float(np.random.uniform(0.5, 0.9))}


def main():
    print("Starting dummy hospital client, connecting to server at localhost:8080...")
    fl.client.start_numpy_client(server_address="localhost:8080", client=DummyHospitalClient())


if __name__ == "__main__":
    main()
