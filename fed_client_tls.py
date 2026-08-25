"""
Week 2, Day 3 -- Secure Communication: TLS-enabled Federated Client

Same as fed_client.py (real local training on partitioned data), but now
connects to the server over an ENCRYPTED TLS connection instead of plain
gRPC. The client verifies the server's certificate against the shared CA
before sending any data -- this is what prevents a hospital from
accidentally connecting to (and leaking updates to) an impostor server.

Run fed_server_tls.py FIRST, then run this 3 times:
    python fed_client_tls.py --client_id 1
    python fed_client_tls.py --client_id 2
    python fed_client_tls.py --client_id 3
"""

import argparse
import sys
import os
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch, Dataset, DataLoader
from monai.transforms import Compose, Activations, AsDiscrete
import flwr as fl

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "baseline"))
sys.path.append(os.path.dirname(__file__))
from dataset import get_synthetic_datalist  # noqa: E402
from data_pipeline import get_train_transforms  # noqa: E402
from fed_utils import build_model, get_model_parameters, set_model_parameters, DEVICE
from fed_client import FedMedClient  # reuse the real training logic from Day 1-2

CERTS_DIR = os.path.join(os.path.dirname(__file__), "certs")
NUM_CLIENTS = 3


def load_ca_certificate():
    ca_path = os.path.join(CERTS_DIR, "ca.crt")
    if not os.path.exists(ca_path):
        raise FileNotFoundError(
            f"Missing CA certificate: {ca_path}\n"
            "Run `python generate_certs.py` first (only needs to be done once, on the server side, "
            "then share ca.crt with each client)."
        )
    with open(ca_path, "rb") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True, help="Which mock hospital (1, 2, or 3)")
    args = parser.parse_args()

    root_certificate = load_ca_certificate()
    print(f"[Hospital {args.client_id}] Loaded CA certificate -- will verify server identity over TLS.")

    print(f"Starting Hospital {args.client_id} client over a SECURE (TLS) connection...")
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FedMedClient(client_id=args.client_id),
        root_certificates=root_certificate,
    )


if __name__ == "__main__":
    main()
