"""
Week 2, Day 3 -- Secure Communication: TLS-enabled Federated Server

Same as fed_server.py, but the gRPC connection is now encrypted using TLS
certificates (run generate_certs.py first to create them).

Run:
    python fed_server_tls.py
"""

import sys
import os
import flwr as fl
from flwr.common import ndarrays_to_parameters

sys.path.append(os.path.dirname(__file__))
from fed_utils import build_model, get_model_parameters

MIN_CLIENTS = 3
NUM_ROUNDS = 5
CERTS_DIR = os.path.join(os.path.dirname(__file__), "certs")


def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_dice = sum(num_examples * m["dice"] for num_examples, m in metrics) / total_examples
    return {"dice": weighted_dice}


def load_certificates():
    ca_path = os.path.join(CERTS_DIR, "ca.crt")
    cert_path = os.path.join(CERTS_DIR, "server.pem")
    key_path = os.path.join(CERTS_DIR, "server.key")

    for path in (ca_path, cert_path, key_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing certificate file: {path}\n"
                "Run `python generate_certs.py` first to create the certificates."
            )

    with open(ca_path, "rb") as f:
        ca_cert = f.read()
    with open(cert_path, "rb") as f:
        server_cert = f.read()
    with open(key_path, "rb") as f:
        server_key = f.read()

    return (ca_cert, server_cert, server_key)


def main():
    certificates = load_certificates()
    print("Loaded TLS certificates -- server will require encrypted connections.")

    initial_model = build_model()
    initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"Starting TLS-secured federated server -- waiting for {MIN_CLIENTS} hospital(s)...")
    print("Run `python fed_client_tls.py --client_id N` (N=1,2,3) in 3 separate terminals now.")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        certificates=certificates,
    )


if __name__ == "__main__":
    main()
