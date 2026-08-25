"""
Mid-Project Review -- Node Resilience

Same as fed_server_tls.py (real training + TLS), but now the server can
tolerate a hospital dropping offline mid-training: a round proceeds as
long as at least MIN_CLIENTS (< TOTAL_CLIENTS) are available, instead of
requiring all 3 to be present.

Key change: min_fit_clients / min_evaluate_clients / min_available_clients
are now set to a number LESS than the total number of hospitals, and
fraction_fit / fraction_evaluate control what fraction of AVAILABLE
clients get sampled each round.

Run this FIRST, then run fed_client_tls.py in 3 terminals as usual. To
actually TEST resilience: once training has started, close one of the
3 client terminals (Ctrl+C) partway through -- the remaining 2 hospitals
should keep completing rounds instead of the whole server stalling/crashing.

Run:
    python fed_server_resilient.py
"""

import sys
import os
import flwr as fl
from flwr.common import ndarrays_to_parameters

sys.path.append(os.path.dirname(__file__))
from fed_utils import build_model, get_model_parameters

TOTAL_CLIENTS = 3
MIN_CLIENTS = 2          # <-- resilience: only require 2 out of 3 to proceed
NUM_ROUNDS = 5
CERTS_DIR = os.path.join(os.path.dirname(__file__), "certs")


def weighted_average(metrics):
    """Aggregates the 'dice' metric across whichever clients actually
    reported results this round -- works fine even if fewer than
    TOTAL_CLIENTS participated."""
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
                "Run `python generate_certs.py` first."
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

    initial_model = build_model()
    initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

    strategy = fl.server.strategy.FedAvg(
        # Resilience settings: proceed with as few as MIN_CLIENTS, not all TOTAL_CLIENTS
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        fraction_fit=1.0,       # sample 100% of AVAILABLE (connected) clients each round
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"Starting RESILIENT federated server.")
    print(f"Will proceed with as few as {MIN_CLIENTS} out of {TOTAL_CLIENTS} hospitals connected.")
    print("Run `python fed_client_tls.py --client_id N` (N=1,2,3) in 3 separate terminals.")
    print("Try closing one client terminal mid-training (Ctrl+C) to test resilience.")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        certificates=certificates,
    )


if __name__ == "__main__":
    main()
