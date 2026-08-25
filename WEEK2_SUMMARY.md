# Week 2 Checkpoint — FedMed Project

## Summary

Week 2 turned Week 1's networking skeleton into a genuinely working
privacy-preserving federated learning system: real local training on
partitioned data, real weight aggregation, and an encrypted communication
channel between the central server and each hospital node.

## Day 1-2: Real Federated Training Loop

- **Data partitioning** (`dataset.py`): each simulated hospital now loads
  only its OWN slice of the dataset (`get_synthetic_datalist(client_id=...)`)
  — no hospital ever sees another's data, simulating true data silos.
- **Federated client** (`fed_client.py`): each hospital's Flower client now
  actually loads the global U-Net weights, trains locally for real epochs
  on its own data partition, and returns genuinely updated weights (not
  dummy random noise like Week 1).
- **Federated server** (`fed_server.py`): initializes training from the
  model's real starting weights (not per-client random weights), runs
  FedAvg aggregation across 3 hospitals, and tracks the real (weighted)
  Dice score across rounds.
- **Shared utilities** (`fed_utils.py`): conversion functions between
  PyTorch model weights and the NumPy arrays Flower sends over the network.

**Result:** 5 full federated rounds completed across 3 hospitals, with real
per-hospital training loss decreasing within each round and real (if noisy,
given tiny synthetic data) Dice scores aggregated centrally. Zero data ever
left any individual hospital's simulated node — only model weights traveled.

## Day 3: Secure Communication (TLS)

- **Certificate generation** (`generate_certs.py`): creates a local
  Certificate Authority (CA) and a server certificate signed by it, using
  OpenSSL. Handles a Windows/Anaconda-specific issue where OpenSSL couldn't
  locate its own config file, by auto-detecting and setting `OPENSSL_CONF`.
- **TLS server** (`fed_server_tls.py`): same real FedAvg logic as Day 1-2,
  now requires clients to connect over an encrypted gRPC/TLS channel.
- **TLS client** (`fed_client_tls.py`): verifies the server's certificate
  against the shared CA before sending any weight updates — protects
  against a hospital accidentally connecting to (and leaking updates to)
  an impostor server.

**Result:** All 3 hospitals successfully completed 5 full federated rounds
over the TLS-encrypted connection, with identical training behavior to the
unencrypted version — confirming TLS is transparent to the ML logic and
adds security without breaking functionality.

## Important caveat (carried over from Week 1)

All training this week was still on the **synthetic dataset** — the real
BraTS/Decathlon download has not completed (slow network). The mechanism
(partitioning, local training, aggregation, encryption) is fully proven;
the actual Dice score numbers are not yet meaningful and should be
re-validated once real data is available.

## Not yet started (later weeks' scope, per original plan)

- Privacy & Encryption: TenSEAL homomorphic encryption of weight updates
  (currently weights travel in plaintext *within* the TLS tunnel — TLS
  protects the network transport, but the server still sees raw weights;
  homomorphic encryption would let the server aggregate without ever
  seeing decrypted weights at all).
- Training Dashboard (React/Recharts) for visualizing convergence.
- Node resilience (surviving a hospital dropping mid-round).

## Files added this week

| File | Purpose |
|---|---|
| `fed_utils.py` | PyTorch <-> NumPy weight conversion helpers |
| `fed_client.py` | Real federated client (local training, no TLS) |
| `fed_server.py` | Real federated server (FedAvg, no TLS) |
| `generate_certs.py` | Creates local CA + server TLS certificates |
| `fed_server_tls.py` | TLS-secured federated server |
| `fed_client_tls.py` | TLS-secured federated client |
| `dataset.py` (updated) | Added `client_id` partitioning support |
