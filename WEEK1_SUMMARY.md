# Week 1 Checkpoint — FedMed Project

## Summary

Week 1 focused on two parallel tracks: (1) building a centralized ML baseline
pipeline, and (2) proving the federated learning networking skeleton works
before adding real distributed training logic in Week 2.

## Track A: Centralized Baseline (PPML Engineering)

- Built a MONAI-based data pipeline: loading, normalization, random cropping,
  and augmentation for 3D brain MRI volumes (`data_pipeline.py`).
- Built and trained a 3D U-Net (`train_baseline.py`) using Dice loss and the
  Dice metric for evaluation.
- **Status:** pipeline fully functional end-to-end. Training was run on a
  small **synthetic dataset** (6 fake volumes) to unblock development while
  the real BraTS/Decathlon dataset (~7GB) downloads in the background —
  the real download was slow on this network and is not yet complete.
- **Action item for Week 2:** once the real dataset finishes downloading,
  set `USE_SYNTHETIC = False` and rerun `train_baseline.py` to get the
  actual reportable baseline Dice score.

## Track B: Distributed Systems (Node Scaffolding)

- Set up a Flower server (`server.py`) using the FedAvg strategy.
- Built a parameterized Flower client (`client.py`) simulating a hospital
  node, using dummy weights (real model integration comes in Week 2).
- Successfully ran 3 simultaneous client instances (`--client_id 1/2/3`)
  representing 3 mock hospital nodes, all connecting to one central server.
- **Result:** 3 rounds completed, 3/3 clients participated in every round,
  **0 failures**.

## Known issues resolved this week

- MONAI's `DecathlonDataset` requires `root_dir` to already exist — fixed
  by adding `os.makedirs(..., exist_ok=True)`.
- Flower's optional TensorBoard integration pulled in an incompatible
  `h5py`/NumPy combination on Windows — fixed via `pip install --upgrade h5py`.
- Real dataset download speed was a bottleneck — worked around using a
  synthetic data generator (`synthetic_data.py`) to keep development moving.

## Not yet started (Week 2 scope)

- Real Federated Training Loop: server broadcasting the U-Net to nodes,
  local training, FedAvg aggregation.
- Secure Communication: gRPC with TLS certificates.
- Privacy & Encryption: TenSEAL homomorphic encryption for weight updates.
- Training Dashboard (React/Recharts).

## Files in repo

| File | Purpose |
|---|---|
| `dataset.py` | Day 2 — dataset loading & visual sanity check |
| `synthetic_data.py` | Generates fake data for pipeline testing |
| `data_pipeline.py` | Day 3 — training-time transforms, cropping, DataLoader |
| `train_baseline.py` | Day 4 — U-Net model, training loop, Dice metric |
| `server.py` | Day 5/6 — Flower FedAvg server |
| `client.py` | Day 5/6 — Flower client, parameterized for multiple nodes |
