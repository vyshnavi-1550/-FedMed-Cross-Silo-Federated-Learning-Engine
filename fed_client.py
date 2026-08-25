"""
Week 2, Day 1-2 — Real Federated Client

Unlike Week 1's dummy client (random weights), this client:
1. Loads its OWN partition of the data (simulating one hospital's private data)
2. Receives the global U-Net weights from the server
3. Actually trains locally for a few epochs on ITS OWN data only
4. Sends the updated weights back (never the data itself)

Run server.py FIRST, then run this 3 times with different --client_id
values, in 3 separate terminals, to simulate 3 hospitals:

    python fed_client.py --client_id 1
    python fed_client.py --client_id 2
    python fed_client.py --client_id 3

Needs baseline/data_pipeline.py and baseline/dataset.py on the Python path
-- run this from the project root, or adjust sys.path below.
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
from data_pipeline import get_train_transforms, get_federated_datalist  # noqa: E402
from fed_utils import build_model, get_model_parameters, set_model_parameters, DEVICE

LOCAL_EPOCHS = 2  # epochs of LOCAL training per federated round (keep small -- this runs every round)
LEARNING_RATE = 1e-3
NUM_CLIENTS = 3  # total number of simulated hospitals


class FedMedClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = build_model()

        # Each hospital only loads ITS OWN slice of the REAL data -- this is
        # the simulated data silo, built from the same real patient list
        # your baseline training uses (not the separate synthetic dataset.json).
        datalist = get_federated_datalist(client_id=client_id, num_clients=NUM_CLIENTS)
        print(f"[Hospital {client_id}] Local dataset size: {len(datalist)} samples")

        dataset = Dataset(data=datalist, transform=get_train_transforms())
        self.train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

        self.loss_function = DiceLoss(sigmoid=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.post_label = Compose([AsDiscrete()])

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        """Called by the server each round: load global weights, train locally, return updated weights."""
        set_model_parameters(self.model, parameters)
        self.model.train()

        print(f"[Hospital {self.client_id}] Starting local training ({LOCAL_EPOCHS} epochs)...")
        for epoch in range(LOCAL_EPOCHS):
            epoch_loss = 0.0
            steps = 0
            for batch in self.train_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                steps += 1
            print(f"[Hospital {self.client_id}]   local epoch {epoch + 1}/{LOCAL_EPOCHS} — loss: {epoch_loss / max(steps, 1):.4f}")

        num_examples = len(self.train_loader.dataset)
        return get_model_parameters(self.model), num_examples, {}

    def evaluate(self, parameters, config):
        """Called by the server each round: load global weights, evaluate on local data."""
        set_model_parameters(self.model, parameters)
        self.model.eval()

        self.dice_metric.reset()
        total_loss = 0.0
        steps = 0
        with torch.no_grad():
            for batch in self.train_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                total_loss += loss.item()
                steps += 1

                preds = [self.post_pred(i) for i in decollate_batch(outputs)]
                labels_list = [self.post_label(i) for i in decollate_batch(labels)]
                self.dice_metric(y_pred=preds, y=labels_list)

        mean_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        avg_loss = total_loss / max(steps, 1)

        print(f"[Hospital {self.client_id}] Local eval — loss: {avg_loss:.4f} — Dice: {mean_dice:.4f}")
        num_examples = len(self.train_loader.dataset)
        return avg_loss, num_examples, {"dice": mean_dice}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True, help="Which mock hospital (1, 2, or 3)")
    args = parser.parse_args()

    print(f"Starting REAL Hospital {args.client_id} client (training on local partition)...")
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FedMedClient(client_id=args.client_id),
    )


if __name__ == "__main__":
    main()
