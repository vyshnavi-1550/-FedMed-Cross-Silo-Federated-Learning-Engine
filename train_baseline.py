"""
Day 4 — Train the Centralized Baseline U-Net

Builds a MONAI 3D U-Net, trains it on the data pipeline from Day 3, and
tracks the Dice score (the standard accuracy metric for segmentation).

This is your BASELINE -- the number the federated model (built in later
weeks) needs to approach without ever pooling raw patient data centrally.

Run:
    python train_baseline.py

Notes:
- USE_SYNTHETIC=True (inherited from data_pipeline.py) means this trains
  on FAKE data right now. The model WILL "learn" to find the fake blobs
  (that's a real, if trivial, segmentation task) but the resulting Dice
  score is NOT your real baseline. Re-run with real data once the
  Decathlon download finishes to get your actual reportable baseline.
- Kept deliberately small/fast (few epochs, small model) so it runs
  quickly on CPU while you're developing. Scale up once confirmed working.
"""

import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import Compose, Activations, AsDiscrete

from data_pipeline import get_train_dataloader, PATCH_SIZE

NUM_EPOCHS = 20        # small for fast iteration; increase for a real baseline run
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    model = UNet(
        spatial_dims=3,
        in_channels=4,      # 4 MRI modalities (FLAIR, T1, T1c, T2)
        out_channels=1,     # binary: tumor vs. background (real BraTS has 3 sub-regions -- start binary, extend later)
        channels=(16, 32, 64, 128),  # smaller than typical (usually 16,32,64,128,256) for faster CPU training
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(DEVICE)
    return model


def main():
    print(f"Using device: {DEVICE}")

    train_loader = get_train_dataloader()
    model = build_model()

    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete()])

    best_dice = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        step = 0

        for batch in train_loader:
            step += 1
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= step

        # Evaluate Dice on the same data (fine for a quick dev loop; use a
        # held-out validation split for your real baseline run)
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch in train_loader:
                images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = model(images)
                outputs = [post_pred(i) for i in decollate_batch(outputs)]
                labels_list = [post_label(i) for i in decollate_batch(labels)]
                dice_metric(y_pred=outputs, y=labels_list)

        mean_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        print(f"Epoch {epoch}/{NUM_EPOCHS} — loss: {epoch_loss:.4f} — Dice: {mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), "best_baseline_model.pth")

    print(f"\nTraining complete. Best Dice score: {best_dice:.4f}")
    print("Saved best model weights to best_baseline_model.pth")
    print("\n*** Remember: if USE_SYNTHETIC=True, this Dice score is from fake data ***")
    print("*** and is NOT your real baseline. Switch to real data before reporting. ***")


if __name__ == "__main__":
    main()
