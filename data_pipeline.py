"""
Day 3 — Training-Time Data Pipeline

Builds the transform pipeline actually used for TRAINING (as opposed to
Day 2's viewing-only pipeline). The key addition is RandCropByPosNegLabeld,
which pulls random smaller 3D patches out of each volume -- this (a) makes
training feasible memory-wise, and (b) gives the model varied views of the
data across epochs instead of always seeing the exact same full volume.

Also wraps everything in a MONAI Dataset + PyTorch DataLoader for batching.

Run:
    python data_pipeline.py
This will load a batch and print its shape, to confirm the pipeline works
before we plug it into the actual training loop (Day 4).
"""

import os
import json
import torch
from monai.data import Dataset, DataLoader
from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)

DATA_ROOT = "./data"
USE_SYNTHETIC = True  # keep True until the real download finishes (see dataset.py notes)
PATCH_SIZE = (32, 32, 16)  # smaller than full volume -- adjust down further if synthetic volumes are small
BATCH_SIZE = 2


def get_train_transforms():
    """Training pipeline: load -> normalize -> random-crop -> random augment."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys="image", a_min=0, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            # Randomly crop a patch, weighted toward including tumor voxels (pos) sometimes
            # and pure background (neg) other times -- keeps the model seeing both classes.
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            # Cheap, safe augmentations for medical volumes
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_synthetic_datalist():
    task_dir = os.path.join(DATA_ROOT, "Task01_BrainTumour")
    json_path = os.path.join(task_dir, "dataset.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError("No synthetic data found. Run `python synthetic_data.py` first.")
    with open(json_path) as f:
        meta = json.load(f)
    return [
        {
            "image": os.path.join(task_dir, entry["image"].lstrip("./")),
            "label": os.path.join(task_dir, entry["label"].lstrip("./")),
        }
        for entry in meta["training"]
    ]


def get_train_dataloader():
    """Builds and returns the training DataLoader. Import and reuse this in Day 4's training script."""
    transforms = get_train_transforms()

    if USE_SYNTHETIC:
        datalist = get_synthetic_datalist()
        dataset = Dataset(data=datalist, transform=transforms)
    else:
        dataset = DecathlonDataset(
            root_dir=DATA_ROOT,
            task="Task01_BrainTumour",
            section="training",
            transform=transforms,
            download=True,
            cache_rate=0.0,
            num_workers=2,
        )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return loader


def main():
    loader = get_train_dataloader()
    print(f"DataLoader ready. Fetching one batch (batch_size={BATCH_SIZE})...")

    batch = next(iter(loader))
    print(f"Batch image shape: {batch['image'].shape}")  # (B, C, H, W, D)
    print(f"Batch label shape: {batch['label'].shape}")  # (B, C, H, W, D)
    print(f"Image value range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    print(f"Label unique values: {torch.unique(batch['label'])}")
    print("\nPipeline check passed -- ready for Day 4 (model + training loop).")


if __name__ == "__main__":
    main()
