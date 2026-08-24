"""
Day 2 — Get & Explore the Dataset

Downloads the Medical Segmentation Decathlon "Task01_BrainTumour" dataset
(a clean, pre-packaged stand-in for BraTS) and visualizes a few slices to
confirm the MRI volumes and tumor masks look correct.

Run:
    python dataset.py
"""

import os
import json
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    ToTensord,
)

DATA_ROOT = "./data"  # dataset will be downloaded here (~7GB, be patient)
USE_SYNTHETIC = True  # set True temporarily if the real download is too slow (see synthetic_data.py)


def get_transforms():
    """Minimal transform pipeline just for viewing raw data (Day 2).
    A richer training-time pipeline (with random cropping) comes in Day 3."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys="image", a_min=0, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_synthetic_datalist():
    """Load the file list from the synthetic dataset.json (run synthetic_data.py first)."""
    task_dir = os.path.join(DATA_ROOT, "Task01_BrainTumour")
    json_path = os.path.join(task_dir, "dataset.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "No synthetic data found. Run `python synthetic_data.py` first to generate it."
        )
    with open(json_path) as f:
        meta = json.load(f)
    # dataset.json stores relative paths like "./imagesTr/xxx.nii.gz" -> resolve to absolute
    datalist = []
    for entry in meta["training"]:
        datalist.append(
            {
                "image": os.path.join(task_dir, entry["image"].lstrip("./")),
                "label": os.path.join(task_dir, entry["label"].lstrip("./")),
            }
        )
    return datalist


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)  # MONAI requires this folder to already exist

    if USE_SYNTHETIC:
        print("USE_SYNTHETIC=True — loading fake local data (for pipeline testing only).")
        datalist = get_synthetic_datalist()
        dataset = Dataset(data=datalist, transform=get_transforms())
    else:
        print("Downloading / loading Task01_BrainTumour (this can take a while the first time)...")
        dataset = DecathlonDataset(
            root_dir=DATA_ROOT,
            task="Task01_BrainTumour",
            section="training",
            transform=get_transforms(),
            download=True,
            cache_rate=0.0,  # don't cache into RAM yet, we're just exploring
            num_workers=2,
        )

    print(f"Loaded {len(dataset)} training samples.")

    # Grab one sample and confirm shapes
    sample = dataset[0]
    image, label = sample["image"], sample["label"]
    print(f"Image shape: {image.shape}")  # (C, H, W, D) — 4 MRI modalities (FLAIR, T1, T1c, T2)
    print(f"Label shape: {label.shape}")  # (C, H, W, D) — tumor sub-region masks

    # Visualize the slice with the LARGEST tumor area (mid-slice can miss a
    # randomly-placed blob, especially after resampling changes dimensions)
    tumor_pixel_counts = label[0].sum(dim=(0, 1))  # sum over H,W for each depth slice
    best_slice = int(tumor_pixel_counts.argmax())
    print(f"Slice with most tumor pixels: {best_slice} (tumor pixel count: {int(tumor_pixel_counts[best_slice])})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image[0, :, :, best_slice], cmap="gray")
    axes[0].set_title("FLAIR MRI (tumor slice)")
    axes[0].axis("off")

    axes[1].imshow(image[0, :, :, best_slice], cmap="gray")
    axes[1].imshow(label[0, :, :, best_slice], cmap="hot", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Tumor Mask Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("sample_check.png", dpi=150)
    print("Saved a visual sanity check to sample_check.png — open it and confirm the tumor mask overlaps a bright region on the scan.")


if __name__ == "__main__":
    main()