"""
Synthetic Dataset Generator — for unblocking development while the real
BraTS/Decathlon download is slow.

This creates a handful of FAKE brain MRI volumes + tumor masks, saved as
.nii.gz files on disk, in the exact same folder structure MONAI's
DecathlonDataset expects. This lets you build and test your ENTIRE
pipeline (transforms, model, training loop) today, without waiting on
a multi-hour download.

IMPORTANT: this is only for testing your code runs correctly end-to-end.
Any accuracy/Dice numbers from training on this fake data are meaningless
-- you MUST swap back to real data before reporting your baseline Dice
score. The data here is random noise with a random blob "tumor", it has
no real medical structure to learn.

Run:
    python synthetic_data.py

This creates:
    ./data/Task01_BrainTumour/imagesTr/*.nii.gz   (fake MRI volumes)
    ./data/Task01_BrainTumour/labelsTr/*.nii.gz   (fake tumor masks)
    ./data/Task01_BrainTumour/dataset.json         (metadata MONAI needs)
"""

import os
import json
import numpy as np
import nibabel as nib

DATA_ROOT = "./data"
TASK_DIR = os.path.join(DATA_ROOT, "Task01_BrainTumour")
IMAGES_DIR = os.path.join(TASK_DIR, "imagesTr")
LABELS_DIR = os.path.join(TASK_DIR, "labelsTr")

N_SAMPLES = 6          # small number, just enough to run through the pipeline
VOLUME_SHAPE = (64, 64, 64)  # small volume, real BraTS is much bigger (240,240,155)
N_MODALITIES = 4        # real Task01_BrainTumour has 4 MRI modalities (FLAIR,T1,T1c,T2)


def make_fake_volume():
    """Random noise volume + a bright blob to fake a 'brain scan', per modality."""
    vol = np.random.normal(loc=200, scale=50, size=(*VOLUME_SHAPE, N_MODALITIES)).astype(np.float32)
    vol = np.clip(vol, 0, None)
    return vol


def make_fake_label():
    """Random small blob 'tumor' mask (values 0=background, 1=tumor)."""
    label = np.zeros(VOLUME_SHAPE, dtype=np.uint8)
    cx, cy, cz = [np.random.randint(15, s - 15) for s in VOLUME_SHAPE]
    r = np.random.randint(4, 10)
    xx, yy, zz = np.ogrid[: VOLUME_SHAPE[0], : VOLUME_SHAPE[1], : VOLUME_SHAPE[2]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= r ** 2
    label[mask] = 1
    return label


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    affine = np.eye(4)  # identity affine, fine for fake data
    file_entries = []

    for i in range(N_SAMPLES):
        image = make_fake_volume()
        label = make_fake_label()

        image_name = f"BRATS_fake_{i:03d}.nii.gz"
        label_name = f"BRATS_fake_{i:03d}.nii.gz"

        nib.save(nib.Nifti1Image(image, affine), os.path.join(IMAGES_DIR, image_name))
        nib.save(nib.Nifti1Image(label, affine), os.path.join(LABELS_DIR, label_name))

        file_entries.append(
            {"image": f"./imagesTr/{image_name}", "label": f"./labelsTr/{label_name}"}
        )

    # MONAI's DecathlonDataset expects a dataset.json describing splits
    dataset_json = {
        "name": "FakeBrainTumour",
        "description": "Synthetic placeholder data for pipeline testing only",
        "tensorImageSize": "4D",
        "modality": {"0": "FLAIR", "1": "T1w", "2": "T1gd", "3": "T2w"},
        "labels": {"0": "background", "1": "tumor"},
        "numTraining": N_SAMPLES,
        "numTest": 0,
        "training": file_entries,
        "test": [],
    }

    with open(os.path.join(TASK_DIR, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Created {N_SAMPLES} synthetic samples in {TASK_DIR}")
    print("You can now run dataset.py with USE_SYNTHETIC=True to test your pipeline immediately.")


if __name__ == "__main__":
    main()
