# FedMed — Week 1 Starter

## Day 1 — Setup

```bash
# create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

Folder structure:
```
fedmed_week1/
├── requirements.txt
├── baseline/          <- centralized U-Net baseline (Days 2-4)
│   └── dataset.py      <- Day 2 script (run this next)
└── federated/          <- Flower server/client skeleton (Days 5-6)
```

Check your setup works:
```bash
python -c "import torch, monai, flwr; print('torch:', torch.__version__); print('monai:', monai.__version__); print('flower:', flwr.__version__)"
```

If you have a GPU, confirm PyTorch sees it:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
(CPU works fine too, just slower — the dataset used here is small enough to train a few epochs on CPU for a baseline check.)

## Day 2 — Dataset

```bash
cd baseline
python dataset.py
```

This will:
1. Download the Medical Segmentation Decathlon "Task01_BrainTumour" dataset (~7GB, first run only) — this is an easier-to-obtain stand-in for BraTS with the same kind of data (multi-modal brain MRI + tumor masks).
2. Print the shape of one sample (4 MRI modalities × 3D volume, and its tumor mask).
3. Save `sample_check.png` — open it and confirm the tumor mask overlay actually lines up with a bright/abnormal region in the scan. This is your sanity check before building anything else.

**If the download is too slow/large:** you can substitute a smaller Kaggle-hosted pre-processed BraTS subset — same idea, just point `LoadImaged` at your local files instead of using `DecathlonDataset`. Let me know if you want that swapped in.

## Next up (Day 3-4)

Once `dataset.py` runs cleanly and `sample_check.png` looks right, next is:
- Day 3: a proper training-time pipeline with random cropping (`RandCropByPosNegLabeld`) and a `DataLoader`.
- Day 4: the actual `UNet` model, `DiceLoss`, training loop, and your baseline Dice score.

Say the word and I'll generate those next.
