"""
Week 2 — Shared Federated Utilities

Helper functions used by both the Flower client and server for converting
between PyTorch model weights and the plain NumPy arrays Flower sends over
the network, plus a shared model-builder so client and server always agree
on architecture.
"""

import torch
from collections import OrderedDict
from monai.networks.nets import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    """Same architecture as train_baseline.py -- keep these in sync."""
    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(DEVICE)


def get_model_parameters(model):
    """PyTorch model -> list of NumPy arrays (what Flower sends over the network)."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_parameters(model, parameters):
    """List of NumPy arrays (received from Flower) -> load into a PyTorch model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
