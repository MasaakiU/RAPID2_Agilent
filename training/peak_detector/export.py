"""Write a checkpoint's weights as a .npz of numpy arrays, by their names in the state_dict, and its config in it as
"config" (JSON): what the app reads, with no torch.

    python export.py runs/transformer/best.pt transformer.npz
"""

import json
import sys

import numpy as np
import torch


def export(checkpoint_path: str, npz_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arrays = {name: tensor.float().numpy() if tensor.is_floating_point() else tensor.numpy()
              for name, tensor in checkpoint["model"].items()}
    np.savez_compressed(npz_path, config=json.dumps(checkpoint["config"] | {"step": checkpoint["step"]}), **arrays)


if __name__ == "__main__":
    export(*sys.argv[1:3])
