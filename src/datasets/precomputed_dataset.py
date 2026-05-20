import glob
import os

import torch
from torch.utils.data import Dataset


class PrecomputedDataset(Dataset):
    """
    Dataset for precomputed image-latent pairs.

    Each sample contains:
        image tensor:  [1, H, W], normalized to [-1, 1]
        latent tensor: [4, h, w], scaled latent representation

    The image and latent files must share the same filename.
    """

    def __init__(self, latent_dir, img_dir):
        self.img_dir = img_dir
        self.latent_dir = latent_dir
        self.files = sorted(glob.glob(os.path.join(img_dir, "*.pt")))

        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {img_dir}")

        print(f"Dataset loaded: {len(self.files)} pairs.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        file_id = os.path.basename(img_path)
        latent_path = os.path.join(self.latent_dir, file_id)

        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Latent file not found: {latent_path}")

        img_tensor = torch.load(img_path, map_location="cpu", weights_only=True)
        z_gt = torch.load(latent_path, map_location="cpu", weights_only=True)

        return img_tensor, z_gt
