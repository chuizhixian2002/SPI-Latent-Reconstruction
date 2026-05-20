import csv
import os

import numpy as np
import torch
from PIL import Image


def save_metrics(history, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, filename)
    keys = list(history.keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch"] + keys)

        for i in range(len(history[keys[0]])):
            row = [i + 1] + [history[k][i] for k in keys]
            writer.writerow(row)


def load_gray_image(img_path, img_size, device):
    """
    Load image, convert to grayscale, resize, and normalize to [-1, 1].
    """

    img = Image.open(img_path).convert("L")
    img = img.resize((img_size, img_size), Image.BICUBIC)

    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor * 2.0 - 1.0

    return img_tensor.to(device)


def save_gray_tensor(img_tensor, save_path):
    """
    Save a single-channel image tensor in [0, 1] as an 8-bit grayscale image.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img_np = img_tensor.squeeze().detach().cpu().numpy()
    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(img_uint8, mode="L").save(save_path)
