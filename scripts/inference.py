import argparse
import os
import sys

# Allow running the script directly from the repository root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.models import SpiDecoder, SpiStudent_ImageDomain
from src.utils.io_utils import load_gray_image, save_gray_tensor


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--encoder_ckpt", type=str, required=True, help="Path to trained SPI encoder checkpoint.")
    parser.add_argument("--decoder_ckpt", type=str, required=True, help="Path to trained decoder checkpoint.")
    parser.add_argument("--output", type=str, default="results/inference/reconstruction.png")

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--sampling_ratio", type=float, default=0.05)
    parser.add_argument("--scale_factor", type=float, default=0.18215)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = SpiStudent_ImageDomain(
        img_size=args.img_size,
        sampling_ratio=args.sampling_ratio,
    ).to(device)

    decoder = SpiDecoder(
        latent_channels=4,
        out_channels=1,
    ).to(device)

    encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device), strict=False)
    decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device), strict=False)

    encoder.eval()
    decoder.eval()

    img = load_gray_image(args.image, args.img_size, device)

    with torch.no_grad():
        z_est, _, _ = encoder(img)
        recon = decoder(z_est / args.scale_factor).clamp(0, 1)

    save_gray_tensor(recon, args.output)
    print(f"Saved reconstruction to: {args.output}")


if __name__ == "__main__":
    main()
