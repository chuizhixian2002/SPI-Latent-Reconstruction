import os
import random

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


def _to_img(tensor):
    return tensor.squeeze(0).detach().cpu().numpy()


def save_decoder_visualization(decoder, dataset, epoch, save_dir, scale_factor, device, num_samples=3):
    """
    Visualize decoder reconstruction: ground truth image vs reconstructed image.
    """

    decoder.eval()
    os.makedirs(save_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    z_list = []
    gt_list = []

    for idx in indices:
        img, z = dataset[idx]
        z_list.append(z)
        gt_list.append(img)

    z_batch = torch.stack(z_list).to(device)
    gt_batch = torch.stack(gt_list).to(device)

    with torch.no_grad():
        pred_img = decoder(z_batch / scale_factor)
        pred_vis = pred_img.clamp(0, 1)
        gt_vis = (gt_batch / 2 + 0.5).clamp(0, 1)

    fig, axs = plt.subplots(len(indices), 2, figsize=(6, 3 * len(indices)))
    if len(indices) == 1:
        axs = [axs]

    plt.suptitle(f"Decoder Epoch {epoch}", fontsize=16)

    for i in range(len(indices)):
        axs[i][0].imshow(_to_img(gt_vis[i]), cmap="gray")
        axs[i][0].set_title("Ground Truth")
        axs[i][0].axis("off")

        axs[i][1].imshow(_to_img(pred_vis[i]), cmap="gray")
        axs[i][1].set_title("Decoder Recon")
        axs[i][1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vis_decoder_epoch_{epoch:03d}.png"))
    plt.close()


def save_guided_encoder_visualization(encoder, decoder, dataset, epoch, save_dir, cfg, num_samples=3):
    """
    Visualize guided encoder reconstruction.
    """

    encoder.eval()
    decoder.eval()
    os.makedirs(save_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    img_list = []
    z_gt_list = []

    for idx in indices:
        x, z = dataset[idx]
        img_list.append(x)
        z_gt_list.append(z)

    batch_x = torch.stack(img_list).to(cfg.device)
    batch_z_gt = torch.stack(z_gt_list).to(cfg.device)

    with torch.no_grad():
        z_est, _, _ = encoder(batch_x)
        pred_vis = decoder(z_est / cfg.scale_factor).clamp(0, 1)
        pseudo_gt = decoder(batch_z_gt / cfg.scale_factor).clamp(0, 1)

    fig, axs = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    if len(indices) == 1:
        axs = [axs]

    plt.suptitle(f"Guided Encoder Epoch {epoch}", fontsize=16)

    for i in range(len(indices)):
        gt_vis = (batch_x[i] * 0.5 + 0.5).clamp(0, 1)

        axs[i][0].imshow(_to_img(gt_vis), cmap="gray")
        axs[i][0].set_title("Input Image")
        axs[i][0].axis("off")

        axs[i][1].imshow(_to_img(pred_vis[i]), cmap="gray")
        axs[i][1].set_title("Student Recon")
        axs[i][1].axis("off")

        axs[i][2].imshow(_to_img(pseudo_gt[i]), cmap="gray")
        axs[i][2].set_title("Decoder Target")
        axs[i][2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vis_guided_epoch_{epoch:03d}.png"))
    plt.close()
