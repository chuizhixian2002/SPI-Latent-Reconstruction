import os
import sys

# Allow running the script directly from the repository root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.datasets.precomputed_dataset import PrecomputedDataset
from src.losses.losses import CharbonnierLoss
from src.metrics.image_metrics import calculate_psnr, calculate_ssim
from src.models import SpiDecoder
from src.utils.config import get_device, load_yaml_config, parse_config_arg
from src.utils.io_utils import save_metrics
from src.utils.seed import set_seed
from src.utils.visualization import save_decoder_visualization


def main():
    args = parse_config_arg("configs/decoder_config.yaml")
    cfg = load_yaml_config(args.config)

    device = get_device(cfg.device)
    set_seed(cfg.seed)

    save_dir = cfg.output.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"Start decoder training | Batch: {cfg.train.batch_size} | Device: {device}")

    decoder = SpiDecoder(
        latent_channels=cfg.model.latent_channels,
        out_channels=cfg.model.out_channels,
        base_dim=cfg.model.base_dim,
        depth=cfg.model.depth,
    ).to(device)

    full_dataset = PrecomputedDataset(
        latent_dir=cfg.data.latent_dir,
        img_dir=cfg.data.img_dir,
    )

    val_size = int(len(full_dataset) * cfg.train.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset_subset = torch.utils.data.Subset(
        val_dataset,
        range(min(cfg.train.val_subset_size, len(val_dataset))),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers_train,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset_subset,
        batch_size=16,
        shuffle=False,
        num_workers=cfg.train.num_workers_val,
        pin_memory=True,
    )

    optimizer = optim.AdamW(
        decoder.parameters(),
        lr=cfg.train.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs,
        eta_min=1e-6,
    )

    use_amp = bool(cfg.train.use_amp) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    criterion = CharbonnierLoss().to(device)

    best_psnr = 0.0
    history = {
        "loss": [],
        "val_psnr": [],
        "val_ssim": [],
    }

    print(f"Training samples: {train_size}")

    for epoch in range(cfg.train.epochs):
        decoder.train()
        acc_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{cfg.train.epochs}", leave=False)

        for gt_img, z_gt in pbar:
            gt_img = gt_img.to(device, non_blocking=True)
            z_gt = z_gt.to(device, non_blocking=True)

            target = (gt_img / 2 + 0.5).clamp(0, 1)
            input_z = z_gt / cfg.train.scale_factor

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_img = decoder(input_z)
                loss = criterion(pred_img, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        decoder.eval()
        psnr_acc = 0.0
        ssim_acc = 0.0

        with torch.no_grad():
            for gt_img, z_gt in val_loader:
                gt_img = gt_img.to(device)
                z_gt = z_gt.to(device)

                target = (gt_img / 2 + 0.5).clamp(0, 1)
                input_z = z_gt / cfg.train.scale_factor

                pred_img = decoder(input_z).clamp(0, 1)

                psnr_acc += calculate_psnr(pred_img, target)
                ssim_acc += calculate_ssim(pred_img, target)

        avg_loss = acc_loss / len(train_loader)
        avg_psnr = psnr_acc / len(val_loader)
        avg_ssim = ssim_acc / len(val_loader)

        history["loss"].append(avg_loss)
        history["val_psnr"].append(avg_psnr)
        history["val_ssim"].append(avg_ssim)

        save_metrics(history, save_dir, cfg.output.metrics_file)

        print(
            f"Ep {epoch + 1:03d} | "
            f"Loss: {avg_loss:.4f} | "
            f"PSNR: {avg_psnr:.2f} | "
            f"SSIM: {avg_ssim:.4f}"
        )

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(decoder.state_dict(), os.path.join(save_dir, "best_decoder.pth"))

        if cfg.output.save_vis_every > 0 and (epoch + 1) % cfg.output.save_vis_every == 0:
            save_decoder_visualization(
                decoder=decoder,
                dataset=val_dataset_subset,
                epoch=epoch + 1,
                save_dir=save_dir,
                scale_factor=cfg.train.scale_factor,
                device=device,
            )

    torch.save(decoder.state_dict(), os.path.join(save_dir, "last_decoder.pth"))


if __name__ == "__main__":
    main()
