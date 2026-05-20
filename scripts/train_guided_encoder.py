import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.datasets.precomputed_dataset import PrecomputedDataset
from src.losses.losses import SpiDistillationLoss
from src.metrics.image_metrics import calculate_psnr, calculate_ssim
from src.models import SpiDecoder, SpiStudent_ImageDomain
from src.utils.config import get_device, load_yaml_config, parse_config_arg
from src.utils.io_utils import load_gray_image, save_metrics
from src.utils.seed import set_seed


def collect_external_images(image_dir):
    patterns = ["*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    return sorted(image_paths)


def test_and_save(encoder, decoder, img_paths, img_size, scale_factor, device, save_dir, epoch):
    if not img_paths:
        print("Warning: no external evaluation images found. Skip external evaluation.")
        return None, None

    epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch:03d}", "external_eval")
    os.makedirs(epoch_save_dir, exist_ok=True)

    psnr_list = []
    ssim_list = []

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for img_path in img_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            img_in = load_gray_image(img_path, img_size, device)

            z_est, _, _ = encoder(img_in)
            recon = decoder(z_est / scale_factor).clamp(0, 1)
            target = (img_in / 2 + 0.5).clamp(0, 1)

            pred_np = (recon.squeeze().cpu().numpy() * 255.0).astype(np.float32)
            target_np = (target.squeeze().cpu().numpy() * 255.0).astype(np.float32)

            psnr = ski_psnr(target_np, pred_np, data_range=255)
            ssim = ski_ssim(target_np, pred_np, data_range=255)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            save_name = f"{stem}_psnr{psnr:.2f}_ssim{ssim:.4f}.png"
            save_path = os.path.join(epoch_save_dir, save_name)
            Image.fromarray(np.clip(pred_np, 0, 255).astype(np.uint8), mode="L").save(save_path)

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def build_optimizer(model, lr):
    if not hasattr(model, "spi_layer"):
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    spi_params = list(model.spi_layer.parameters())
    base_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("spi_layer")
    ]

    return optim.AdamW(
        [
            {"params": spi_params, "lr": lr},
            {"params": base_params, "lr": lr * 0.1},
        ],
        weight_decay=1e-4,
    )


def main():
    args = parse_config_arg("configs/guided_encoder_config.yaml")
    cfg = load_yaml_config(args.config)

    device = get_device(cfg.device)
    set_seed(cfg.seed)

    save_dir = cfg.output.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"Start decoder-guided SPI encoder training | Device: {device}")

    student_encoder = SpiStudent_ImageDomain(
        img_size=cfg.model.img_size,
        sampling_ratio=cfg.model.sample_ratio,
    ).to(device)

    fixed_decoder = SpiDecoder(
        latent_channels=cfg.model.latent_channels,
        out_channels=cfg.model.out_channels,
        base_dim=cfg.model.decoder_base_dim,
        depth=cfg.model.decoder_depth,
    ).to(device)

    print(f"Loading decoder from: {cfg.checkpoints.decoder_ckpt}")
    fixed_decoder.load_state_dict(torch.load(cfg.checkpoints.decoder_ckpt, map_location=device))
    fixed_decoder.eval()
    for param in fixed_decoder.parameters():
        param.requires_grad = False

    if cfg.checkpoints.pretrain_path and os.path.exists(cfg.checkpoints.pretrain_path):
        print(f"Loading pretrained encoder from: {cfg.checkpoints.pretrain_path}")
        student_encoder.load_state_dict(
            torch.load(cfg.checkpoints.pretrain_path, map_location=device),
            strict=False,
        )
    else:
        print("No pretrained encoder found. Train from scratch.")

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
        batch_size=32,
        shuffle=False,
        num_workers=cfg.train.num_workers_val,
        pin_memory=True,
    )

    optimizer = build_optimizer(student_encoder, cfg.train.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,
        T_mult=1,
        eta_min=1e-7,
    )

    use_amp = bool(cfg.train.use_amp) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    criterion = SpiDistillationLoss(
        w_lat=cfg.loss.w_lat,
        w_pix=cfg.loss.w_pix,
        w_grad=cfg.loss.w_grad,
    ).to(device)

    total_params = sum(p.numel() for p in student_encoder.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")

    best_psnr = 0.0
    history = {"loss_total": [], "val_psnr": [], "val_ssim": []}
    external_images = collect_external_images(cfg.data.external_eval_dir)

    for epoch in range(cfg.train.epochs):
        student_encoder.train()
        acc_total = 0.0

        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{cfg.train.epochs}", leave=False)
        for img, z_gt in pbar:
            img = img.to(device, non_blocking=True)
            z_gt = z_gt.to(device, non_blocking=True)

            with torch.no_grad():
                x_teacher_dec = fixed_decoder(z_gt / cfg.train.scale_factor).detach()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                z_est, _, _ = student_encoder(img)
                x_student_dec = fixed_decoder((z_est / cfg.train.scale_factor).float())
                loss, loss_dict = criterion(
                    z_est.float(),
                    z_gt.float(),
                    x_student_dec,
                    x_teacher_dec,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc_total += loss.item()
            pbar.set_postfix(pixel=f"{loss_dict['pixel'].item():.4f}")

        scheduler.step()

        student_encoder.eval()
        psnr_acc = 0.0
        ssim_acc = 0.0
        with torch.no_grad():
            for img, _ in val_loader:
                img = img.to(device)
                z_est, _, _ = student_encoder(img)
                recon = fixed_decoder(z_est / cfg.train.scale_factor).clamp(0, 1)
                target = (img * 0.5 + 0.5).clamp(0, 1)

                psnr_acc += calculate_psnr(recon, target)
                ssim_acc += calculate_ssim(recon, target)

        avg_loss = acc_total / len(train_loader)
        avg_psnr = psnr_acc / len(val_loader)
        avg_ssim = ssim_acc / len(val_loader)

        history["loss_total"].append(avg_loss)
        history["val_psnr"].append(avg_psnr)
        history["val_ssim"].append(avg_ssim)
        save_metrics(history, save_dir, cfg.output.metrics_file)

        print(
            f"Ep {epoch + 1:03d} | "
            f"Loss: {avg_loss:.4f} | "
            f"PSNR: {avg_psnr:.2f} | "
            f"SSIM: {avg_ssim:.4f}"
        )

        if cfg.output.eval_external_each_epoch:
            ext_psnr, ext_ssim = test_and_save(
                encoder=student_encoder,
                decoder=fixed_decoder,
                img_paths=external_images,
                img_size=cfg.model.img_size,
                scale_factor=cfg.train.scale_factor,
                device=device,
                save_dir=save_dir,
                epoch=epoch + 1,
            )
            if ext_psnr is not None:
                print(f"External | PSNR: {ext_psnr:.2f} dB | SSIM: {ext_ssim:.4f}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(student_encoder.state_dict(), os.path.join(save_dir, "best_student_guided.pth"))

    torch.save(student_encoder.state_dict(), os.path.join(save_dir, "last_student_guided.pth"))


if __name__ == "__main__":
    main()
