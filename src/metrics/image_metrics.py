from math import exp

import torch
import torch.nn.functional as F


def calculate_psnr(img1, img2):
    """
    Calculate PSNR for a batch of images in [0, 1].
    """

    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    psnr_per_image = 10 * torch.log10(1.0 / (mse + 1e-10))

    return torch.mean(psnr_per_image).item()


def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    window_1d = gaussian_window(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()

    return window


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate SSIM for a batch of images in [0, 1].
    """

    with torch.no_grad():
        channel = img1.size(1)
        window = create_window(window_size, channel).to(img1.device).type_as(img1)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )

        if size_average:
            return ssim_map.mean().item()

        return ssim_map.mean(1).mean(1).mean(1)
