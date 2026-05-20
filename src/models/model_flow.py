import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_kron import SpiDecoder


class SignSTE(torch.autograd.Function):
    """Straight-through estimator for binary SPI patterns."""

    @staticmethod
    def forward(ctx, z: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(z)
        a = torch.sign(z)
        a[a == 0] = 1.0
        return a

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (z,) = ctx.saved_tensors
        return grad_output * (1.0 - torch.tanh(z).pow(2))


class DropPath(nn.Module):
    """Stochastic depth. This local implementation avoids an extra timm dependency."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class GlobalSpiLayerFast(nn.Module):
    """
    Learnable global SPI sampling layer.

    The layer learns binary measurement patterns with SignSTE, obtains compressed
    measurements, and reconstructs a coarse image with a DGI-style adjoint step.
    """

    def __init__(self, img_size: int = 128, sampling_ratio: float = 0.05):
        super().__init__()
        self.img_size = img_size
        self.num_pixels = img_size * img_size
        self.num_measurements = int(round(self.num_pixels * sampling_ratio))
        self.measurement_side = int(math.ceil(math.sqrt(self.num_measurements)))

        encoder = torch.empty(self.num_measurements, self.num_pixels)
        nn.init.orthogonal_(encoder)
        self.encoder = nn.Parameter(encoder.view(self.num_measurements, 1, img_size, img_size))

        mask = torch.ones(1, self.measurement_side * self.measurement_side)
        mask[:, self.num_measurements:] = 0
        mask = mask.view(1, 1, self.measurement_side, self.measurement_side)
        self.register_buffer("mask_2d", mask)

        self.bn_meas = nn.BatchNorm2d(1)
        self.bn_image = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        encoder_flat = self.encoder.view(self.num_measurements, self.num_pixels)
        binary_pattern = SignSTE.apply(encoder_flat) / math.sqrt(self.num_pixels)

        y_flat = torch.matmul(x_flat, binary_pattern.T)
        y_delta = y_flat - y_flat.mean(dim=1, keepdim=True)

        x_rec_flat = torch.matmul(y_delta, binary_pattern)
        x_rec = x_rec_flat.view(batch_size, 1, self.img_size, self.img_size)
        x_rec = self.bn_image(x_rec)

        padded_len = self.measurement_side * self.measurement_side
        y_padded = F.pad(y_delta, (0, padded_len - self.num_measurements))
        y_2d = y_padded.view(batch_size, 1, self.measurement_side, self.measurement_side)
        y_2d = self.bn_meas(y_2d)

        mask = self.mask_2d.expand(batch_size, -1, -1, -1)
        y_with_mask = torch.cat([y_2d, mask], dim=1)

        return y_with_mask, x_rec


# Backward-compatible alias for older checkpoints or scripts.
GlobalSpiLayer_Fast = GlobalSpiLayerFast


class LayerNorm(nn.Module):
    """LayerNorm for [B, C, H, W] feature maps."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class NeXtBlock(nn.Module):
    """ConvNeXt-style block with LayerScale and stochastic depth."""

    def __init__(self, dim: int, kernel_size: int = 3, drop_path: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim, 1, 1))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return residual + self.drop_path(x)


class AdaptiveGate(nn.Module):
    """Lightweight channel gate for fused explicit and implicit features."""

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        hidden_channels = max(in_channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.gate(x).view(x.size(0), x.size(1), 1, 1)
        return x * weight


class CrossDomainFusion(nn.Module):
    """Fuse explicit back-projection features and implicit measurement features."""

    def __init__(self, explicit_dim: int = 16, implicit_dim: int = 16):
        super().__init__()
        total_dim = explicit_dim + implicit_dim

        self.sft_gamma = nn.Conv2d(implicit_dim, explicit_dim, kernel_size=3, padding=1)
        self.sft_beta = nn.Conv2d(implicit_dim, explicit_dim, kernel_size=3, padding=1)
        self.gate = AdaptiveGate(total_dim, reduction=4)
        self.mix_conv = nn.Sequential(
            nn.Conv2d(total_dim, total_dim, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, explicit_feat: torch.Tensor, implicit_feat: torch.Tensor) -> torch.Tensor:
        gamma = self.sft_gamma(implicit_feat)
        beta = self.sft_beta(implicit_feat)
        explicit_feat = explicit_feat * (1.0 + gamma) + beta

        fused = torch.cat([explicit_feat, implicit_feat], dim=1)
        fused = self.gate(fused)
        return self.mix_conv(fused)


class ImplicitFeatureExtractor(nn.Module):
    """Extract aligned features from 2D-folded SPI measurements."""

    def __init__(self, in_channels: int = 2, out_channels: int = 16, target_size: tuple[int, int] = (32, 32)):
        super().__init__()
        self.target_size = target_size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, y_2d: torch.Tensor) -> torch.Tensor:
        feat = self.net(y_2d)
        return F.adaptive_avg_pool2d(feat, self.target_size)


class SpiBackboneSuperFused(nn.Module):
    """Backbone for fused explicit/implicit SPI features."""

    def __init__(self, in_channels: int = 32, latent_channels: int = 4, base_dim: int = 128, depth: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            LayerNorm(base_dim),
        )
        self.blocks = nn.Sequential(
            *[NeXtBlock(base_dim, kernel_size=3, drop_path=0.2) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            LayerNorm(base_dim),
            nn.Conv2d(base_dim, latent_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        residual = x
        x = self.blocks(x)
        x = x + residual
        return self.head(x)


# Backward-compatible alias for older checkpoints or scripts.
SpiBackbone_SuperFused = SpiBackboneSuperFused


class SpiStudent_ImageDomain(nn.Module):
    """
    Decoder-guided SPI student encoder.

    Input:
        image: [B, 1, H, W], normalized to [-1, 1]

    Output:
        z_est: [B, 4, H/4, W/4]
        reg_loss: reserved for compatibility
        x_rec: coarse back-projected image
    """

    def __init__(self, img_size: int = 128, sampling_ratio: float = 0.05):
        super().__init__()
        self.spi_layer = GlobalSpiLayerFast(img_size=img_size, sampling_ratio=sampling_ratio)
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=4)

        explicit_channels = 16
        implicit_channels = 16
        feature_size = img_size // 4

        self.implicit_branch = ImplicitFeatureExtractor(
            in_channels=2,
            out_channels=implicit_channels,
            target_size=(feature_size, feature_size),
        )
        self.cross_fusion = CrossDomainFusion(
            explicit_dim=explicit_channels,
            implicit_dim=implicit_channels,
        )
        self.backbone = SpiBackboneSuperFused(
            in_channels=explicit_channels + implicit_channels,
            latent_channels=4,
            base_dim=128,
            depth=3,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None, torch.Tensor]:
        y_meas, x_rec = self.spi_layer(x)
        explicit_feat = self.unshuffle(x_rec)
        implicit_feat = self.implicit_branch(y_meas)
        fused_feat = self.cross_fusion(explicit_feat, implicit_feat)
        z_est = self.backbone(fused_feat)
        return z_est, None, x_rec


class WrapperModel(nn.Module):
    """Optional end-to-end wrapper: image -> SPI encoder -> decoder -> image."""

    def __init__(self, img_size: int = 128, sampling_ratio: float = 0.05):
        super().__init__()
        self.encoder = SpiStudent_ImageDomain(img_size=img_size, sampling_ratio=sampling_ratio)
        self.decoder = SpiDecoder(latent_channels=4, out_channels=1, base_dim=128, depth=6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_est, _, _ = self.encoder(x)
        return self.decoder(z_est)
