import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Channel-wise LayerNorm for tensors with shape [B, C, H, W]."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class NeXtBlock(nn.Module):
    """Lightweight ConvNeXt-style residual block."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x


class SpiDecoder(nn.Module):
    """
    Latent-to-image decoder.

    Input:
        z: [B, latent_channels, h, w]

    Output:
        image: [B, out_channels, 4h, 4w], normalized to [0, 1]
    """

    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 1,
        base_dim: int = 128,
        depth: int = 6,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(latent_channels, base_dim, kernel_size=1),
            LayerNorm(base_dim),
        )

        self.stages = nn.Sequential(
            *[NeXtBlock(dim=base_dim, kernel_size=3) for _ in range(depth)]
        )

        self.pre_upsample = nn.Conv2d(base_dim, out_channels * 16, kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(upscale_factor=4)
        self.final_refine = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.head(z)
        x = self.stages(x)
        x = self.pre_upsample(x)
        x = self.upsample(x)
        x = self.final_refine(x)
        return torch.sigmoid(x)
