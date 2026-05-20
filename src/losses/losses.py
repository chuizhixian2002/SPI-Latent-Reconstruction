import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss, a smooth approximation of L1 loss.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class GradientLoss(nn.Module):
    """
    Sobel-gradient consistency loss.
    """

    def __init__(self):
        super().__init__()

        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)

        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)

        self.register_buffer("weight_x", kernel_x)
        self.register_buffer("weight_y", kernel_y)

    def forward(self, pred, target):
        pred_gx = F.conv2d(pred, self.weight_x, padding=1)
        pred_gy = F.conv2d(pred, self.weight_y, padding=1)

        target_gx = F.conv2d(target, self.weight_x, padding=1)
        target_gy = F.conv2d(target, self.weight_y, padding=1)

        return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


class SpiDistillationLoss(nn.Module):
    """
    Distillation loss used for guided SPI encoder training.

    It contains:
        1. latent-level L1 + cosine consistency
        2. image-domain Charbonnier reconstruction loss
        3. image-gradient consistency loss
    """

    def __init__(self, w_lat=0.1, w_pix=6.0, w_grad=1.0):
        super().__init__()

        self.w_lat = w_lat
        self.w_pix = w_pix
        self.w_grad = w_grad

        self.criterion_pixel = CharbonnierLoss()
        self.criterion_grad = GradientLoss()
        self.criterion_l1 = nn.L1Loss()

    def forward(self, z_est, z_gt, x_student_dec, x_gt):
        loss_dict = {}

        z_est_flat = z_est.flatten(1)
        z_gt_flat = z_gt.flatten(1)

        loss_lat_cos = (1 - F.cosine_similarity(z_est_flat, z_gt_flat, dim=1)).mean()
        loss_lat_l1 = self.criterion_l1(z_est, z_gt)

        loss_dict["latent"] = loss_lat_l1 + 0.1 * loss_lat_cos
        loss_dict["pixel"] = self.criterion_pixel(x_student_dec, x_gt)
        loss_dict["grad"] = self.criterion_grad(x_student_dec, x_gt)

        total_loss = (
            self.w_lat * loss_dict["latent"]
            + self.w_pix * loss_dict["pixel"]
            + self.w_grad * loss_dict["grad"]
        )

        return total_loss, loss_dict
