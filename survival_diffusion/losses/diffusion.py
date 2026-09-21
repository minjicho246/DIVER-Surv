"""DDPM noise-prediction loss used by DIVER-Surv."""

import torch.nn.functional as F


def diffusion_loss(pred_noise, target_noise):
    """Return mean squared error between predicted and injected noise."""
    return F.mse_loss(pred_noise.float(), target_noise.float())
