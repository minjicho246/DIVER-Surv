"""Diffusion noising, timestep conditioning, and image-token projection."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet3DAutoencoder


def sinusoidal_timestep_embedding(timesteps, dim, max_period=10000):
    """Construct sinusoidal diffusion timestep features."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    half = dim // 2
    device = timesteps.device
    if half > 0:
        freq = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    else:
        emb = timesteps.float().unsqueeze(1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DiffusionImageEncoder3D(nn.Module):
    """Predict diffusion noise and project latent tokens for survival prediction."""

    def __init__(
        self,
        in_channels=4,
        base_filters=32,
        latent_channels=4,
        latent_dim=128,
        embed_dim=256,
        dropout_rate=0.2,
        norm_type="group",
        gn_groups=8,
        diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        time_embed_dim=128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.diffusion_steps = diffusion_steps
        self.time_embed_dim = time_embed_dim

        self.denoiser = UNet3DAutoencoder(
            in_channels=in_channels,
            base_filters=base_filters,
            latent_channels=latent_channels,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )

        time_hidden_dim = max(time_embed_dim, in_channels * 4)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_hidden_dim),
            nn.SiLU(),
            nn.Linear(time_hidden_dim, in_channels * 2),
        )

        # Project each image modality latent independently:
        # latent: [B, N_IMG, latent_dim] -> img_emb: [B, N_IMG, embed_dim]
        self.img_proj = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)

    def q_sample(self, x0, t, noise):
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1, 1).to(x0.dtype)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

    def sample_timesteps(self, batch_size, device, t_max=None):
        if t_max is None:
            t_max = self.diffusion_steps - 1
        t_max = int(max(0, min(self.diffusion_steps - 1, t_max)))
        return torch.randint(0, t_max + 1, (batch_size,), device=device)

    def forward(self, x, add_noise=True, t_override=None):
        B = x.size(0)
        device = x.device

        if add_noise:
            if t_override is None:
                t = self.sample_timesteps(B, device)
            else:
                t = t_override.to(device=device, dtype=torch.long).clamp_(
                    0, self.diffusion_steps - 1
                )
            target_noise = torch.randn_like(x)
            x_noisy = self.q_sample(x, t, target_noise)
        else:
            t = torch.zeros((B,), dtype=torch.long, device=device)
            target_noise = torch.zeros_like(x)
            x_noisy = x

        t_emb = sinusoidal_timestep_embedding(t, self.time_embed_dim)
        time_scale_shift = (
            self.time_mlp(t_emb).to(dtype=x.dtype).view(B, self.in_channels * 2, 1, 1, 1)
        )
        scale, shift = torch.chunk(time_scale_shift, chunks=2, dim=1)
        denoiser_in = x_noisy * (1.0 + scale) + shift

        pred_noise, latent = self.denoiser(denoiser_in)
        img_emb = self.img_proj(latent)

        return {
            "pred_noise": pred_noise,
            "target_noise": target_noise,
            "img_emb": img_emb,
            "timestep": t,
        }
