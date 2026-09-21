"""3D U-Net autoencoder and convolution blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_3d_norm(num_channels, norm_type="group", gn_groups=8):
    """Construct the requested normalization with a valid group count."""
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return nn.BatchNorm3d(num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True, track_running_stats=False)
    groups = min(max(1, gn_groups), num_channels)
    while groups > 1 and (num_channels % groups != 0):
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class DoubleConv(nn.Module):
    """Apply two 3D convolution, normalization, and ReLU blocks."""

    def __init__(self, in_ch, out_ch, norm_type="group", gn_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            make_3d_norm(out_ch, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            make_3d_norm(out_ch, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsample by max pooling followed by a double convolution."""

    def __init__(self, in_ch, out_ch, norm_type="group", gn_groups=8):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_ch, out_ch, norm_type=norm_type, gn_groups=gn_groups),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample, pad to the skip shape, concatenate, and convolve."""

    def __init__(self, in_ch, skip_ch, out_ch, norm_type="group", gn_groups=8):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(skip_ch + out_ch, out_ch, norm_type=norm_type, gn_groups=gn_groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad dynamically
        diff = [x2.size(i) - x1.size(i) for i in range(2, 5)]
        x1 = F.pad(
            x1,
            [
                diff[2] // 2,
                diff[2] - diff[2] // 2,
                diff[1] // 2,
                diff[1] - diff[1] // 2,
                diff[0] // 2,
                diff[0] - diff[0] // 2,
            ],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet3DAutoencoder(nn.Module):
    """Encode 3D volumes into latent tokens and decode with U-Net skip features."""

    def __init__(
        self,
        in_channels=4,
        base_filters=32,
        latent_channels=4,
        latent_dim=128,
        dropout_rate=0.2,
        norm_type="group",
        gn_groups=8,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_dim = latent_dim

        f1, f2, f3, f4 = base_filters, base_filters * 2, base_filters * 4, base_filters * 8

        self.inc = DoubleConv(in_channels, f1, norm_type=norm_type, gn_groups=gn_groups)
        self.down1 = Down(f1, f2, norm_type=norm_type, gn_groups=gn_groups)
        self.down2 = Down(f2, f3, norm_type=norm_type, gn_groups=gn_groups)
        self.down3 = Down(f3, f4, norm_type=norm_type, gn_groups=gn_groups)
        self.down4 = Down(f4, f4, norm_type=norm_type, gn_groups=gn_groups)

        # FC dimensions depend on the input shape; initialize on the first forward pass.
        self.fc_enc = None
        self.fc_dec = None

        self.up4 = Up(f4, f4, f4, norm_type=norm_type, gn_groups=gn_groups)
        self.up3 = Up(f4, f3, f3, norm_type=norm_type, gn_groups=gn_groups)
        self.up2 = Up(f3, f2, f2, norm_type=norm_type, gn_groups=gn_groups)
        self.up1 = Up(f2, f1, f1, norm_type=norm_type, gn_groups=gn_groups)
        self.outc = nn.Conv3d(f1, in_channels, 1)
        self.latent_drop = nn.Dropout(p=dropout_rate)

    def _init_fcs(self, x):
        if self.fc_enc is None:
            flat_dim = x.numel() // x.size(0)
            hidden_dim = self.latent_channels * self.latent_dim
            self.fc_enc = nn.Linear(flat_dim, hidden_dim).to(x.device)
            self.fc_dec = nn.Linear(hidden_dim, flat_dim).to(x.device)

    def encode(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Bottleneck feature map

        self._init_fcs(x5)

        b = x5.size(0)
        z = self.fc_enc(x5.view(b, -1))
        z = self.latent_drop(z)
        z = z.view(b, self.latent_channels, self.latent_dim)
        return z, (x1, x2, x3, x4, x5)

    def decode(self, z, features):
        x1, x2, x3, x4, x5 = features
        b = z.size(0)

        flat_dec = self.fc_dec(z.view(b, -1))
        # Reshape using captured shape from encoder pass
        x5_rec = flat_dec.view(b, *x5.shape[1:])

        d4 = self.up4(x5_rec, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        return self.outc(d1)

    def forward(self, x):
        z, feats = self.encode(x)
        recon = self.decode(z, feats)
        return recon, z
