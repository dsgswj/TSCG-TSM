import torch
import torch.nn as nn
import numbers
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    """LayerNorm for 2D feature maps (B, C, H, W)."""

    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        # Use standard nn.LayerNorm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = to_3d(x)
        x = self.norm(x)
        x = to_4d(x, h, w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.SiLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SCGM(nn.Module):
    """Spatial-Channel Global Modulator."""

    def __init__(self, feature_dim):
        super().__init__()
        self.norm = LayerNorm(feature_dim)
        self.chan_attn = ChannelAttention(feature_dim)
        self.spat_attn = SpatialAttention()
        self.conv_1 = nn.Conv2d(feature_dim, feature_dim, 1, 1, 0)
        self.conv_2 = nn.Conv2d(feature_dim, feature_dim, 1, 1, 0)

    def forward(self, x):
        x_norm = self.norm(x)
        x_attn = self.conv_1(self.chan_attn(x_norm) * x_norm) + self.conv_2(self.spat_attn(x_norm) * x_norm)
        return x + x_attn


# Alias for backward compatibility
SpatialChannelGatedModule = SCGM

