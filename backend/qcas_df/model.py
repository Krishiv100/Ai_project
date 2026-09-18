import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class DSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, in_ch, 3, stride, groups=in_ch),
            ConvBNAct(in_ch, out_ch, 1, 1, p=0),
        )

    def forward(self, x):
        return self.block(x)


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_channels, 32, 5, 2),
            DSBlock(32, 64, 2),
            DSBlock(64, 96, 2),
            DSBlock(96, 160, 2),
            DSBlock(160, 224, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.fc(self.pool(self.net(x)))


class QualityEstimator(nn.Module):
    """Predict a scalar degradation severity and quality embedding."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNAct(3, 16, 5, 2),
            DSBlock(16, 24, 2),
            DSBlock(24, 40, 2),
            DSBlock(40, 64, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.embed = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
        )
        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, x):
        z = self.embed(self.encoder(x))
        q = torch.sigmoid(self.pred(z)).squeeze(1)
        return z, q


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """x in [0, 1], shape Bx3xHxW. Returns YCbCr approximately in [0,1]."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (-0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = 0.5 + (0.5 * r - 0.418688 * g - 0.081312 * b)
    return torch.cat([y, cb, cr], dim=1)


class QualityConditionedSpectralDecomposer(nn.Module):
    """Differentiable low/mid/high frequency decomposition.

    Unlike fixed 1/16 and 1/8 cutoffs, two ordered radial boundaries are predicted
    from the estimated input quality. Smooth sigmoid masks make the boundaries
    differentiable. The operation is applied independently to Y, Cb and Cr.
    """

    def __init__(self, quality_dim: int = 64, temperature: float = 0.035):
        super().__init__()
        self.temperature = temperature
        self.boundary_head = nn.Sequential(
            nn.Linear(quality_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        self.register_buffer("_dummy", torch.tensor(0.0), persistent=False)
        self._grid_cache = {}

    def _radial_grid(self, h: int, w: int, device, dtype):
        key = (h, w, str(device), str(dtype))
        if key in self._grid_cache:
            return self._grid_cache[key]
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        r = torch.sqrt(gx * gx + gy * gy) / math.sqrt(2.0)
        r = r.clamp(0.0, 1.0)[None, None]
        self._grid_cache[key] = r
        return r

    def _boundaries(self, q_embed):
        raw = self.boundary_head(q_embed)
        # b1 in [0.08, 0.38]
        b1 = 0.08 + 0.30 * torch.sigmoid(raw[:, 0])
        # gap in [0.12, ~0.48], b2 <= 0.88
        max_gap = (0.88 - b1).clamp(min=0.12)
        gap = 0.12 + (max_gap - 0.12) * torch.sigmoid(raw[:, 1])
        b2 = b1 + gap
        return b1, b2

    def forward(self, x: torch.Tensor, q_embed: torch.Tensor):
        b, c, h, w = x.shape
        ycc = rgb_to_ycbcr(x)
        spec = torch.fft.fftshift(torch.fft.fft2(ycc, norm="ortho"), dim=(-2, -1))
        r = self._radial_grid(h, w, x.device, x.dtype)
        b1, b2 = self._boundaries(q_embed)
        b1v = b1[:, None, None, None]
        b2v = b2[:, None, None, None]
        t = self.temperature

        low = torch.sigmoid((b1v - r) / t)
        high = torch.sigmoid((r - b2v) / t)
        mid = torch.sigmoid((r - b1v) / t) * torch.sigmoid((b2v - r) / t)
        masks = torch.cat([low, mid, high], dim=1)
        masks = masks / (masks.sum(dim=1, keepdim=True) + 1e-6)

        outs = []
        for i in range(3):
            masked = spec * masks[:, i : i + 1]
            recon = torch.fft.ifft2(torch.fft.ifftshift(masked, dim=(-2, -1)), norm="ortho").real
            outs.append(recon)
        out = torch.cat(outs, dim=1)  # B x 9 x H x W
        return out, {"b1": b1, "b2": b2, "masks": masks}


class SpatialDifferenceInput(nn.Module):
    """Parameter-free local average-difference cue inspired by ADC."""

    def forward(self, x):
        local_avg = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        diff = x - local_avg
        return torch.cat([x, diff], dim=1)  # 6 channels


class QCASDF(nn.Module):
    """Quality-Conditioned Adaptive Spectral Deepfake detector.

    Core ideas:
      1. estimate input degradation quality,
      2. use quality-conditioned ordered spectral boundaries,
      3. preserve chromatic frequency cues in YCbCr,
      4. dynamically gate spatial and spectral evidence.
    """

    def __init__(self, feature_dim=256, quality_dim=64, num_classes=2):
        super().__init__()
        self.quality = QualityEstimator(quality_dim)
        self.spatial_input = SpatialDifferenceInput()
        self.spectral = QualityConditionedSpectralDecomposer(quality_dim)
        self.spatial_encoder = FeatureEncoder(6, feature_dim)
        self.spectral_encoder = FeatureEncoder(9, feature_dim)

        self.gate = nn.Sequential(
            nn.Linear(quality_dim + 2 * feature_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3 + quality_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        q_embed, q_pred = self.quality(x)

        s_in = self.spatial_input(x)
        s_feat = self.spatial_encoder(s_in)

        f_in, spec_info = self.spectral(x, q_embed)
        f_feat = self.spectral_encoder(f_in)

        gate_logits = self.gate(torch.cat([q_embed, s_feat, f_feat], dim=1))
        gate = torch.softmax(gate_logits, dim=1)
        fused = gate[:, 0:1] * s_feat + gate[:, 1:2] * f_feat
        interaction = torch.abs(s_feat - f_feat)

        logits = self.classifier(
            torch.cat([fused, s_feat, f_feat, q_embed], dim=1)
        )
        return {
            "logits": logits,
            "quality": q_pred,
            "gate": gate,
            "b1": spec_info["b1"],
            "b2": spec_info["b2"],
            "spatial_feat": s_feat,
            "spectral_feat": f_feat,
            "interaction": interaction,
        }
