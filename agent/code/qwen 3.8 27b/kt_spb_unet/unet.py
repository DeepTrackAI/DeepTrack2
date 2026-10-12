"""U-Net segmentation network for two-channel (KT, SPB) fluorescence.

Input : (B, 2, H, W)   -- channel0 = KT/green, channel1 = SPB/red.
Output: (B, 2, H, W)   -- per-pixel logits (apply sigmoid for probabilities).

Architecture (depths=N)
-----------------------
    encoder0 -> skip0
    pool
    encoder1 -> skip1
    pool
    ...
    encoder(N-2) -> skip(N-2)
    pool
    encoder(N-1) -> skip(N-1)            # deepest encoder, channels ch_N
    bottleneck(ch_N -> ch_N, same size)
    ---------------------------------
    up0(ch_N -> ch_{N-1}) + concat skip(N-1) -> dec0 -> ch_{N-1}
    up1(ch_{N-1} -> ch_{N-2}) + concat skip(N-2) -> dec1 -> ch_{N-2}
    ...
    up_{N-1}(ch_2 -> ch_1) + concat skip0 -> dec_{N-1} -> ch_1
    head(ch_1 -> out)

Canvas must be divisible by ``2**(depths-1)`` (e.g. 416 for depths=4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Standard U-Net.  ``base`` = width of level 0; ``depths`` = number of
    encoder levels (default 4)."""

    def __init__(self, in_channels: int = 2, out_channels: int = 2,
                 base: int = 64, depths: int = 4) -> None:
        super().__init__()
        assert depths >= 2
        self.depths = depths
        # Channel count at each encoder level.
        ch = [base * (2 ** i) for i in range(depths)]

        # Encoder path
        self.encoders = nn.ModuleList([
            _conv_block(in_channels if i == 0 else ch[i - 1], ch[i])
            for i in range(depths)
        ])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _conv_block(ch[-1], ch[-1])

        # Decoder path: ``depths`` stages, each up-casts from ch[i] to ch[i-1]
        # and concats with the level-(i-1) skip (ch[i-1] channels) -> 2*ch[i-1]
        # -> ch[i-1].
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(ch[i], ch[i - 1], kernel_size=2, stride=2)
            for i in range(depths - 1, 0, -1)
        ])
        self.decoders = nn.ModuleList([
            _conv_block(2 * ch[i - 1], ch[i - 1])
            for i in range(depths - 1, 0, -1)
        ])
        self.head = nn.Conv2d(ch[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips: list[torch.Tensor] = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            skips.append(x)
            if i < self.depths - 1:
                x = self.pool(x)
        x = self.bottleneck(x)

        # Decoder: the deepest skip (level N-1) is unneeded -- the bottleneck
        # already sits at that level; use the first N-1 skips, deepest first.
        decoder_skips = reversed(skips[:-1])
        for up, dec, skip in zip(self.up, self.decoders, decoder_skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.head(x)


def dice_bce_loss(pred: torch.Tensor, target: torch.Tensor,
                  weights: tuple[float, float] = (1.0, 1.5)) -> torch.Tensor:
    """Soft Dice + BCE, per-channel; the SPB channel is weighted higher
    (rarer class: 2 vs 6 objects).

    pred:    (B, 2, H, W) logits
    target:  (B, 2, H, W) {0,1}
    """
    p = torch.sigmoid(pred)
    # Per-class soft Dice (B,) per class.
    inter = (p * target).sum(dim=(2, 3))                # (B, 2)
    union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  # (B, 2)
    dice_per_ch = 1.0 - (2 * inter + 1e-7) / (union + 1e-7)   # (B, 2)

    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    per_class = [bce[:, c].mean() + dice_per_ch[:, c].mean() * weights[c]
                 for c in range(2)]
    return torch.stack(per_class).mean()


if __name__ == "__main__":
    for base, depths in [(32, 3), (32, 4), (64, 4)]:
        f = 2 ** (depths - 1)
        H = (416 + f - 1) // f * f
        W = (640 + f - 1) // f * f
        net = UNet(in_channels=2, out_channels=2, base=base, depths=depths)
        n = sum(p.numel() for p in net.parameters())
        x = torch.randn(2, 2, H, W)
        y = net(x)
        t = torch.rand(2, 2, H, W)
        l = dice_bce_loss(y, t)
        print(f"base={base} depths={depths} canvas {H}x{W}: "
              f"{n/1e6:.2f}M | out {tuple(y.shape)} | loss {l.item():.4f}")
    print("UNet OK")
