"""The detectors (SPEC.md, the candidates K, A and A'), each of the input channels (B, 3, length) to its heads on its
grid of cells, `stride` points each:

    heat:       (B, cells) logits of an apex in the cell
    offset:     (B, cells) where in the cell, 0 to 1
    log_widths: (B, 2, cells) log of the distances to 5 % of the height, left and right, in points
    denoised:   (B, 2, length) the clean signal's linear and log channels, of those that have the head

    A  "transformer": convolutions, a Transformer encoder at a quarter of the length, and back up with skips
    A2 "unet":        convolutions only, a U-Net with dilated blocks at its bottom
    K  "kensert":     the network of Kensert et al. (resources/model/2022_Kensert), of 3 channels in, at stride 32
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from features import CHANNELS

PRIOR = 0.01  # of an apex in a cell, which the heat's bias starts at


class ConvBlock(nn.Module):
    """Two convolutions, GroupNorm and GELU, with a residual."""

    def __init__(self, channels_in: int, channels_out: int, kernel: int = 5, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(channels_in, channels_out, kernel, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(8, channels_out), channels_out)
        self.conv2 = nn.Conv1d(channels_out, channels_out, kernel, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(min(8, channels_out), channels_out)
        self.skip = nn.Conv1d(channels_in, channels_out, 1) if channels_in != channels_out else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        return F.gelu(self.norm2(self.conv2(h)) + self.skip(x))


class Heads(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.out = nn.Conv1d(channels, 4, 1)
        nn.init.constant_(self.out.bias[0], -math.log((1 - PRIOR) / PRIOR))

    def forward(self, x) -> dict:
        out = self.out(x)
        return {"heat": out[:, 0], "offset": torch.sigmoid(out[:, 1]), "log_widths": out[:, 2:4]}


class Detector(nn.Module):
    """Pads the input to a multiple of `multiple` and cuts the heads back to its cells."""
    stride = 1
    multiple = 1

    def forward(self, x):
        length = x.shape[-1]
        padding = (-length) % self.multiple
        out = self.body(F.pad(x, (0, padding)))
        cells = -(-length // self.stride)
        out = {key: value[..., :cells] for key, value in out.items() if key != "denoised"} | (
            {"denoised": out["denoised"][..., :length]} if "denoised" in out else {}
        )
        return out


class TransformerUNet(Detector):
    multiple = 4

    def __init__(self, width: int = 64, middle: int = 96, d_model: int = 128, layers: int = 4, heads: int = 4,
                 feedforward: int = 512, dropout: float = 0.1, denoise: bool = True):
        super().__init__()
        self.stem = ConvBlock(CHANNELS, width, 7)
        self.down1 = nn.Sequential(nn.Conv1d(width, middle, 4, 2, 1), ConvBlock(middle, middle))
        self.down2 = nn.Sequential(nn.Conv1d(middle, d_model, 4, 2, 1), ConvBlock(d_model, d_model))
        # a position of its own of each token, by its neighbors: nothing absolute, as the apexes are anywhere
        self.position = nn.Conv1d(d_model, d_model, 31, padding=15, groups=d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, feedforward, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.up2 = ConvBlock(d_model + middle, middle)
        self.up1 = ConvBlock(middle + width, width)
        self.heads = Heads(width)
        self.denoise = nn.Conv1d(width, 2, 1) if denoise else None

    def body(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        t = s2 + self.position(s2)
        t = self.encoder(t.transpose(1, 2)).transpose(1, 2)
        u2 = self.up2(torch.cat([F.interpolate(t, size=s1.shape[-1], mode="linear", align_corners=False), s1], 1))
        u1 = self.up1(torch.cat([F.interpolate(u2, size=s0.shape[-1], mode="linear", align_corners=False), s0], 1))
        out = self.heads(u1)
        if self.denoise is not None:
            out["denoised"] = self.denoise(u1)
        return out


class ConvUNet(Detector):
    multiple = 8

    def __init__(self, channels: tuple = (32, 64, 128, 256), dilations: tuple = (1, 2, 4, 8, 16), denoise: bool = True):
        super().__init__()
        self.stem = ConvBlock(CHANNELS, channels[0], 7)
        self.downs = nn.ModuleList(
            nn.Sequential(nn.Conv1d(low, high, 4, 2, 1), ConvBlock(high, high))
            for low, high in zip(channels[:-1], channels[1:])
        )
        self.bottom = nn.Sequential(*(ConvBlock(channels[-1], channels[-1], 5, dilation) for dilation in dilations))
        self.ups = nn.ModuleList(ConvBlock(high + low, low) for low, high in zip(channels[:-1], channels[1:]))
        self.heads = Heads(channels[0])
        self.denoise = nn.Conv1d(channels[0], 2, 1) if denoise else None

    def body(self, x):
        skips = [self.stem(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        h = self.bottom(skips.pop())
        for up, skip in zip(reversed(self.ups), reversed(skips)):
            h = up(torch.cat([F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False), skip], 1))
        out = self.heads(h)
        if self.denoise is not None:
            out["denoised"] = self.denoise(h)
        return out


class KensertNet(Detector):
    """Each block a Conv1d of 9, BatchNorm, ReLU and MaxPool1d of 2, and beside it a strided Conv1d, BatchNorm, ReLU,
    added; then a Conv1d of 1, BatchNorm, ReLU, and the heads. As the authors' ConvNet, but for the channels in and
    the heads out."""
    stride = 32
    multiple = 32

    def __init__(self, filters: tuple = (64, 128, 128, 256, 256), kernel: int = 9):
        super().__init__()

        def conv(channels_in, channels_out, stride):
            return nn.Sequential(nn.Conv1d(channels_in, channels_out, kernel, stride, kernel // 2),
                                 nn.BatchNorm1d(channels_out), nn.ReLU())

        self.mains, self.residuals = nn.ModuleList(), nn.ModuleList()
        channels_in = CHANNELS
        for channels_out in filters:
            self.mains.append(nn.Sequential(conv(channels_in, channels_out, 1), nn.MaxPool1d(2)))
            self.residuals.append(conv(channels_in, channels_out, 2))
            channels_in = channels_out
        self.last = nn.Sequential(nn.Conv1d(channels_in, channels_in, 1), nn.BatchNorm1d(channels_in), nn.ReLU())
        self.heads = Heads(channels_in)

    def body(self, x):
        for main, residual in zip(self.mains, self.residuals):
            x = main(x) + residual(x)
        return self.heads(self.last(x))


MODELS = {"transformer": TransformerUNet, "unet": ConvUNet, "kensert": KensertNet}


def build(name: str, **options) -> Detector:
    return MODELS[name](**options)
