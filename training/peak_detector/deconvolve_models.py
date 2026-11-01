"""The later stage's model (SPEC_deconvolution.md 4), and the later stage on a window: the model, then the heights
by least squares. Of the input channels x (B, 3, N) and the candidates (B, K, 4: apex, confidence, left and right
widths to 5 %, in points; candidate_mask (B, K) of those there), in the linear channel's units (deconvolve_data.py):

    real:       (B, K) logits of each candidate's being a peak's
    log_height: (B, K) log of its height as the network sees it, the prior of the least squares; from the signal's
                at its apex
    shape:      (n, N) of each candidate there (n of candidate_mask, in its order), of top 1, 0 where no peak may be
    baseline, others: (B, N); out on the log channel's scale (features.py), as fine near 0 as the noise, and back
                to the linear one (background "log"; "linear", of the runs before it, out as they are: as fine as some
                1e-4 of the window's top, some 17 times the noise of windows of a top 1e5 times it)
    log_sigma:  (B, N); the noise from the window's (features.Scaling)

    encoder: "transformer", convolutions and a Transformer at a quarter of the length (as models.TransformerUNet), or
             "unet", convolutions only (as models.ConvUNet): features at each point, and tokens to attend to
    decoder: a query a candidate, of the features at its apex, its widths and confidence, and where it is; layers
             of attention among the queries (to tell duplicates apart) and to the tokens
    shapes:  on a grid of `grid` points about each candidate, `reach` times its widths each side, convolutions of
             the features there, the input channels, and where the other candidates are, as its query has them; out,
             the top's place and the log slopes up to it and down from it, each >= 0, summed from the top out to a
             unimodal shape of no formula (unimodal); past the grid, on at the slopes of its ends
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deconvolve import clusters, corrections, heights
from deconvolve_data import collate, inputs
from features import CHANNELS, LOG_SCALE
from models import ConvBlock

REAL = 0.5  # of the realness, from which a candidate is taken for a peak's
STRENGTH = 0.01  # of the pull of the heights to the network's, of what the data tell of each alone (deconvolve.heights)
BUMP = 1 / math.sqrt(2 * math.log(20))  # sd of a Gaussian down to 5 % at 1
FLOOR = 5.0  # of the log noise below the window's (features.Scaling), the least taken in weighing the points
BACKGROUND = "log"  # of the baseline and the others out: on the log channel's scale; "linear", as they are
# of the baseline and the others out on the log channel's scale, the most, in the window's tops: where no loss sees
# them (the noise not known), they went up to 1e11, and the losses' values to inf x 0
MOST = 4.0


def unlog(values: torch.Tensor, noise: torch.Tensor, most: float = MOST) -> torch.Tensor:
    """Of the log channel's scale (features.py: sign(v) log1p(|v| / noise) / LOG_SCALE), v again, of |v| up to most."""
    limit = torch.log1p(most / noise) / LOG_SCALE
    return torch.sign(values) * noise * torch.expm1(torch.minimum(values.abs(), limit) * LOG_SCALE)


def sinusoid(positions: torch.Tensor, channels: int) -> torch.Tensor:
    """(..., channels) of positions (...) in points: sines and cosines of wavelengths 4 to 10000 points."""
    half = channels // 2
    wavelengths = 4 * 2500 ** (torch.arange(half, device=positions.device) / max(half - 1, 1))
    angles = 2 * math.pi * positions[..., None].float() / wavelengths
    return torch.cat([angles.sin(), angles.cos()], -1)


def gather(values: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """values (B, C, N) of item rows (P) at positions (P, M) in points, by linear interpolation, the ends held:
    (P, M, C)."""
    batch, channels, length = values.shape
    flat = values.transpose(1, 2).reshape(batch * length, channels)
    positions = positions.clamp(0, length - 1)
    low = positions.floor().long().clamp(max=max(length - 2, 0))
    part = (positions - low).unsqueeze(-1).to(values.dtype)
    at = rows[:, None] * length + low
    return flat[at] * (1 - part) + flat[(at + 1).clamp(max=batch * length - 1)] * part


def sides(points: torch.Tensor, apex, left, right) -> torch.Tensor:
    """Where points are of candidates (apex, left, right, each (P, 1)): in their widths, < 0 on the left."""
    distance = points - apex
    return torch.where(distance < 0, distance / left, distance / right)


def unimodal(rise: torch.Tensor, fall: torch.Tensor, top: torch.Tensor, where: torch.Tensor,
             reach: float) -> torch.Tensor:
    """The log of a shape of top 1 at where (P, Q), of the network's out on a grid from -reach to reach (P, M): log
    slopes up exp(rise) and down exp(fall), and the top's place, of softmax weights top on the grid. Up to the top,
    the slopes up summed from it; after it, those down: of any depth each side. Past the grid, on at the slopes of
    its ends."""
    count = rise.shape[-1]
    step = 2 * reach / (count - 1)
    place = (torch.softmax(top, -1) * torch.linspace(-reach, reach, count, device=top.device)).sum(-1, keepdim=True)

    def integral(slopes, at):
        """Of the slopes from -reach to at, linear in between the grid's points."""
        summed = F.pad(torch.cumsum((slopes[..., 1:] + slopes[..., :-1]) / 2, -1) * step, (1, 0))
        index = ((at + reach) / step).clamp(0, count - 1)
        low = index.floor().long().clamp(max=count - 2)
        part = index - low
        inside = summed.gather(-1, low) * (1 - part) + summed.gather(-1, low + 1) * part
        return inside + slopes[..., :1] * (at + reach).clamp(max=0) + slopes[..., -1:] * (at - reach).clamp(min=0)

    up, down = torch.exp(rise.clamp(max=8.0)), torch.exp(fall.clamp(max=8.0))
    return torch.minimum(integral(up, where) - integral(up, place), integral(down, place) - integral(down, where))


class TransformerEncoder(nn.Module):
    multiple = 4

    def __init__(self, width: int = 64, middle: int = 96, d_model: int = 128, layers: int = 4, heads: int = 4,
                 feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.channels, self.token_channels, self.stride = width, d_model, 4
        self.stem = ConvBlock(CHANNELS, width, 7)
        self.down1 = nn.Sequential(nn.Conv1d(width, middle, 4, 2, 1), ConvBlock(middle, middle))
        self.down2 = nn.Sequential(nn.Conv1d(middle, d_model, 4, 2, 1), ConvBlock(d_model, d_model))
        self.position = nn.Conv1d(d_model, d_model, 31, padding=15, groups=d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, feedforward, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.up2 = ConvBlock(d_model + middle, middle)
        self.up1 = ConvBlock(middle + width, width)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        t = self.encoder((s2 + self.position(s2)).transpose(1, 2)).transpose(1, 2)
        u2 = self.up2(torch.cat([F.interpolate(t, size=s1.shape[-1], mode="linear", align_corners=False), s1], 1))
        u1 = self.up1(torch.cat([F.interpolate(u2, size=s0.shape[-1], mode="linear", align_corners=False), s0], 1))
        return u1, t


class UNetEncoder(nn.Module):
    def __init__(self, channels: tuple = (32, 64, 128, 256), dilations: tuple = (1, 2, 4, 8, 16)):
        super().__init__()
        self.channels, self.token_channels = channels[0], channels[-1]
        self.stride = self.multiple = 2 ** (len(channels) - 1)
        self.stem = ConvBlock(CHANNELS, channels[0], 7)
        self.downs = nn.ModuleList(nn.Sequential(nn.Conv1d(low, high, 4, 2, 1), ConvBlock(high, high))
                                   for low, high in zip(channels[:-1], channels[1:]))
        self.bottom = nn.Sequential(*(ConvBlock(channels[-1], channels[-1], 5, dilation) for dilation in dilations))
        self.ups = nn.ModuleList(ConvBlock(high + low, low) for low, high in zip(channels[:-1], channels[1:]))

    def forward(self, x):
        skips = [self.stem(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        bottom = h = self.bottom(skips.pop())
        for up, skip in zip(reversed(self.ups), reversed(skips)):
            h = up(torch.cat([F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False), skip], 1))
        return h, bottom


ENCODERS = {"transformer": TransformerEncoder, "unet": UNetEncoder}


class Deconvolver(nn.Module):
    def __init__(self, encoder: str = "transformer", encoder_options: dict | None = None, d_model: int = 128,
                 layers: int = 3, heads: int = 4, feedforward: int = 512, dropout: float = 0.1, grid: int = 128,
                 reach: float = 4.0, shape_width: int = 64, shape_dilations: tuple = (1, 2, 4, 8),
                 background: str = BACKGROUND):
        super().__init__()
        self.encoder = ENCODERS[encoder](**(encoder_options or {}))
        channels = self.encoder.channels
        self.d_model, self.grid, self.reach, self.background = d_model, grid, reach, background
        self.tokens = nn.Linear(self.encoder.token_channels, d_model)
        self.query_features = nn.Linear(channels, d_model)
        self.query_widths = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerDecoderLayer(d_model, heads, feedforward, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.real = nn.Linear(d_model, 1)
        self.height = nn.Linear(d_model, 1)
        self.trace = nn.Conv1d(channels, 3, 1)
        # the features, the channels, where on the grid (in widths), the width of that side, the others' places
        self.shape_in = nn.Conv1d(channels + CHANNELS + 3, shape_width, 1)
        self.film = nn.Linear(d_model, 2 * shape_width)
        self.shape_blocks = nn.Sequential(*(ConvBlock(shape_width, shape_width, 5, d) for d in shape_dilations))
        self.shape_out = nn.Conv1d(shape_width, 3, 1)
        with torch.no_grad():  # at first, the top at the apex given, and down to 5 % at the widths given
            self.shape_out.weight.mul_(0.1)
            self.shape_out.bias.copy_(torch.tensor([math.log(3.0), math.log(3.0), 0.0]))

    def forward(self, x, candidates, candidate_mask, present, scaling) -> dict:
        """present (B, N): 1 where a peak may be (deconvolve.present); scaling (B, 3): the window's
        (deconvolve_data.inputs)."""
        batch, _, length = x.shape
        features, tokens = self.encoder(F.pad(x, (0, (-length) % self.encoder.multiple)))
        features = features[..., :length]
        noise = (scaling[:, 2] / scaling[:, 1]).float()[:, None]
        with torch.autocast(x.device.type, enabled=False):  # of a baseline far from 0, finer than bf16 has it
            trace = self.trace(features.float())
            baseline, others = trace[:, 0], trace[:, 1]
            if self.background == "log":
                baseline, others = unlog(baseline, noise), unlog(others, noise)
        out = {"baseline": baseline, "others": others, "log_sigma": trace[:, 2] + noise.log()}

        count = candidates.shape[1]
        apex, confidence = candidates[..., 0], candidates[..., 1]
        left, right = candidates[..., 2].clamp(min=0.5), candidates[..., 3].clamp(min=0.5)
        rows = torch.arange(batch, device=x.device).repeat_interleave(count)
        at_apex = gather(features, rows, apex.reshape(-1, 1)).reshape(batch, count, -1)
        query = (self.query_features(at_apex) + sinusoid(apex, self.d_model)
                 + self.query_widths(torch.stack([left.log(), right.log(), confidence], -1)))
        places = (torch.arange(tokens.shape[-1], device=x.device) + 0.5) * self.encoder.stride - 0.5
        memory = self.tokens(tokens.transpose(1, 2)) + sinusoid(places, self.d_model)
        padding = ~candidate_mask
        padding[:, 0] &= candidate_mask.any(1)  # of an item of none, one to attend to, of no use
        query = self.norm(self.decoder(query, memory, tgt_key_padding_mask=padding))
        out["real"] = self.real(query).squeeze(-1).float()
        signal = gather(x[:, :1], rows, apex.reshape(-1, 1)).reshape(batch, count)
        out["log_height"] = self.height(query).squeeze(-1).float() + torch.maximum(signal, noise).log()

        # the candidates' places, of their realness as it is, for each to see the others
        points = torch.arange(length, device=x.device, dtype=torch.float32)
        weight = torch.sigmoid(out["real"].detach()) * candidate_mask
        bumps = torch.exp(-0.5 * (sides(points, apex[..., None], left[..., None], right[..., None]) / BUMP) ** 2)
        crowd = (bumps * weight[..., None]).sum(1, keepdim=True)

        rows, there = rows.reshape(batch, count)[candidate_mask], candidate_mask
        if not len(rows):
            out["shape"] = trace.new_zeros(0, length)
            return out
        apex, left, right = apex[there][:, None], left[there][:, None], right[there][:, None]
        z = torch.linspace(-self.reach, self.reach, self.grid, device=x.device)
        place = apex + torch.where(z < 0, z * left, z * right)
        width = torch.where(z < 0, left.log(), right.log())
        channels = torch.cat([gather(features, rows, place).float(), gather(x, rows, place), gather(crowd, rows, place),
                              torch.stack([z.expand_as(place) / self.reach, width / 5], -1)], -1)
        h = self.shape_in(channels.transpose(1, 2))
        scale, shift = self.film(query[there]).unsqueeze(-1).chunk(2, 1)
        hidden = self.shape_blocks(h * (1 + scale) + shift)
        with torch.autocast(x.device.type, enabled=False):
            rise, fall, top = self.shape_out(hidden.float()).unbind(1)
        log_shape = unimodal(rise, fall, top, sides(points, apex, left, right), self.reach)
        # of top 1 where peaks may be, of the logs: not over a top near 0, whose gradients may overflow
        where = present[rows] > 0
        highest = torch.where(where, log_shape, torch.full_like(log_shape, -1e9)).amax(-1, keepdim=True)
        out["shape"] = torch.where(where, torch.exp((log_shape - highest).clamp(max=0)), 0.0)
        return out


def log_noise(outputs: dict, scaling) -> torch.Tensor:
    """The log noise (B, N) the points are weighed by, of the outputs: not below the window's by more than FLOOR."""
    floor = (scaling[:, 2] / scaling[:, 1]).log().float()[:, None] - FLOOR
    return torch.maximum(outputs["log_sigma"].float(), floor.to(outputs["log_sigma"].device))


def build(**options) -> Deconvolver:
    return Deconvolver(**options)


def options_of(config: dict) -> dict:
    """The model's options of a run's config: of a run before the background's option, its linear one."""
    return {"background": "linear"} | config["model_options"]


def load(checkpoint: str, device="cpu") -> Deconvolver:
    """The model of a checkpoint of train_deconvolution.py, to be used."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build(**options_of(saved["config"]))
    model.load_state_dict(saved["model"])
    return model.to(device).eval()


def run(model, batch: dict, device) -> dict:
    """The model's outputs of a batch (deconvolve_data.collate)."""
    on = {key: batch[key].to(device) for key in ("x", "candidates", "candidate_mask", "present", "scaling")}
    return model(on["x"], on["candidates"], on["candidate_mask"], on["present"], on["scaling"])


@torch.no_grad()
def estimates(outputs: dict, batch: dict, real: float = REAL, strength: float = STRENGTH) -> list[dict]:
    """Of each window of a batch and the model's outputs of it, the later stage's: the candidates, each {"apex",
    "confidence" (its realness), "real" (taken for a peak's), and of those taken "area" and "sigma" (of the area),
    "shape" (the peak on the window's points) and "top" (the point of its top)}, the heights of their shapes by
    least squares (deconvolve.heights: weighed by the noise given, a >= 0, pulled to the network's by strength, with
    the baseline's corrections under their clusters); and the window's baseline (corrected), others, noise and
    residual. In the signal's units."""
    mask = batch["candidate_mask"].numpy()
    numbers = {key: outputs[key].float().cpu().numpy().astype(np.float64)
               for key in ("real", "log_height", "shape", "baseline", "others")}
    numbers["log_sigma"] = log_noise(outputs, batch["scaling"]).cpu().numpy().astype(np.float64)
    realness = 1 / (1 + np.exp(-numbers["real"]))
    x, weighed, valid = (batch[key].numpy().astype(np.float64) for key in ("x", "weighed", "valid"))
    candidates, scaling = batch["candidates"].numpy(), batch["scaling"].numpy()
    out, row = [], 0
    for i in range(len(mask)):
        count = int(mask[i].sum())
        shapes, row = numbers["shape"][row:row + count], row + count
        base, top = scaling[i, 0], scaling[i, 1]
        sigma = np.exp(numbers["log_sigma"][i])
        rest = numbers["baseline"][i] + numbers["others"][i]
        taken = np.flatnonzero(realness[i, :count] >= real)
        spans = clusters([dict(zip(("apex", "left", "right"), candidates[i, k, [0, 2, 3]])) for k in taken],
                         int(valid[i].sum()))
        free = corrections(spans, len(rest))
        found, covariance, amounts = heights(x[i, 0] - rest, shapes[taken], weighed[i] / np.maximum(sigma, 1e-30) ** 2,
                                             prior=np.exp(numbers["log_height"][i, taken]), strength=strength,
                                             free=free)
        corrected = numbers["baseline"][i] + amounts @ free
        sums = (shapes * valid[i]).sum(axis=1)
        peaks, fitted = [], amounts @ free
        for k in range(count):
            peak = {"apex": float(candidates[i, k, 0]), "confidence": float(realness[i, k]), "real": k in taken}
            if peak["real"]:
                j = int(np.searchsorted(taken, k))
                fitted += found[j] * shapes[k]
                peak |= {"area": float(found[j] * sums[k] * top),
                         "sigma": float(np.sqrt(max(covariance[j, j], 0.0)) * sums[k] * top) or None,
                         "shape": found[j] * shapes[k] * top, "top": int(np.argmax(shapes[k]))}
            peaks.append(peak)
        out.append({"peaks": peaks, "baseline": corrected * top + base,
                    "others": numbers["others"][i] * top, "sigma": sigma * top,
                    "residual": (x[i, 0] - rest - fitted) * top})
    return out


def deconvolve(model, signal, missing, valid, candidates, device="cpu", real: float = REAL,
               strength: float = STRENGTH) -> dict:
    """The later stage on one window (valid True in it) and its candidates (each {"apex", "confidence", "left",
    "right"}): as estimates gives it."""
    batch = collate([inputs(signal, missing, valid, candidates)])
    model.eval()
    with torch.no_grad():
        outputs = run(model, batch, device)
    return estimates(outputs, batch, real, strength)[0]
