"""The later stage's losses (SPEC_deconvolution.md 4), in the linear channel's units (deconvolve_data.py):

    shape:    of each candidate matched to a peak with a shape, the L1 of its shape and the peak's (both of top 1) on
              the window, over the peak's sum
    real:     binary cross entropy of the realness, of the candidates scored
    height:   L1 of the log height, of those matched
    baseline, others: log(1 + |error| / the noise), where the noise is known: of what matters to the least squares,
              errors of the noise's size, as much in a window of a high S/N as of a low
    sigma:    L1 of the log noise, where it is known
    area:     of those matched, log(1 + |relative error|) of the areas of the heights by least squares (solve): of
              their shapes, to the signal less the baseline and the others given, weighed by the noise given, with
              the baseline's corrections under their clusters (deconvolve.corrections). What the network learns of
              the shapes is what gives the right areas so
"""

import torch
import torch.nn.functional as F

from deconvolve_models import STRENGTH, log_noise

DEFAULT_WEIGHTS = {"shape": 1.0, "real": 1.0, "height": 0.1, "baseline": 0.2, "others": 0.2, "sigma": 0.1,
                   "area": 1.0}
ROUNDS = 4  # of solving again without those below 0


def corrections(spans, span_mask, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """deconvolve.corrections of spans (B, C, 2) of span_mask (B, C): (B, 2C, N), and their mask (B, 2C)."""
    t = torch.arange(length, device=spans.device, dtype=torch.float32)
    low, high = spans[..., :1], spans[..., 1:]
    inside = ((t >= low) & (t <= high) & span_mask[..., None]).float()
    line = inside * (t - (low + high) / 2) / ((high - low) / 2).clamp(min=1.0)
    return torch.stack([inside, line], 2).flatten(1, 2), span_mask.repeat_interleave(2, 1)


def solve(residual, shapes, weights, include, prior, strength: float = STRENGTH, free=None,
          free_mask=None) -> torch.Tensor:
    """The heights (B, K) of shapes (B, K, N, each of top 1) of those include (B, K) that make up residual (B, N)
    best, by least squares weighted by weights (B, N), pulled to prior (B, K) by strength x what the data tell of each
    alone, the rest 0; with free (B, M, N) of free_mask (B, M), of any sign, fitted with them: as deconvolve.heights.
    Those below 0 left out and the rest solved again, ROUNDS times at most, and any still below held at 0: Lawson
    and Hanson's least squares of a >= 0, but for one coming back once out, which is seldom. Of gradients through
    the solution, to the shapes, the residual and the weights."""
    count = shapes.shape[1]
    columns = shapes if free is None else torch.cat([shapes, free], 1)
    weights = weights / (weights.sum(-1, keepdim=True) / weights.shape[-1]).clamp(min=1e-30)  # of no matter, but range
    gram = torch.einsum("bkn,bn,bjn->bkj", columns, weights, columns).double()
    target = torch.einsum("bkn,bn,bn->bk", columns, weights, residual).double()
    diagonal = torch.diagonal(gram, dim1=1, dim2=2)
    pull = torch.full_like(diagonal, 1e-6)  # of free, only to keep it solvable
    pull[:, :count] = strength
    gram = gram + torch.diag_embed(pull * diagonal)
    target = target + pull * diagonal * F.pad(prior.double(), (0, columns.shape[1] - count))
    keep = torch.cat([include, include[:, :0] if free is None else free_mask], 1) & (diagonal > 0)
    bounded = torch.arange(columns.shape[1], device=keep.device) < count
    eye = torch.eye(columns.shape[1], dtype=gram.dtype, device=gram.device).expand_as(gram)
    for _ in range(ROUNDS):
        both = keep[:, :, None] & keep[:, None, :]
        found = torch.linalg.solve(torch.where(both, gram, eye), torch.where(keep, target, 0.0).unsqueeze(-1))[..., 0]
        below = keep & bounded & (found < 0)
        if not below.any():
            break
        keep = keep & ~below
    return torch.where(keep, found, 0.0)[:, :count].clamp(min=0).float()


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def losses(outputs: dict, batch: dict, weights: dict = DEFAULT_WEIGHTS, strength: float = STRENGTH) -> dict:
    """Each loss, and their weighted sum as "total"."""
    mask = batch["candidate_mask"]
    matched = batch["matched"] * mask
    valid = batch["valid"]
    rows = torch.arange(len(mask), device=mask.device)[:, None].expand_as(mask)[mask]
    shape, truth = outputs["shape"], batch["shape"]
    values = {}
    errors = ((shape - truth).abs() * valid[rows]).sum(-1) / truth.sum(-1).clamp(min=1e-12)
    values["shape"] = masked_mean(errors, matched[mask])
    real = F.binary_cross_entropy_with_logits(outputs["real"], batch["real"], reduction="none")
    values["real"] = masked_mean(real, batch["real_weight"] * mask)
    values["height"] = masked_mean((outputs["log_height"] - batch["height"].clamp(min=1e-30).log()).abs(), matched)
    known, noise = batch["sigma_mask"], batch["sigma"].clamp(min=1e-30)
    for name in ("baseline", "others"):
        values[name] = masked_mean(torch.log1p((outputs[name] - batch[name]).abs() / noise), known)
    values["sigma"] = masked_mean((outputs["log_sigma"] - noise.log()).abs(), known)

    shapes = shape.new_zeros(mask.shape + shape.shape[-1:])
    shapes[mask] = shape
    residual = batch["x"][:, 0] - outputs["baseline"] - outputs["others"]
    log_sigma = log_noise(outputs, batch["scaling"]).detach()
    weighed = batch["weighed"] * torch.exp(-2 * (log_sigma - log_sigma.amin(-1, keepdim=True)))  # up to 1, of range
    prior = outputs["log_height"].detach().clamp(-30, 10).exp()
    free, free_mask = corrections(batch["spans"], batch["span_mask"], shapes.shape[-1])
    solved = solve(residual, shapes, weighed, matched > 0, prior, strength, free, free_mask)
    area = solved * (shapes * valid[:, None]).sum(-1)
    scored = matched * (batch["area"] > 0)
    relative = (area - batch["area"]) / batch["area"].clamp(min=1e-30)
    values["area"] = masked_mean(torch.log1p(relative.abs()), scored)
    values["total"] = sum(weights.get(name, 0.0) * value for name, value in values.items())
    return values
