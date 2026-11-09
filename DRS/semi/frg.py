"""Shared frequency-region-response reliability for the two RRS directions.

Radial FFT cutoff 0.25, k_low=k_high=4, and region-relation grid 8 are fixed.
Evidence is detached and computed in FP32.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

FRG_CUTOFF_RATIO = 0.25
FRG_K_LOW = 4.0
FRG_K_HIGH = 4.0
FRG_REGION_GRID = 8
FRG_EPS = 1e-6


def _check_pair(a, b):
    if a.ndim != 4 or a.shape != b.shape:
        raise ValueError("FRG inputs must have identical BCHW shapes.")


def _freq_decompose(prob, lowpass):
    low = torch.fft.ifft2(torch.fft.fft2(prob) * lowpass).real
    return low, prob - low


@torch.no_grad()
def frequency_reliability(prob_a, prob_b, structure_prior=None):
    _check_pair(prob_a, prob_b)
    with torch.autocast(device_type=prob_a.device.type, enabled=False):
        a, b = prob_a.detach().float(), prob_b.detach().float()
        h, w = a.shape[-2:]
        fy = torch.fft.fftfreq(h, device=a.device).view(h, 1)
        fx = torch.fft.fftfreq(w, device=a.device).view(1, w)
        lowpass = (torch.sqrt(fy * fy + fx * fx) <= 0.5 * FRG_CUTOFF_RATIO).float()
        low_a, high_a = _freq_decompose(a, lowpass.view(1, 1, h, w))
        low_b, high_b = _freq_decompose(b, lowpass.view(1, 1, h, w))
        body = torch.exp(-FRG_K_LOW * (low_a - low_b).abs())
        edge = torch.exp(-FRG_K_HIGH * (high_a - high_b).abs())
        if structure_prior is None:
            structure = torch.ones_like(a)
        else:
            structure = structure_prior.detach().float()
            if structure.ndim == 3:
                structure = structure.unsqueeze(1)
            if structure.shape[-2:] != (h, w):
                structure = F.interpolate(structure, (h, w), mode="bilinear", align_corners=False)
            structure = structure.clamp(0, 1)
        return (body * (1 - structure * (1 - edge))).clamp(0, 1)


@torch.no_grad()
def region_reliability(feat_a, feat_b, *, out_hw=None):
    _check_pair(feat_a, feat_b)
    with torch.autocast(device_type=feat_a.device.type, enabled=False):
        a, b = feat_a.detach().float(), feat_b.detach().float()
        grid = FRG_REGION_GRID
        da = F.normalize(F.adaptive_avg_pool2d(a, (grid, grid)).flatten(2).transpose(1, 2), dim=2, eps=FRG_EPS)
        db = F.normalize(F.adaptive_avg_pool2d(b, (grid, grid)).flatten(2).transpose(1, 2), dim=2, eps=FRG_EPS)
        ra, rb = torch.bmm(da, da.transpose(1, 2)), torch.bmm(db, db.transpose(1, 2))
        relation = (1 - (ra - rb).abs().mean(dim=2)).clamp(0, 1).view(a.shape[0], 1, grid, grid)
        return F.interpolate(relation, size=out_hw or a.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)


@torch.no_grad()
def dual_decoder_reliability(prob_a, prob_b, feat_a, feat_b, structure_prior, *,
                             prob_m, prob_s, morphology_valid_mask=None):
    """Compute detached shared reliability R = C_P * R_F * R_R for DRS."""
    _check_pair(prob_a, prob_b)
    _check_pair(prob_a, prob_m)
    _check_pair(prob_b, prob_s)
    cm = (1 - (prob_a.detach().float() - prob_m.detach().float()).abs()).clamp(0, 1)
    cs = (1 - (prob_b.detach().float() - prob_s.detach().float()).abs()).clamp(0, 1)
    valid = torch.ones_like(cm)
    if morphology_valid_mask is not None:
        _check_pair(prob_a, morphology_valid_mask)
        valid = morphology_valid_mask.detach().to(device=prob_a.device, dtype=torch.float32)
        if not bool(torch.isfinite(valid).all()):
            raise ValueError("morphology_valid_mask contains NaN or Inf.")
        valid = valid.clamp(0, 1)
    cp = ((valid * cm + cs) / (valid + 1)).clamp(0, 1)
    rf = frequency_reliability(prob_a, prob_b, structure_prior)
    rr = region_reliability(feat_a, feat_b, out_hw=prob_a.shape[-2:])
    reliability = (cp * rf * rr).float().clamp(0, 1).detach()
    outputs = {"consistency": cp, "consistency_m": cm, "consistency_s": cs,
               "frequency_reliability": rf, "region_reliability": rr,
               "reliability": reliability,
               "reliability_m": reliability, "reliability_s": reliability}
    if any(not bool(torch.isfinite(v).all()) for v in outputs.values()):
        raise FloatingPointError("Non-finite FRG evidence.")
    return outputs
