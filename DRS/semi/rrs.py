"""Reliability-weighted cross supervision with balanced soft foreground/background."""
from __future__ import annotations
import torch
import torch.nn.functional as F

RRS_EPS = 1e-6


def reliability_weighted_consistency(student_logit, pseudo_prob, reliability, *,
                                     sample_mask=None, valid_mask=None, return_kl=False,
                                     return_components=False):
    if not (student_logit.shape == pseudo_prob.shape == reliability.shape):
        raise ValueError("RRS logits, targets and reliability must have identical shapes.")
    target = pseudo_prob.detach().float()
    bce = F.binary_cross_entropy_with_logits(student_logit.float(), target, reduction="none")
    weight = reliability.detach().float().clamp(0, 1)
    eligible = torch.ones_like(weight)
    if sample_mask is not None:
        mask = torch.as_tensor(sample_mask, device=weight.device, dtype=torch.bool)
        if mask.ndim != 1 or mask.numel() != weight.shape[0]:
            raise ValueError("sample_mask must be a length-B vector.")
        eligible = eligible * mask.view(-1, 1, 1, 1)
    if valid_mask is not None:
        if valid_mask.shape != weight.shape:
            raise ValueError("valid_mask must match the logit shape.")
        eligible = eligible * valid_mask.detach().float().clamp(0, 1)
    fg = eligible * target
    bg = eligible * (1 - target)
    fg_mass, bg_mass = fg.sum(), bg.sum()
    # Class existence is determined by eligible pseudo-label decisions only;
    # the BCE weighting and denominators remain fully soft.
    fg_present = ((target >= 0.5) & (eligible > 0)).any()
    bg_present = ((target < 0.5) & (eligible > 0)).any()
    fg_denominator = torch.where(fg_mass == 0, torch.ones_like(fg_mass), fg_mass)
    bg_denominator = torch.where(bg_mass == 0, torch.ones_like(bg_mass), bg_mass)
    component_count = (fg_present.to(bce.dtype) + bg_present.to(bce.dtype)).clamp_min(1)
    fg_loss = torch.where(fg_present, (bce * weight * fg).sum() / fg_denominator,
                          bce.sum() * 0)
    bg_loss = torch.where(bg_present, (bce * weight * bg).sum() / bg_denominator,
                          bce.sum() * 0)
    loss = (fg_loss + bg_loss) / component_count
    if not (return_kl or return_components):
        return loss
    # KL is logging-only and uses the same soft fg/bg masses as the training loss.
    with torch.no_grad():
        entropy = -(target * target.clamp_min(RRS_EPS).log()
                    + (1 - target) * (1 - target).clamp_min(RRS_EPS).log())
        kl_pixel = (bce.detach() - entropy).clamp_min(0)
        kl_fg = torch.where(fg_present, (kl_pixel * weight * fg).sum() / fg_denominator,
                             kl_pixel.sum() * 0)
        kl_bg = torch.where(bg_present, (kl_pixel * weight * bg).sum() / bg_denominator,
                             kl_pixel.sum() * 0)
        kl = (kl_fg + kl_bg) / component_count
    if return_components:
        return loss, kl, fg_loss.detach(), bg_loss.detach(), bool(fg_present), bool(bg_present)
    return loss, kl


def cross_decoder_rrs_losses(*, morphology_logit, semantic_logit,
                              prob_a_weak, prob_b_weak, reliability,
                              morphology_valid_mask=None):
    """Cross-supervise both DDV students with detached weak targets and shared reliability."""
    if morphology_logit is None or semantic_logit is None:
        raise ValueError("DRS requires both morphology and semantic student views.")
    zero = prob_a_weak.new_zeros(())
    target_m = prob_b_weak
    target_s = prob_a_weak
    weight_m = reliability
    weight_s = reliability
    loss_m_key = "loss_b_to_a"
    loss_s_key = "loss_a_to_b"
    kl_m_key = "rrs_kl_b_to_a"
    kl_s_key = "rrs_kl_a_to_b"
    kl_mean_key = "rrs_kl_mean"
    loss_m, loss_s = zero, zero
    kl_m, kl_s = zero, zero
    losses = []
    divergences = []
    fg_losses, bg_losses = [], []
    loss_m, kl_m, fg_m, bg_m, has_fg_m, has_bg_m = reliability_weighted_consistency(
        morphology_logit, target_m, weight_m, valid_mask=morphology_valid_mask,
        return_kl=True, return_components=True)
    morphology_has_evidence = (
        morphology_valid_mask is None
        or bool((morphology_valid_mask.detach() > 0).any().item())
    )
    if morphology_has_evidence:
        losses.append(loss_m)
        divergences.append(kl_m)
        if has_fg_m:
            fg_losses.append(fg_m)
        if has_bg_m:
            bg_losses.append(bg_m)
    loss_s, kl_s, fg_s, bg_s, has_fg_s, has_bg_s = reliability_weighted_consistency(
        semantic_logit, target_s, weight_s, return_kl=True, return_components=True)
    losses.append(loss_s)
    divergences.append(kl_s)
    if has_fg_s:
        fg_losses.append(fg_s)
    if has_bg_s:
        bg_losses.append(bg_s)
    if not losses:
        graph_zero = loss_m + loss_s
        return {loss_m_key: loss_m, loss_s_key: loss_s,
                "loss_semi": graph_zero, kl_s_key: kl_s,
                kl_m_key: kl_m, kl_mean_key: zero,
                "rrs_fg_loss": zero, "rrs_bg_loss": zero}
    return {loss_m_key: loss_m, loss_s_key: loss_s,
            "loss_semi": torch.stack(losses).mean(),
            kl_s_key: kl_s, kl_m_key: kl_m,
            kl_mean_key: torch.stack(divergences).mean(),
            "rrs_fg_loss": torch.stack(fg_losses).mean() if fg_losses else zero,
            "rrs_bg_loss": torch.stack(bg_losses).mean() if bg_losses else zero}
