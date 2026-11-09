"""Single smooth RRS loss warm-up for DDV + FRG + RRS training."""

from __future__ import annotations


def smoothstep_rrs_weight(epoch: int, *, warmup_epochs: int = 15) -> float:
    """Return the sole training ramp ``3p^2-2p^3`` with one-based ``epoch``.

    The first epoch is deliberately small but non-zero, and the weight reaches
    exactly one at ``warmup_epochs``. DDV, FRG and cross-decoder RRS are otherwise
    active throughout semi-supervised training.
    """

    epoch = int(epoch)
    warmup_epochs = int(warmup_epochs)
    if epoch < 1:
        raise ValueError("epoch is one-based and must be positive.")
    if warmup_epochs < 1:
        raise ValueError("warmup_epochs must be positive.")
    progress = min(1.0, float(epoch) / float(warmup_epochs))
    return progress * progress * (3.0 - 2.0 * progress)


__all__ = ["smoothstep_rrs_weight"]
