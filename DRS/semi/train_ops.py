"""Training for the weak-strong DDV + FRG + RRS framework.

Weak view -> shared encoder + dual decoder -> stable predictions P_A^w/P_B^w
(detached pseudo targets and lesion localization).
Strong view -> DDV-M / DDV-S -> FRG -> RRS.

Only DRS is optimized. FRG evidence, RRS routes, and pseudo targets are
detached; validation and inference use one ordinary DRS forward.
"""

from __future__ import annotations

import json
import gc
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.amp import GradScaler, autocast

from DataLoading import create_segmentation_dataloaders
from loss import (
    SegmentationLoss,
    compute_boundary_loss,
    compute_deep_supervision_loss,
    compute_main_loss,
)
from models.drs import build_model
from Predict import ModelPredictor, _build_full_dataset_test_loader
from utils.checkpoint_discovery import resolve_test_checkpoints as _resolve_test_checkpoints
from semi.ddv import (
    assert_dual_decoder,
    forward_dual_srm_views,
    forward_srm_view,
    inverse_warp_srm_view,
    lesion_aware_morphology_perturbation,
    make_semantic_feature_perturbation,
)
from semi.frg import dual_decoder_reliability
from semi.rrs import cross_decoder_rrs_losses

from semi.schedules import smoothstep_rrs_weight
from utils.training_utils import (
    best_checkpoint_path,
    build_criterion,
    build_loader_config,
    build_meter,
    format_metric_summary,
    load_pretrained_checkpoint_if_needed,
    log_batch,
    prepare_run_dirs,
    print_model_info,
    resolve_unlabeled_config,
    save_checkpoint,
    save_training_curve,
    set_seed,
    unpack_batch,
    unpack_batch_views,
)
from utils.utils import _resolve_train_seeds, _write_multi_seed_summary, validate_config


@contextmanager
def freeze_batchnorm_running_stats(model: torch.nn.Module):
    """Keep training-mode batch normalization without updating running buffers."""

    batchnorm_modules = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    original_tracking = [module.track_running_stats for module in batchnorm_modules]
    try:
        for module in batchnorm_modules:
            module.track_running_stats = False
        yield
    finally:
        for module, track_running_stats in zip(batchnorm_modules, original_tracking):
            module.track_running_stats = track_running_stats


def _select(tensor, selector):
    if tensor is None or selector is None:
        return tensor
    return tensor[selector]


def _model_output(model_out):
    """Validate the DRS output tuple."""

    if not isinstance(model_out, (tuple, list)) or len(model_out) != 3:
        raise RuntimeError("DRS output must be (logits, aux_logits, boundary_logit).")
    return model_out


def compute_branch_supervised_loss(
    *,
    criterion: SegmentationLoss,
    branch_out,
    masks: torch.Tensor,
    selector: Optional[torch.Tensor],
    config: Dict[str, Any],
    boundary_weight: float,
) -> torch.Tensor:
    """Keep the baseline main + deep-supervision + boundary objective unchanged."""

    logits, aux_logits, boundary_logit = branch_out
    sup_logits = _select(logits, selector)
    sup_masks = _select(masks, selector)
    if sup_logits is None or sup_masks is None or int(sup_logits.shape[0]) == 0:
        raise ValueError("A supervised batch must contain at least one labeled sample.")
    sup_aux = [_select(aux, selector) for aux in (aux_logits or [])]
    sup_boundary = _select(boundary_logit, selector)
    main_loss = compute_main_loss(criterion, sup_logits, sup_masks)
    aux_loss, _, ds_weights = compute_deep_supervision_loss(
        criterion=criterion,
        targets=sup_masks,
        aux_logits=sup_aux,
        deep_supervision_weights=config.get("deep_supervision_weights"),
    )
    boundary_loss = compute_boundary_loss(
        sup_boundary,
        sup_masks,
        boundary_kernel_size=int(config.get("boundary_kernel_size", 3)),
    )
    return float(ds_weights[0]) * main_loss + aux_loss + float(boundary_weight) * boundary_loss


def _finite_or_raise(name: str, value: Optional[torch.Tensor]) -> None:
    if value is None:
        return
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"Non-finite tensor detected in {name}.")


SEMI_DIAGNOSTICS = (
    "image_morphology_diff_mean",
    "ddv_morphology_mean", "ddv_semantic_mean", "ddv_mean",
    "strong_image_change",
    "morphology_valid_ratio", "frg_perturbation_consistency_mean",
    "frg_consistency_m_mean", "frg_consistency_s_mean", "frg_frequency_mean",
    "frg_region_mean", "frg_reliability_mean", "frg_reliability_p10",
    "frg_reliability_p50", "frg_reliability_p90", "frg_reliability_p95",
    "frg_reliability_m_mean", "frg_reliability_s_mean",
    "frg_reliability_m_p10", "frg_reliability_m_p50", "frg_reliability_m_p90",
    "frg_reliability_m_above_0p5_ratio",
    "frg_reliability_s_p10", "frg_reliability_s_p50", "frg_reliability_s_p90",
    "frg_reliability_s_above_0p5_ratio",
    "rrs_morphology_weight_mean", "rrs_semantic_weight_mean",
    "frg_feature_spatial_var_a", "frg_feature_spatial_var_b",
    "morphology_localized_sample_ratio",
    "morphology_inverse_error_pixels", "rrs_kl_a_to_b", "rrs_kl_b_to_a", "rrs_kl_mean",
    "rrs_fg_loss", "rrs_bg_loss",
    "frg_reliability_below_0p1_ratio", "frg_reliability_above_0p5_ratio",
    "boundary_normal_disp_mean", "boundary_normal_disp_p90",
    "inward_disp_ratio", "outward_disp_ratio",
    "prototype_valid_ratio", "uncertainty_mean",
    "prototype_fg_bg_distance", "semantic_feature_change",
    "semantic_perturbation_strength", "prototype_quality",
    "semantic_uncertain_feature_change", "semantic_strength_scale",
)


def _zero_semi(reference):
    return {key: reference.new_zeros((), dtype=torch.float32)
            for key in ("loss_semi", "loss_a_to_b", "loss_b_to_a", *SEMI_DIAGNOSTICS)}


def _semi_terms(*, weak_a, weak_b, morphology, semantic, x_w, x_s, x_m,
                morphology_valid_mask, unlabeled_selector, config, morphology_warp=None,
                semantic_perturbation=None):
    if not bool(unlabeled_selector.any()):
        return _zero_semi(weak_a.logits)
    select = unlabeled_selector
    # P_A^w / P_B^w 是 weak 视图的稳定预测：既作 pseudo target，也用于定位病灶。
    pa = torch.sigmoid(weak_a.logits[select].detach().float())
    pb = torch.sigmoid(weak_b.logits[select].detach().float())
    lm = morphology.logits[select] if morphology is not None else None
    ls = semantic.logits[select] if semantic is not None else None
    pm = torch.sigmoid(lm.detach().float()) if lm is not None else None
    ps = torch.sigmoid(ls.detach().float()) if ls is not None else None
    valid = morphology_valid_mask[select] if morphology_valid_mask is not None else None
    result = _zero_semi(pa)
    result["strong_image_change"] = (x_s[select] - x_w[select]).detach().abs().mean()
    differences = []
    if lm is not None:
        dm = (pa - pm).abs()
        result["ddv_morphology_mean"] = (dm * valid).sum() / valid.sum().clamp_min(1)
        result["morphology_valid_ratio"] = valid.mean()
        result["image_morphology_diff_mean"] = (x_m[select] - x_s[select]).detach().abs().mean()
        if morphology_warp is not None:
            result["morphology_localized_sample_ratio"] = morphology_warp.localized_samples[select].mean()
            result["morphology_inverse_error_pixels"] = morphology_warp.inverse_error_pixels[select].mean()
            result["boundary_normal_disp_mean"] = morphology_warp.boundary_normal_disp_mean[select].mean()
            result["boundary_normal_disp_p90"] = morphology_warp.boundary_normal_disp_p90[select].mean()
            result["inward_disp_ratio"] = morphology_warp.inward_disp_ratio[select].mean()
            result["outward_disp_ratio"] = morphology_warp.outward_disp_ratio[select].mean()
            differences.append(result["ddv_morphology_mean"])
    if ls is not None:
        result["ddv_semantic_mean"] = (pb - ps).abs().mean()
        # DDV-S diagnostics 取自 perturbation callable 在 forward 后留下的 last_stats，
        # 因此无需改动 DRS 的 context 接口。statistics 里的量是逐样本的一维张量
        # （长度 = 整个 batch，不是 select 之后的子集），因此先按 batch 校验再取子集。
        statistics = getattr(semantic_perturbation, "last_stats", None) or {}
        batch_size = weak_a.logits.shape[0]
        for key in ("prototype_valid_ratio", "uncertainty_mean",
                    "prototype_fg_bg_distance", "semantic_feature_change",
                    "semantic_perturbation_strength", "prototype_quality",
                    "semantic_uncertain_feature_change", "semantic_strength_scale"):
            values = statistics.get(key)
            if torch.is_tensor(values) and values.flatten().numel() == batch_size:
                result[key] = values.detach().flatten()[select].mean()
        differences.append(result["ddv_semantic_mean"])
    if differences:
        result["ddv_mean"] = torch.stack(differences).mean()
    frg = dual_decoder_reliability(
        pa, pb, weak_a.decoder_feature[select], weak_b.decoder_feature[select],
        weak_a.structure_prior[select], prob_m=pm, prob_s=ps,
        morphology_valid_mask=valid,


    )
    reliability = frg["reliability"]
    reliability_m = frg["reliability_m"]
    reliability_s = frg["reliability_s"]
    for source, destination in (("consistency", "frg_perturbation_consistency_mean"),
                                ("consistency_m", "frg_consistency_m_mean"),
                                ("consistency_s", "frg_consistency_s_mean"),
                                ("frequency_reliability", "frg_frequency_mean"),
                                ("region_reliability", "frg_region_mean")):
        value = frg[source]
        if value is not None:
            result[destination] = value.mean()
    with torch.no_grad():
        for name, view in (("a", weak_a), ("b", weak_b)):
            result[f"frg_feature_spatial_var_{name}"] = (
                view.decoder_feature[select].detach().float().flatten(2).var(dim=2, unbiased=False).mean())
    result["frg_reliability_mean"] = reliability.mean()
    result["frg_reliability_m_mean"] = reliability_m.mean()
    result["frg_reliability_s_mean"] = reliability_s.mean()
    for suffix, direction in (("m", reliability_m), ("s", reliability_s)):
        for percentile in (10, 50, 90):
            result[f"frg_reliability_{suffix}_p{percentile}"] = torch.quantile(
                direction.flatten(), percentile / 100)
        result[f"frg_reliability_{suffix}_above_0p5_ratio"] = (
            direction > 0.5).float().mean()
    for percentile in (10, 50, 90, 95):
        result[f"frg_reliability_p{percentile}"] = torch.quantile(reliability.flatten(), percentile / 100)
    # 共享参考图只用于诊断；实际监督使用上面的两个方向权重。
    result["frg_reliability_below_0p1_ratio"] = (reliability < 0.1).float().mean()
    result["frg_reliability_above_0p5_ratio"] = (reliability > 0.5).float().mean()
    result.update(cross_decoder_rrs_losses(
        morphology_logit=lm, semantic_logit=ls, prob_a_weak=pa, prob_b_weak=pb,
        reliability=reliability,
        morphology_valid_mask=valid))
    if lm is not None:
        result["rrs_morphology_weight_mean"] = (reliability_m * valid).mean()
    if ls is not None:
        result["rrs_semantic_weight_mean"] = reliability_s.mean()
    for name, value in result.items():
        _finite_or_raise(name, value)
    return result

def train_one_epoch(
    *,
    model: torch.nn.Module,
    dataloader,
    criterion: SegmentationLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    config: Dict[str, Any],
    epoch: int,
    phase_name: str = "Train",
) -> Dict[str, float]:
    """Train the full DRS framework with smooth RRS loss warm-up."""

    if int(config.get("num_classes", 1)) != 1:
        raise ValueError("The new semi-supervised framework supports binary segmentation only.")
    assert_dual_decoder(model)
    device = torch.device(str(config["device"]))
    amp_enabled = bool(config.get("use_amp", True)) and device.type == "cuda"
    lambda_rrs = (
        smoothstep_rrs_weight(epoch + 1, warmup_epochs=int(config.get("rrs_warmup_epochs", 15)))

    )

    boundary_weight = float(config.get("boundary_loss_weight", 0.0))
    grad_accum_steps = max(1, int(config.get("grad_accum_steps", 1)))
    max_grad_norm = float(config.get("max_grad_norm", 0.0) or 0.0)
    total_batches = len(dataloader)
    if total_batches <= 0:
        raise RuntimeError("Training dataloader produced no batches.")

    model.train()
    meter = build_meter(config)
    diagnostic_keys = (
        "loss", "supervised_loss", "supervised_weak_loss",
        "loss_a_to_b", "loss_b_to_a", "semi_loss",
        *SEMI_DIAGNOSTICS,
    )

    sums = {key: 0.0 for key in diagnostic_keys}
    labeled_seen = 0
    log_interval = max(1, int(config.get("batch_log_interval", 20)))
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader):
        window_size = min(grad_accum_steps, total_batches - batch_idx) if batch_idx % grad_accum_steps == 0 else window_size
        x_w, x_s, masks, _, _, is_labeled = unpack_batch_views(batch)
        if x_s is None or x_s.shape != x_w.shape:
            raise RuntimeError("DRS requires aligned weak and strong image views.")
        if is_labeled is None:
            raise RuntimeError("Expected a semi-supervised batch with labeled flags.")
        batch_labeled = torch.as_tensor(
            is_labeled, device=x_w.device, dtype=torch.bool
        ).reshape(-1)
        if batch_labeled.numel() != x_w.shape[0]:
            raise ValueError("is_labeled length must equal batch size.")
        labeled_count = int(batch_labeled.sum().item())
        if labeled_count == 0:
            raise RuntimeError("Every training batch needs labeled samples for L_sup.")
        labeled_seen += labeled_count
        x_w = x_w.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()
        labeled = batch_labeled.to(device=device, non_blocking=True)
        unlabeled = ~labeled

        # Weak view: shared encoder + dual decoder -> P_A^w/P_B^w. These are
        # the stable predictions used as detached pseudo targets, reliability
        # reference, and lesion/boundary localization. The weak view carries no
        # appearance perturbation by construction.
        with autocast(device_type=device.type, enabled=amp_enabled):
            weak_a, weak_b = forward_dual_srm_views(model, x_w)
            loss_sup_weak = 0.5 * sum(
                compute_branch_supervised_loss(
                    criterion=criterion, branch_out=view.supervised_output,
                    masks=masks, selector=labeled, config=config,
                    boundary_weight=boundary_weight)
                for view in (weak_a, weak_b))
            loss_sup = loss_sup_weak
        metric_logits = 0.5 * (weak_a.logits[labeled].detach() + weak_b.logits[labeled].detach())
        # Detached weak pseudo mask locates the lesion for DDV-M, forms
        # DDV-S prototypes/uncertainty, and supplies RRS targets.
        weak_pseudo = 0.5 * (
            torch.sigmoid(weak_a.logits.detach().float())
            + torch.sigmoid(weak_b.logits.detach().float()))

        morphology = None
        morphology_warp = None
        x_m = None
        semantic = None
        semantic_perturbation = None
        with freeze_batchnorm_running_stats(model):
            morphology_warp = lesion_aware_morphology_perturbation(
                x_s,
                weak_pseudo,
            )
            x_m = morphology_warp.image
            with autocast(device_type=device.type, enabled=amp_enabled):
                morphology_raw = forward_srm_view(model, x_m, decoder_branch="a")
                morphology = inverse_warp_srm_view(
                    morphology_raw,
                    morphology_warp.inverse_grid,
                )
            semantic_perturbation = make_semantic_feature_perturbation(weak_pseudo)
            with autocast(device_type=device.type, enabled=amp_enabled):
                semantic = forward_srm_view(
                    model,
                    x_s,
                    bottleneck_perturbation=semantic_perturbation,
                    decoder_branch="b",
                )

        if morphology is None and semantic is None:
            raise RuntimeError("Semi training requires at least one active DDV branch.")

        semi = _semi_terms(
            weak_a=weak_a,
            weak_b=weak_b,
            morphology=morphology,
            semantic=semantic,
            x_w=x_w,
            x_s=x_s,
            x_m=x_m,
            morphology_valid_mask=(
                morphology_warp.valid_mask if morphology_warp is not None else None
            ),
            unlabeled_selector=unlabeled,
            config=config,
            morphology_warp=morphology_warp,
            semantic_perturbation=semantic_perturbation,
        )
        loss_semi = semi["loss_semi"]
        loss = loss_sup + lambda_rrs * loss_semi
        _finite_or_raise("total loss", loss)

        scaled_loss = loss / float(window_size)
        if scaler is None:
            scaled_loss.backward()
        else:
            scaler.scale(scaled_loss).backward()
        should_step = (batch_idx + 1) % grad_accum_steps == 0 or batch_idx + 1 == total_batches
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)

        meter.update(metric_logits, masks[labeled])
        sums["loss"] += float(loss.detach().item())
        sums["supervised_loss"] += float(loss_sup.detach().item())
        sums["supervised_weak_loss"] += float(loss_sup_weak.detach().item())
        sums["loss_a_to_b"] += float(semi["loss_a_to_b"].detach().item())
        sums["loss_b_to_a"] += float(semi["loss_b_to_a"].detach().item())
        sums["semi_loss"] += float(loss_semi.detach().item())
        for key in SEMI_DIAGNOSTICS:
            sums[key] += float(semi[key].detach().item())

        if (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == total_batches:
            running = meter.compute()
            log_batch(
                phase_name,
                batch_idx,
                total_batches,
                {
                    **running,
                    "loss": sums["loss"] / float(batch_idx + 1),
                    "objective_loss": sums["loss"] / float(batch_idx + 1),
                    "supervised_loss": sums["supervised_loss"] / float(batch_idx + 1),
                    "supervised_weak_loss": sums["supervised_weak_loss"] / float(batch_idx + 1),
                },
            )

    metrics = meter.compute()
    metrics.update({key: value / float(total_batches) for key, value in sums.items()})
    metrics.update(
        sup_weak=metrics["supervised_weak_loss"],
        sup_total=metrics["supervised_loss"],
    )
    metrics["ddv_semantic_ratio"] = metrics["ddv_semantic_mean"] / (
        metrics["ddv_morphology_mean"] + 1e-12
    )
    metrics.update(
        lambda_RRS=lambda_rrs,
        labeled_samples=float(labeled_seen),
    )
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def validate_one_epoch(
    *,
    model: torch.nn.Module,
    dataloader,
    criterion: SegmentationLoss,
    config: Dict[str, Any],
    epoch: int,
    phase_name: str = "Val",
) -> Dict[str, Any]:
    """Validate with one normal model forward and no training-only memory."""

    del epoch
    assert_dual_decoder(model)
    device = torch.device(str(config["device"]))
    amp_enabled = bool(config.get("use_amp", True)) and device.type == "cuda"
    boundary_weight = float(config.get("boundary_loss_weight", 0.0))
    meter = build_meter(config)
    total_loss = 0.0
    sample_count = 0
    if len(dataloader) <= 0:
        raise RuntimeError("Validation dataloader produced no batches.")

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, _, masks, _, _, _ = unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()
            with autocast(device_type=device.type, enabled=amp_enabled):
                branch = _model_output(model(inputs))
                loss = compute_branch_supervised_loss(
                    criterion=criterion,
                    branch_out=branch,
                    masks=masks,
                    selector=None,
                    config=config,
                    boundary_weight=boundary_weight,
                )
            batch_size = int(inputs.shape[0])
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            meter.update(branch[0], masks)

    metrics: Dict[str, Any] = meter.compute()
    metrics["loss"] = total_loss / float(max(1, sample_count))
    print(format_metric_summary(phase_name, metrics))
    return metrics


def _load_resume_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    config: Dict[str, Any],
    train_batch_sampler=None,
) -> tuple[int, float, int, Dict[str, float], Optional[Path]]:
    """Restore the model, optimizer, scheduler, AMP, and sampler state."""

    path_text = str(config.get("resume_checkpoint_path") or "").strip()
    if not path_text:
        return 0, -1.0, 0, {}, None
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"resume_checkpoint_path does not exist: {path}")
    checkpoint = torch.load(path, map_location=config["device"], weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Resume checkpoint must contain a training state_dict.")
    saved_config = dict(checkpoint.get("config") or {})
    try:
        model.load_state_dict(
            checkpoint["state_dict"],
            strict=bool(config.get("checkpoint_strict_load", True)),
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Resume checkpoint is incompatible with the shared-encoder dual-decoder model."
        ) from exc
    if saved_config.get("method_name") != "DRS":
        raise ValueError("Resume requires a DRS training checkpoint; use pretrained_checkpoint_path to initialize weights from an older model.")

    critical_defaults: Dict[str, float] = {
        "rrs_warmup_epochs": 15.0,
    }
    for name, default in critical_defaults.items():
        saved_value = float(saved_config.get(name, default))
        current_value = float(config.get(name, default))
        if saved_value != current_value:
            raise ValueError(
                f"Resume hyperparameter mismatch for {name}: "
                f"saved={saved_value!r}, current={current_value!r}."
            )
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("grad_scaler") is not None:
        scaler.load_state_dict(checkpoint["grad_scaler"])
    saved_sampler_state = checkpoint.get("batch_sampler_state")
    if saved_sampler_state is not None:
        loader = getattr(train_batch_sampler, "load_state_dict", None)
        if not callable(loader):
            raise ValueError(
                "Resume checkpoint contains two-stream sampler state, but the "
                "current train loader has no compatible stateful sampler."
            )
        loader(saved_sampler_state)
    elif callable(getattr(train_batch_sampler, "state_dict", None)):
        print("Warning: resume checkpoint has no batch_sampler_state; stream order restarts.")

    start_epoch = int(checkpoint.get("epoch", 0))
    best_dice = float(checkpoint.get("best_dice", -1.0))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    module_metrics = {
        str(key): float(value)
        for key, value in dict(checkpoint.get("best_module_metrics") or {}).items()
        if key in (*SEMI_DIAGNOSTICS, "loss_a_to_b", "loss_b_to_a",
                   "semi_loss", "lambda_RRS")
        and isinstance(value, (int, float))
    }
    print(f"Resumed training state from: {path} (next epoch={start_epoch + 1})")
    return start_epoch, best_dice, best_epoch, module_metrics, path


def _history_template() -> Dict[str, list[Any]]:
    names = ("loss", "supervised_loss", "supervised_weak_loss",
             "sup_weak", "sup_strong", "sup_total",
             "loss_a_to_b", "loss_b_to_a", "semi_loss",
             "lambda_RRS", *SEMI_DIAGNOSTICS,
             "dice", "miou", "pre_percent", "accuracy", "sens_percent", "spe_percent")
    history = {f"train_{name}": [] for name in names}
    history.update({f"val_{name}": [] for name in
                    ("loss", "dice", "miou", "pre_percent", "accuracy", "sens_percent", "spe_percent")})
    return history

def _append_history(
    history: Dict[str, list[Any]],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, Any],
) -> None:
    mapping = {
        "train_loss": (train_metrics, "loss"),
        "train_supervised_loss": (train_metrics, "supervised_loss"),
        "train_loss_a_to_b": (train_metrics, "loss_a_to_b"),
        "train_loss_b_to_a": (train_metrics, "loss_b_to_a"),
        "train_semi_loss": (train_metrics, "semi_loss"),
        "val_loss": (val_metrics, "loss"),
        "train_dice": (train_metrics, "dice"),
        "val_dice": (val_metrics, "dice"),
        "train_miou": (train_metrics, "miou"),
        "val_miou": (val_metrics, "miou"),
        "train_pre_percent": (train_metrics, "precision"),
        "val_pre_percent": (val_metrics, "precision"),
        "train_accuracy": (train_metrics, "accuracy"),
        "val_accuracy": (val_metrics, "accuracy"),
        "train_sens_percent": (train_metrics, "sens_percent"),
        "val_sens_percent": (val_metrics, "sens_percent"),
        "train_spe_percent": (train_metrics, "spe_percent"),
        "val_spe_percent": (val_metrics, "spe_percent"),
        "train_lambda_RRS": (train_metrics, "lambda_RRS"),
    }
    for history_key, (source, metric_key) in mapping.items():
        history[history_key].append(float(source.get(metric_key, 0.0)))
    for history_key in history:
        if history_key in mapping or not history_key.startswith("train_"):
            continue
        metric_key = history_key[len("train_") :]
        history[history_key].append(float(train_metrics.get(metric_key, 0.0)))


def format_epoch_module_summary(config, metrics):
    value = lambda key: float(metrics.get(key, 0))
    return [
        f"[DRS] loss={value('loss'):.4f} supervised={value('supervised_loss'):.4f} "
        f"RRS={value('semi_loss'):.4f} warmup={value('lambda_RRS'):.3f}",
        f"[DDV] morphology={value('ddv_morphology_mean'):.4f} "
        f"semantic={value('ddv_semantic_mean'):.4f} valid={value('morphology_valid_ratio'):.4f}",
        f"[FRG] C_P={value('frg_perturbation_consistency_mean'):.4f} "
        f"R_F={value('frg_frequency_mean'):.4f} R_R={value('frg_region_mean'):.4f} "
        f"R={value('frg_reliability_mean'):.4f}",
        f"[RRS] A_to_B={value('loss_a_to_b'):.4f} B_to_A={value('loss_b_to_a'):.4f} "
        f"foreground={value('rrs_fg_loss'):.4f} background={value('rrs_bg_loss'):.4f}",
    ]


def _run_test_only_experiment(
    base_config: Dict[str, Any], train_seed: int, batch_tag: str,
) -> Dict[str, Any]:
    """Use a source-trained model on every target sample, without training splits."""
    checkpoint_path = _resolve_test_checkpoints(base_config, [train_seed])[train_seed]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_config = dict(checkpoint.get("config") or {})
    source_seed = source_config.get("seed", source_config.get("train_seed"))
    if source_seed is not None and int(source_seed) != train_seed:
        raise ValueError(f"Seed {train_seed} was assigned a seed {source_seed} checkpoint: {checkpoint_path}")
    config = {**base_config, **source_config}
    for key in (
        "image_path", "mask_path", "mask_suffixes", "dataset_name", "results_root",
        "device", "batch_size", "num_workers", "batch_log_interval", "verbose",
        "threshold", "metric_include_background", "ignore_index", "mask_threshold",
        "max_test_visualizations", "fixed_test_visualization_json", "predict_artifact_prefix",
        "train_seeds", "test_checkpoint_paths", "test_checkpoint_path", "test_checkpoint_dir",
        "test_checkpoint_search_dir", "test_normalization",
    ):
        if key in base_config:
            config[key] = base_config[key]
    normalization = str(base_config.get("test_normalization", "config")).lower()
    if normalization not in {"checkpoint", "config"}:
        raise ValueError("test_normalization must be 'checkpoint' or 'config'.")
    if normalization == "config":
        for key in ("normalize", "mean", "std"):
            if key in base_config:
                config[key] = base_config[key]
    if "fixed_test_visualization_json" not in base_config:
        config["fixed_test_visualization_json"] = ""
    config.update(
        seed=int(train_seed), train_seed=int(train_seed), test_only=True,
        predict_full_dataset=True, evaluation_mode="full_dataset_test",
        val_split=0.0, test_split=0.0, unlabeled_ratio=0.0,
        requested_unlabeled_ratio=0.0, effective_unlabeled_ratio=0.0,
        use_two_stream_batch_sampler=False,
        test_normalization=normalization, test_source_checkpoint=str(checkpoint_path),
        test_source_seed=source_seed, test_source_image_path=source_config.get("image_path"),
    )
    validate_config(config)
    set_seed(train_seed)
    test_loader = _build_full_dataset_test_loader(config)
    config["test_total_samples"] = len(test_loader.dataset)
    prepare_run_dirs(config, batch_tag=batch_tag)
    best_dice = checkpoint.get("best_dice")
    best_epoch = int(checkpoint.get("epoch", 0))
    del checkpoint
    print(f"Test-only: source seed={train_seed}, checkpoint={checkpoint_path}")
    print(f"Target dataset: {config['image_path']} (all {config['test_total_samples']} samples)")
    print(f"Normalization [{normalization}]: mean={config.get('mean')}, std={config.get('std')}")
    predictor = ModelPredictor(args=config, checkpoint_path=str(checkpoint_path), test_loader=test_loader)
    return {
        "train_seed": train_seed, "run_name": config["run_name"], "run_root": config["run_root"],
        "best_model_path": str(checkpoint_path),
        "best_val_dice": float(best_dice) if best_dice is not None else float("nan"),
        "best_epoch": best_epoch, "best_module_metrics": {},
        "evaluation_mode": "full_dataset_test", "test_metrics": predictor.run(),
    }


def _run_single_seed_experiment(
    base_config: Dict[str, Any],
    train_seed: int,
    batch_tag: str,
) -> Dict[str, Any]:
    """Build data and one DRS for a complete seed run."""

    if bool(base_config.get("test_only", False)):
        return _run_test_only_experiment(base_config, train_seed, batch_tag)
    config = dict(base_config)
    config.update(
        seed=int(train_seed),
        train_seed=int(train_seed),
        
    )
    resolve_unlabeled_config(config)
    set_seed(int(train_seed))
    prepare_run_dirs(config, batch_tag=batch_tag)
    train_loader, val_loader, test_loader, dataset_info = create_segmentation_dataloaders(
        image_path=config["image_path"],
        mask_path=config["mask_path"],
        batch_size=int(config["batch_size"]),
        img_size=int(config["img_size"]),
        normalize=bool(config["normalize"]),
        num_workers=int(config["num_workers"]),
        seed=int(config["split_seed"]),
        val_split=float(config["val_split"]),
        test_split=float(config["test_split"]),
        config=build_loader_config(config),
        
    )
    config["num_classes"] = int(dataset_info["num_classes"])
    config["class_names"] = list(dataset_info["class_names"])
    target_best_path = best_checkpoint_path(config)

    sampler = str(dataset_info.get("train_batch_sampler", "random_shuffle"))
    if sampler == "two_stream_balanced":
        sampler = (
            f"{int(dataset_info['train_labeled_batch_size'])}L+"
            f"{int(dataset_info['train_unlabeled_batch_size'])}U"
        )
    print(f"Method: {config.get('method_name')} | seed={train_seed}")
    print(
        f"Data: train={int(dataset_info.get('train_total_samples', 0))} "
        f"(L={int(dataset_info.get('train_labeled_samples', 0))}, "
        f"U={int(dataset_info.get('train_unlabeled_samples', 0))}) "
        f"val={int(dataset_info.get('val_total_samples', 0))} "
        f"test={int(dataset_info.get('test_total_samples', 0))} sampler={sampler}"
    )
    print("Architecture: one shared SRM encoder + two independent decoders; prediction=mean_logits.")
    print("DRS: DDV-M + DDV-S / FRG / reciprocal RRS; GT=weak; prediction=mean_logits.")

    def experiment_result(
        *,
        model_path: Path,
        best_val_dice: float,
        best_epoch: int,
        test_metrics: Dict[str, Any],
        module_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        return {
            "train_seed": int(train_seed),
            "run_name": str(config["run_name"]),
            "run_root": str(config["run_root"]),
            "best_model_path": str(model_path),
            "best_val_dice": float(best_val_dice),
            "best_epoch": int(best_epoch),
            "best_module_metrics": dict(module_metrics or {}),
            "test_metrics": test_metrics,
        }

    model = build_model(config).to(config["device"])
    assert_dual_decoder(model)
    print_model_info(model)
    criterion = build_criterion(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["initial_lr"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["num_epochs"])),
    )
    scaler = GradScaler(
        "cuda",
        enabled=str(config["device"]).startswith("cuda") and bool(config.get("use_amp", True)),
    )

    resume_requested = bool(str(config.get("resume_checkpoint_path") or "").strip())
    if not resume_requested:
        load_pretrained_checkpoint_if_needed(model, config)
    (
        start_epoch,
        best_dice,
        best_epoch,
        best_module_metrics,
        resume_path,
    ) = _load_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        train_batch_sampler=getattr(train_loader, "batch_sampler", None),
    )
    best_model_source = resume_path
    history = _history_template()

    for epoch in range(start_epoch, int(config["num_epochs"])):
        print(f"Epoch {epoch + 1}/{int(config['num_epochs'])}")
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            epoch=epoch,
        )
        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            config=config,
            epoch=epoch,
        )
        scheduler.step()
        _append_history(history, train_metrics, val_metrics)

        current_dice = float(val_metrics["dice"])
        improved = current_dice > best_dice
        if improved:
            best_dice = current_dice
            best_epoch = epoch + 1
            best_model_source = target_best_path
            best_module_metrics = {
                key: float(train_metrics[key])
                for key in (*SEMI_DIAGNOSTICS, "loss_a_to_b", "loss_b_to_a",
                            "semi_loss", "lambda_RRS")
                if key in train_metrics
            }

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            best_dice=best_dice,
            val_metrics=val_metrics,
            is_best=improved,
            extra_state={
                "best_epoch": int(best_epoch),
                "best_module_metrics": dict(best_module_metrics),
                "module_training_config": {
                    name: config.get(name)
                    for name in ("rrs_warmup_epochs",)
                },
                "grad_scaler": scaler.state_dict(),
                "batch_sampler_state": (
                    train_loader.batch_sampler.state_dict()
                    if callable(getattr(train_loader.batch_sampler, "state_dict", None))
                    else None
                ),
            },
        )
        print("Epoch module summary:")
        for summary_line in format_epoch_module_summary(config, train_metrics):
            print(summary_line)
        print(
            f"[Validation] Dice={current_dice:.4f} best_Dice={best_dice:.4f}"
            + (f" * saved={target_best_path}" if improved else "")
        )

    history_path = Path(config["save_dir_png"]) / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)
    if bool(config.get("save_training_curve", True)) and history["train_loss"]:
        save_training_curve(history, config["save_dir_png"])
    if best_model_source is None or not best_model_source.is_file():
        raise RuntimeError("Training finished without a usable checkpoint.")
    print(
        f"Best model | epoch={best_epoch} Dice={best_dice:.4f} "
        f"checkpoint={os.path.abspath(str(best_model_source))}"
    )
    # ModelPredictor intentionally rebuilds the inference model from the saved
    # checkpoint. Release the training copy first so final testing cannot hold
    # two DRSs on the GPU at once.
    del model, optimizer, scheduler, scaler
    gc.collect()
    if str(config["device"]).startswith("cuda"):
        torch.cuda.empty_cache()
    predictor = ModelPredictor(
        args=config,
        checkpoint_path=str(best_model_source),
        test_loader=test_loader,
    )
    return experiment_result(
        model_path=best_model_source,
        best_val_dice=best_dice,
        best_epoch=best_epoch,
        module_metrics=best_module_metrics,
        test_metrics=predictor.run(),
    )


def run_multi_seed_experiments(config: Dict[str, Any]) -> str:
    """Run configured seeds and write the existing mean/std report."""

    config = dict(config)
    if not bool(config.get("test_only", False)):
        validate_config(config)
    train_seeds = _resolve_train_seeds(config)
    if bool(config.get("test_only", False)):
        checkpoints = _resolve_test_checkpoints(config, train_seeds)
        if config.get("test_checkpoint_dir"):
            config["test_checkpoint_search_dir"] = config.pop("test_checkpoint_dir")
        config.pop("test_checkpoint_path", None)
        config["test_checkpoint_paths"] = {seed: str(path) for seed, path in checkpoints.items()}
    batch_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = [
        _run_single_seed_experiment(config, seed, batch_tag)
        for seed in train_seeds
    ]
    summary_path = _write_multi_seed_summary(
        model_type=str(config.get("model_type", config.get("method_name", "model"))),
        batch_tag=batch_tag,
        results_root=str(config.get("results_root") or "results"),
        run_summaries=summaries,
    )
    print(f"Saved multi-seed summary to: {os.path.abspath(summary_path)}")
    return summary_path


__all__ = [
    "_model_output",
    "compute_branch_supervised_loss",
    "freeze_batchnorm_running_stats",
    "format_epoch_module_summary",
    "run_multi_seed_experiments",
    "train_one_epoch",
    "validate_one_epoch",
]
