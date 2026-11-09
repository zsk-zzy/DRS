"""项目通用工具函数。

当前文件主要负责：
1. 目录与运行配置的保存
2. checkpoint 的保存与加载
3. 训练结果可视化
4. 测试阶段恢复最佳模型
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
from datetime import datetime
from statistics import fmean, stdev
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from loss import SegmentationLoss
from metric.segmentation_metrics import SegmentationMetricMeter
from models.drs import build_model


def ensure_dir(path: str) -> str:
    """确保目录存在，不存在就自动创建。"""

    os.makedirs(path, exist_ok=True)
    return path


def _validate_primary_training_strategy(config: Dict[str, Any], num_classes: int) -> None:
    """Validate the single-model DDV + FRG + RRS trainer."""

    if num_classes != 1:
        raise ValueError("DRS requires binary segmentation (num_classes=1).")

    def finite_float(name: str, default: float) -> float:
        try:
            value = float(config.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value}.")
        return value

    def integer(name: str, default: int) -> int:
        raw_value = config.get(name, default)
        if isinstance(raw_value, bool):
            raise ValueError(f"{name} must be an integer, got {raw_value!r}.")
        try:
            numeric_value = float(raw_value)
            value = int(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc
        if not math.isfinite(numeric_value) or numeric_value != value:
            raise ValueError(f"{name} must be an integer, got {raw_value!r}.")
        return value

    unlabeled_ratio = finite_float("unlabeled_ratio", 0.0)
    if not 0.0 < unlabeled_ratio < 1.0:
        raise ValueError(
            f"DRS requires unlabeled_ratio in (0, 1), got {unlabeled_ratio}."
        )
    if not bool(config.get("use_two_stream_batch_sampler", False)):
        raise ValueError(
            "DRS requires use_two_stream_batch_sampler=True so every "
            "semi-supervised batch has labeled and unlabeled samples."
        )

    if "rrs_warmup_epochs" not in config:
        raise ValueError("Missing explicit rrs_warmup_epochs config key.")
    if integer("rrs_warmup_epochs", 0) <= 0:
        raise ValueError("rrs_warmup_epochs must be a positive integer.")

    if integer("grad_accum_steps", 1) < 1:
        raise ValueError("grad_accum_steps must be a positive integer.")
    if finite_float("max_grad_norm", 1.0) <= 0.0:
        raise ValueError("max_grad_norm must be finite and positive.")
    if integer("num_epochs", 1) < 1:
        raise ValueError("num_epochs must be a positive integer.")
    if finite_float("initial_lr", 1e-3) <= 0.0:
        raise ValueError("initial_lr must be finite and positive.")
    if finite_float("weight_decay", 1e-4) < 0.0:
        raise ValueError("weight_decay must be finite and non-negative.")
    if finite_float("boundary_loss_weight", 0.1) < 0.0:
        raise ValueError("boundary_loss_weight must be finite and non-negative.")
    boundary_kernel = integer("boundary_kernel_size", 3)
    if boundary_kernel <= 0 or boundary_kernel % 2 == 0:
        raise ValueError("boundary_kernel_size must be a positive odd integer.")


def validate_config(config: Dict[str, Any]) -> None:
    """检查训练配置是否满足当前分割任务的基本要求。"""

    image_path = config.get("image_path")
    mask_path = config.get("mask_path")
    if not image_path or not mask_path:
        raise ValueError("Please set both image_path and mask_path before training.")

    if not os.path.isdir(image_path):
        raise FileNotFoundError(f"image_path does not exist: {image_path}")
    if not os.path.isdir(mask_path):
        raise FileNotFoundError(f"mask_path does not exist: {mask_path}")

    img_size = int(config.get("img_size", 0))
    if img_size <= 0:
        raise ValueError(f"img_size must be positive, got {img_size}")

    batch_size = int(config.get("batch_size", 0))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not bool(config.get("test_only", False)) and bool(config.get("use_two_stream_batch_sampler", False)):
        labeled_batch_size = int(config.get("labeled_batch_size", 1))
        if not 1 <= labeled_batch_size < batch_size:
            raise ValueError(
                "labeled_batch_size must be in [1, batch_size) when two-stream sampling is enabled, "
                f"got {labeled_batch_size} for batch_size={batch_size}."
            )
        if float(config.get("unlabeled_ratio", 0.0)) <= 0.0:
            raise ValueError("Two-stream sampling requires unlabeled_ratio > 0.")

    num_classes = int(config.get("num_classes", 1))
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if not bool(config.get("test_only", False)):
        _validate_primary_training_strategy(config, num_classes)

    threshold = float(config.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    val_split = float(config.get("val_split", 0.1))
    test_split = float(config.get("test_split", 0.1))
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    if not 0.0 <= test_split < 1.0:
        raise ValueError(f"test_split must be in [0, 1), got {test_split}")
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be smaller than 1.0")

    features = config.get("features")
    if features is not None:
        if not isinstance(features, (list, tuple)) or len(features) not in (4, 5):
            raise ValueError("features must be a list/tuple with 4 (encoder) or 5 (encoder+bottleneck) channel widths")
        if any(int(channel) <= 0 for channel in features):
            raise ValueError(f"features must contain positive integers, got {features}")


# 训练与实验管理辅助函数
def _slugify_path_component(value: str) -> str:
    """将任意标签规整成适合放入文件路径的安全字符串。"""

    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or "run"


def _resolve_train_seeds(config: Dict[str, Any]) -> list[int]:
    """解析 train_seeds，同时保留顺序并去重。

    train_seeds 是唯一训练 seed 配置：写一个值/单元素列表就跑单 seed，
    写多个值就顺序跑多 seed 并在最后汇总均值和标准差。
    """

    raw = config.get("train_seeds")
    if raw is None:
        raise ValueError("Please set train_seeds, e.g. [43] for a single-seed run.")

    if isinstance(raw, (list, tuple, set, range)):
        candidates = list(raw)
    else:
        candidates = [raw]

    resolved: list[int] = []
    for item in candidates:
        seed = int(item)
        if seed not in resolved:
            resolved.append(seed)

    if not resolved:
        raise ValueError("train_seeds must contain at least one seed.")

    return resolved


def _sample_std(values: Sequence[float]) -> float:
    """返回样本标准差；当只有一个值时返回 0。"""

    return float(stdev(values)) if len(values) > 1 else 0.0


def _format_mean_std(values: Sequence[float]) -> str:
    """将指标聚合结果格式化为便于论文记录的“均值 ± 标准差”形式。"""

    clean_values = [float(value) for value in values]
    return f"{fmean(clean_values):.4f} \u00B1 {_sample_std(clean_values):.4f}"


def _write_multi_seed_summary(
    *,
    model_type: str,
    batch_tag: str,
    results_root: str,
    run_summaries: Sequence[Dict[str, Any]],
) -> str:
    """写出每个 seed 的结果明细，以及可直接复制的均值和标准差汇总行。"""

    if not run_summaries:
        raise ValueError("run_summaries must not be empty.")

    model_tag = _slugify_path_component(model_type)
    summary_dir = os.path.join(results_root, model_tag)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"multi_seed_summary_{batch_tag}.txt")

    def _metric_value(metrics: Dict[str, Any], key: str) -> float:
        if key == "precision":
            if "precision" in metrics:
                return float(metrics["precision"])
            return float(metrics.get("pre_percent", 0.0)) / 100.0
        if key == "pre_percent":
            if "pre_percent" in metrics:
                return float(metrics["pre_percent"])
            return float(metrics.get("precision", 0.0)) * 100.0
        return float(metrics[key])

    def _module_metric_items(metrics: Dict[str, Any]) -> list[tuple[str, float]]:
        """Return stable, future-proof scalar diagnostics for the new method."""

        priority = (
            "ddv_morphology_mean", "ddv_semantic_mean", "morphology_valid_ratio",
            "frg_perturbation_consistency_mean", "frg_frequency_mean", "frg_region_mean",
            "frg_reliability_m_mean", "frg_reliability_m_p10",
            "frg_reliability_m_p50", "frg_reliability_m_p90",
            "frg_reliability_m_above_0p5_ratio",
            "frg_reliability_s_mean", "frg_reliability_s_p10",
            "frg_reliability_s_p50", "frg_reliability_s_p90",
            "frg_reliability_s_above_0p5_ratio", "frg_reliability_mean",
            "frg_reliability_p50",
            "frg_reliability_p90", "frg_reliability_p95",
            "rrs_morphology_weight_mean", "rrs_semantic_weight_mean",
            "loss_a_to_b", "loss_b_to_a", "semi_loss", "lambda_RRS",
        )
        ordered_names = [name for name in priority if name in metrics]
        ordered_names.extend(sorted(name for name in metrics if name not in ordered_names))
        items: list[tuple[str, float]] = []
        for name in ordered_names:
            value = metrics[name]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                continue
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                items.append((name, numeric_value))
        return items

    full_dataset_test = all(item.get("evaluation_mode") == "full_dataset_test" for item in run_summaries)
    val_label = "Source Best Val Dice" if full_dataset_test else "Best Val Dice"
    val_key_label = "source_best_val_dice" if full_dataset_test else "best_val_dice"
    aggregate_specs = [
        ("best_val_dice", val_label),
        ("dice", "Dice"),
        ("miou", "mIoU"),
        ("precision", "PRE"),
        ("accuracy", "Accuracy"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ]
    copy_specs = [
        ("dice", "Dice"),
        ("miou", "mIoU"),
        ("precision", "PRE"),
        ("accuracy", "Accuracy"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ]

    lines = [
        f"Model: {model_type}",
        f"Batch: {batch_tag}",
        f"Seeds: {', '.join(str(int(item['train_seed'])) for item in run_summaries)}",
        "",
        "Per-seed results:",
    ]
    if full_dataset_test:
        lines.insert(3, "Evaluation: full target dataset; no training or validation on target data")

    for item in run_summaries:
        metrics = item["test_metrics"]
        lines.append(
            f"seed={int(item['train_seed'])}  "
            f"run={item['run_name']}  "
            f"best_epoch={int(item.get('best_epoch', 0))}  "
            "decoders=2  "
            f"{val_key_label}={float(item['best_val_dice']):.4f}  "
            f"dice={_metric_value(metrics, 'dice'):.4f}  "
            f"mIoU={_metric_value(metrics, 'miou'):.4f}  "
            f"PRE={_metric_value(metrics, 'precision'):.4f}  "
            f"accuracy={_metric_value(metrics, 'accuracy'):.4f}  "
            f"sensitivity={_metric_value(metrics, 'sensitivity'):.4f}  "
            f"specificity={_metric_value(metrics, 'specificity'):.4f}"
        )
        module_metrics = dict(item.get("best_module_metrics") or {})
        module_items = _module_metric_items(module_metrics)
        if module_items:
            lines.append(
                "DDV/FRG/RRS@best: "
                + "  ".join(f"{name}={value:.4f}" for name, value in module_items)
            )
        lines.append(f"root={os.path.abspath(str(item['run_root']))}")
        lines.append(f"best_model={os.path.abspath(str(item['best_model_path']))}")

    lines.append("")
    lines.append("Aggregate (mean \u00B1 std):")
    for key, label in aggregate_specs:
        if key == "best_val_dice":
            values = [float(item["best_val_dice"]) for item in run_summaries]
        else:
            values = [_metric_value(item["test_metrics"], key) for item in run_summaries]
        lines.append(f"{label}: {_format_mean_std(values)}")

    lines.append("")
    lines.append("Copy Header:")
    lines.append("\t".join(label for _, label in copy_specs))
    lines.append("Copy Values:")
    lines.append(
        "\t".join(
            _format_mean_std([_metric_value(item["test_metrics"], key) for item in run_summaries])
            for key, _ in copy_specs
        )
    )

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return summary_path


def _mask_array_to_foreground(mask_array: np.ndarray) -> np.ndarray:
    """Convert a raw mask array to a binary foreground map."""

    arr = np.asarray(mask_array)
    if arr.ndim == 3:
        arr = arr[..., :3]
        if np.all(arr == arr[..., :1]):
            arr = arr[..., 0]
        else:
            arr = arr.max(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported mask shape for visualization selection: {arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    unique_values = np.unique(arr)
    rounded = set(np.round(unique_values).astype(np.int64).tolist())

    # Binary masks often use {0,1} or {0,255}. Multiclass masks use class ids > 1.
    if rounded.issubset({0, 1, 255}):
        if float(arr.max(initial=0.0)) > 1.0:
            foreground = arr >= 127.0
        else:
            foreground = arr >= 0.5
    else:
        foreground = arr > 0.0
    return foreground.astype(np.uint8, copy=False)


def _foreground_shape_complexity(foreground: np.ndarray) -> float:
    """Compute a lightweight boundary complexity score from a binary map."""

    fg = np.asarray(foreground, dtype=np.uint8)
    area = float(fg.sum())
    if area <= 0:
        return 0.0

    h_transitions = np.not_equal(fg[:, 1:], fg[:, :-1]).sum(dtype=np.int64)
    v_transitions = np.not_equal(fg[1:, :], fg[:-1, :]).sum(dtype=np.int64)
    perimeter = float(h_transitions + v_transitions)
    return float((perimeter * perimeter) / (4.0 * math.pi * area + 1e-6))


def get_or_create_fixed_test_visualization_sample_ids(
    *,
    test_dataset: Any,
    config: Optional[Dict[str, Any]] = None,
    num_samples: int = 4,
) -> tuple[list[str], str]:
    """
    Create (once) and reuse a fixed set of representative test sample_ids.

    Strategy:
    1) small lesion (area ~20% quantile)
    2) medium lesion (area ~50% quantile)
    3) large lesion (area ~80% quantile)
    4) complex boundary sample (prefer non-tiny masks)
    """

    cfg = dict(config or {})
    target_count = max(1, int(num_samples))
    schema_version = int(cfg.get("fixed_test_visualization_schema_version", 2) or 2)
    min_area_ratio = float(cfg.get("fixed_test_visualization_min_area_ratio", 0.001) or 0.001)
    prefer_non_empty = bool(cfg.get("fixed_test_visualization_non_empty_only", True))
    force_refresh = bool(cfg.get("fixed_test_visualization_force_refresh", False))

    configured_json = str(cfg.get("fixed_test_visualization_json", "") or "").strip()
    if configured_json:
        json_path = configured_json
    else:
        results_root = str(cfg.get("results_root", "results") or "results")
        json_path = os.path.join(results_root, "fixed_test_visualization_sample_ids.json")

    json_dir = os.path.dirname(json_path)
    if json_dir:
        ensure_dir(json_dir)

    if os.path.isfile(json_path) and not force_refresh:
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            sample_ids = [str(item).strip() for item in saved.get("sample_ids", []) if str(item).strip()]
            saved_version = int(saved.get("schema_version", 0) or 0)
            if sample_ids and saved_version >= schema_version:
                return sample_ids[:target_count], json_path
        except Exception:
            pass

    dataset_samples = list(getattr(test_dataset, "samples", []) or [])
    records: list[Dict[str, Any]] = []
    for sample in dataset_samples:
        sample_id = str(sample.get("sample_id", "")).strip()
        mask_path = str(sample.get("mask_path", "")).strip()
        if not sample_id or not mask_path or not os.path.isfile(mask_path):
            continue
        try:
            from PIL import Image

            with Image.open(mask_path) as mask_img:
                mask_arr = np.asarray(mask_img)
            foreground = _mask_array_to_foreground(mask_arr)
            area_ratio = float(foreground.mean())
            complexity = _foreground_shape_complexity(foreground)
            records.append(
                {
                    "sample_id": sample_id,
                    "mask_path": mask_path,
                    "area_ratio": area_ratio,
                    "complexity": complexity,
                }
            )
        except Exception:
            continue

    if not records:
        return [], json_path

    target_count = min(target_count, len(records))
    if target_count <= 0:
        return [], json_path
    area_values = [float(item["area_ratio"]) for item in records]
    complexity_values = [float(item["complexity"]) for item in records]

    all_indices = list(range(len(records)))
    if prefer_non_empty:
        eligible_indices = [idx for idx, area in enumerate(area_values) if area >= min_area_ratio]
        if len(eligible_indices) < max(2, target_count):
            eligible_indices = [idx for idx, area in enumerate(area_values) if area > 0.0]
        if not eligible_indices:
            eligible_indices = list(all_indices)
    else:
        eligible_indices = list(all_indices)

    eligible_area_values = [area_values[idx] for idx in eligible_indices]
    median_area = float(np.median(np.asarray(eligible_area_values, dtype=np.float64)))

    selected_indices: list[int] = []
    selected_roles: list[str] = []
    excluded: set[int] = set()

    def _append_index(index: Optional[int], role: str) -> None:
        if index is None or index in excluded:
            return
        excluded.add(int(index))
        selected_indices.append(int(index))
        selected_roles.append(role)

    def _pick_area_quantile_index(pool_indices: Sequence[int], quantile: float) -> Optional[int]:
        pool = [int(idx) for idx in pool_indices if int(idx) not in excluded]
        if not pool:
            return None
        q_values = np.asarray([area_values[idx] for idx in pool], dtype=np.float64)
        target = float(np.quantile(q_values, float(quantile)))
        pool.sort(
            key=lambda idx: (
                abs(area_values[idx] - target),
                -complexity_values[idx],
                idx,
            )
        )
        return int(pool[0])

    if target_count >= 1:
        _append_index(_pick_area_quantile_index(eligible_indices, quantile=0.20), "small_area")
    if target_count >= 2:
        _append_index(_pick_area_quantile_index(eligible_indices, quantile=0.50), "medium_area")
    if target_count >= 3:
        _append_index(_pick_area_quantile_index(eligible_indices, quantile=0.80), "large_area")
    if target_count >= 4:
        complex_pool = [idx for idx in eligible_indices if idx not in excluded]
        if complex_pool and len(complex_pool) >= 4:
            area_floor = float(np.quantile(np.asarray([area_values[idx] for idx in complex_pool], dtype=np.float64), 0.30))
            non_tiny_pool = [idx for idx in complex_pool if area_values[idx] >= area_floor]
            if non_tiny_pool:
                complex_pool = non_tiny_pool
        if complex_pool:
            complex_pool.sort(
                key=lambda idx: (
                    complexity_values[idx],
                    -abs(area_values[idx] - median_area),
                    area_values[idx],
                ),
                reverse=True,
            )
            _append_index(int(complex_pool[0]), "complex_boundary")

    if len(selected_indices) < target_count:
        fallback_eligible = sorted(
            [idx for idx in eligible_indices if idx not in excluded],
            key=lambda i: (
                complexity_values[i],
                -abs(area_values[i] - median_area),
                area_values[i],
            ),
            reverse=True,
        )
        for idx in fallback_eligible:
            if idx in excluded:
                continue
            _append_index(idx, "fallback_eligible")
            if len(selected_indices) >= target_count:
                break

    if len(selected_indices) < target_count:
        fallback_all = sorted(
            [idx for idx in all_indices if idx not in excluded],
            key=lambda i: (
                complexity_values[i],
                -abs(area_values[i] - median_area),
                area_values[i],
            ),
            reverse=True,
        )
        for idx in fallback_all:
            _append_index(idx, "fallback_all")
            if len(selected_indices) >= target_count:
                break

    selected_records = [records[idx] for idx in selected_indices[:target_count]]
    selected_sample_ids = [str(item["sample_id"]) for item in selected_records]

    payload = {
        "schema_version": int(schema_version),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_strategy": "non-empty filtered area-quantiles (20/50/80) + complex boundary",
        "selection_config": {
            "min_area_ratio": float(min_area_ratio),
            "prefer_non_empty": bool(prefer_non_empty),
            "force_refresh": bool(force_refresh),
        },
        "target_count": int(target_count),
        "dataset_size": int(len(records)),
        "sample_ids": selected_sample_ids,
        "details": [
            {
                "role": selected_roles[idx] if idx < len(selected_roles) else "selected",
                "sample_id": str(item["sample_id"]),
                "area_ratio": float(item["area_ratio"]),
                "complexity": float(item["complexity"]),
                "mask_path": str(item["mask_path"]),
            }
            for idx, item in enumerate(selected_records)
        ],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return selected_sample_ids, json_path


def _denormalize_image_for_visualization(
    image_tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    normalize: bool = True,
) -> np.ndarray:
    """把模型输入张量还原成可显示的 RGB 图像。"""

    image = image_tensor.detach().cpu().float().clone()
    if image.dim() == 4:
        image = image[0]

    if normalize:
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        image = image * std_tensor + mean_tensor

    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def _tensor_to_display_map(
    tensor: torch.Tensor,
    *,
    num_classes: int = 1,
    threshold: float = 0.5,
    treat_as_prediction: bool = False,
) -> np.ndarray:
    """把 tensor 统一转成可视化用的二维 numpy 图。"""

    data = tensor.detach().cpu().float().clone()
    if data.dim() == 4:
        data = data[0]

    if treat_as_prediction:
        if num_classes <= 1:
            if data.dim() == 3:
                data = data[0] if data.shape[0] == 1 else data.mean(dim=0)
            if data.dim() != 2:
                raise ValueError("Binary prediction tensor must be 2D/3D/4D.")
            if float(data.amin().item()) < 0.0 or float(data.amax().item()) > 1.0:
                data = torch.sigmoid(data)
            data = (data >= float(threshold)).float()
        else:
            if data.dim() == 3 and data.shape[0] > 1:
                data = torch.argmax(data, dim=0).float()
            elif data.dim() == 3 and data.shape[0] == 1:
                data = data[0]
            elif data.dim() != 2:
                raise ValueError("Multiclass prediction tensor must be 2D/3D/4D.")
    else:
        if data.dim() == 3:
            data = data[0] if data.shape[0] == 1 else data.mean(dim=0)
        if data.dim() != 2:
            raise ValueError("Visualization tensor must be 2D/3D/4D.")
        data_min = float(data.amin().item())
        data_max = float(data.amax().item())
        if data_min < 0.0 or data_max > 1.0:
            data = torch.sigmoid(data)
        data = data.clamp(0.0, 1.0)

    return data.numpy()


def _foreground_from_display_map(display_map: np.ndarray) -> np.ndarray:
    """把可视化二维图统一转成前景布尔 mask。"""

    array = np.asarray(display_map)
    if array.ndim != 2:
        raise ValueError("display_map must be a 2D array.")
    return array > 0.5 if np.issubdtype(array.dtype, np.floating) else array > 0
def _mask_to_boundary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    从二值 mask 中提取边界。
    不依赖 BURG，GT 和预测 mask 都可以用。
    """
    fg = _foreground_from_display_map(mask).astype(np.uint8)

    if fg.ndim != 2:
        raise ValueError("mask must be 2D.")

    pad = int(radius)
    padded = np.pad(fg, pad_width=pad, mode="constant", constant_values=0)

    h, w = fg.shape
    eroded = np.ones_like(fg, dtype=np.uint8)

    k = 2 * pad + 1
    for dy in range(k):
        for dx in range(k):
            eroded &= padded[dy:dy + h, dx:dx + w]

    boundary = fg ^ eroded
    return boundary.astype(np.float32)

def _apply_color_overlay(
    image_np: np.ndarray,
    mask: np.ndarray,
    color: Sequence[float],
    alpha: float,
) -> np.ndarray:
    """把指定颜色覆盖到 mask 区域，便于高亮显示差异。"""

    output = image_np.copy()
    if not np.any(mask):
        return output

    color_np = np.asarray(color, dtype=np.float32)
    output[mask] = (1.0 - alpha) * output[mask] + alpha * color_np
    return np.clip(output, 0.0, 1.0)


def _draw_mask_contour(
    ax,
    mask: np.ndarray,
    *,
    color: str,
    linewidth: float = 2.0,
    alpha: float = 1.0,
    linestyle: str = "-",
    zorder: int = 5,
    outline: bool = True,
) -> None:
    """在坐标轴上绘制二值前景轮廓，支持虚线、层级和白色描边。"""

    import matplotlib.patheffects as pe

    fg_mask = _foreground_from_display_map(mask)
    if fg_mask.shape[0] < 2 or fg_mask.shape[1] < 2:
        return
    if not fg_mask.any() or fg_mask.all():
        return

    contour_set = ax.contour(
        fg_mask.astype(np.float32),
        levels=[0.5],
        colors=[color],
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=alpha,
        zorder=zorder,
    )

    if outline:
        contour_artists = getattr(contour_set, "collections", None)
        if contour_artists is None:
            contour_artists = [contour_set] if hasattr(contour_set, "set_path_effects") else []
        for coll in contour_artists:
            coll.set_path_effects([
                pe.Stroke(linewidth=linewidth + 2.2, foreground="white"),
                pe.Normal(),
            ])


def save_segmentation_visualization_panel(
    image_tensor: torch.Tensor,
    pred_mask: torch.Tensor,
    save_path: str,
    config: Dict[str, Any],
    target_mask: Optional[torch.Tensor] = None,
    panel_title: Optional[str] = None,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(save_path) or ".")
    vis_dpi = int(config.get("visualization_dpi", config.get("panel_dpi", 300)))
    vis_dpi = max(150, vis_dpi)
    vis_pad_inches = float(config.get("visualization_pad_inches", 0.02))

    image_np = _denormalize_image_for_visualization(
        image_tensor=image_tensor,
        mean=config.get("mean", [0.5, 0.5, 0.5]),
        std=config.get("std", [0.5, 0.5, 0.5]),
        normalize=bool(config.get("normalize", True)),
    )

    num_classes = int(config.get("num_classes", 1))
    threshold = float(config.get("threshold", 0.5))

    pred_np = _tensor_to_display_map(
        pred_mask,
        num_classes=num_classes,
        threshold=threshold,
        treat_as_prediction=True,
    )

    target_np = None
    if target_mask is not None:
        target_np = _tensor_to_display_map(
            target_mask,
            num_classes=num_classes,
            threshold=threshold,
            treat_as_prediction=True,
        )

    pred_boundary = _mask_to_boundary(pred_np, radius=1)

    if target_np is not None:
        gt_boundary = _mask_to_boundary(target_np, radius=1)

        pred_overlay = _apply_color_overlay(
            image_np=image_np,
            mask=_foreground_from_display_map(pred_np),
            color=(0.1, 0.75, 0.35),
            alpha=0.38,
        )

        boundary_overlay = image_np.copy()
        boundary_overlay = _apply_color_overlay(
            image_np=boundary_overlay,
            mask=gt_boundary > 0,
            color=(0.17, 0.48, 0.90),
            alpha=0.90,
        )
        boundary_overlay = _apply_color_overlay(
            image_np=boundary_overlay,
            mask=pred_boundary > 0,
            color=(1.0, 0.35, 0.20),
            alpha=0.90,
        )

        fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=vis_dpi)

        axes[0].imshow(image_np)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(target_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("GT Mask")
        axes[1].axis("off")

        axes[2].imshow(pred_overlay)
        axes[2].imshow(pred_np, cmap="Greens", alpha=0.28, vmin=0, vmax=1)
        axes[2].set_title("Pred Mask")
        axes[2].axis("off")

        axes[3].imshow(boundary_overlay)
        _draw_mask_contour(axes[3], target_np, color="#2C7BE5", linewidth=2.5, linestyle="-", zorder=5)
        _draw_mask_contour(axes[3], pred_np, color="#FF6B3D", linewidth=2.3, linestyle="--", zorder=6)
        axes[3].set_title("Boundary: GT / Pred")
        axes[3].axis("off")

    else:
        boundary_overlay = image_np.copy()
        boundary_overlay = _apply_color_overlay(
            image_np=boundary_overlay,
            mask=pred_boundary > 0,
            color=(1.0, 0.35, 0.20),
            alpha=0.90,
        )

        pred_overlay = _apply_color_overlay(
            image_np=image_np,
            mask=_foreground_from_display_map(pred_np),
            color=(0.1, 0.75, 0.35),
            alpha=0.38,
        )

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=vis_dpi)

        axes[0].imshow(image_np)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(pred_overlay)
        axes[1].imshow(pred_np, cmap="Greens", alpha=0.28, vmin=0, vmax=1)
        axes[1].set_title("Pred Mask")
        axes[1].axis("off")

        axes[2].imshow(boundary_overlay)
        _draw_mask_contour(axes[2], pred_np, color="#FF6B3D", linewidth=2.3, linestyle="--", zorder=6)
        axes[2].set_title("Pred Boundary")
        axes[2].axis("off")

    if panel_title is None:
        panel_title = "Segmentation Visualization"

    fig.suptitle(panel_title, fontsize=14, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(save_path, dpi=vis_dpi, bbox_inches="tight", pad_inches=vis_pad_inches)
    plt.close(fig)
    return save_path


def load_best_model(
    config: Dict[str, Any],
    model_path: str,
    model_type: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """测试阶段加载最佳模型，并返回模型、推理配置和 checkpoint 内容。"""

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    checkpoint_config = dict(checkpoint.get("config") or {})
    runtime_config = dict(config or {})

    model_config = dict(checkpoint_config or runtime_config)
    if not model_config:
        raise ValueError("Checkpoint does not contain config and no fallback config was provided.")

    if model_type is not None:
        model_config["model_type"] = model_type

    device = runtime_config.get("device") or checkpoint_config.get("device")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(model_config)
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_result = model.load_state_dict(state_dict, strict=True)
    if getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None):
        raise RuntimeError(
            "Checkpoint loaded with unexpected incompatibilities: "
            f"missing={getattr(load_result, 'missing_keys', [])}, "
            f"unexpected={getattr(load_result, 'unexpected_keys', [])}"
        )
    merged_config = dict(model_config)
    for key in {
        "device",
        "normalize", "mean", "std", "mask_threshold", "mask_suffixes",
        "metric_include_background", "ignore_index", "batch_log_interval",
        "results_root", "run_root", "run_name", "predict_artifact_prefix",
        "test_only", "test_total_samples", "test_normalization", "evaluation_mode",
        "image_path",
        "mask_path",
        "prediction_dir",
        "batch_size",
        "num_workers",
        "threshold",
        "val_split",
        "test_split",
        "max_test_visualizations",
        "fixed_test_visualization_json",
        "save_frcs_intermediate_visualizations",
        "frcs_intermediate_strong_flip",
        "frcs_intermediate_dpi",
        "save_dir_model",
        "save_dir_png",
        "split_seed",
        "train_seed",
        "seed",
    }:
        value = runtime_config.get(key)
        if value is not None:
            merged_config[key] = value

    merged_config["device"] = device
    if model_type is not None:
        merged_config["model_type"] = model_type

    model = model.to(device)
    model.eval()

    print(f"Loaded best model from: {os.path.abspath(model_path)}")
    print(f"Model type: {merged_config.get('model_type', 'model')}")
    print(f"Device: {device}")
    return model, merged_config, checkpoint


__all__ = [
    "TrainingArtifactManager",
    "_build_metric_meter",
    "_empty_epoch_metrics",
    "_get_loss_breakdown",
    "_gradients_are_finite",
    "_is_memory_error",
    "_log_running_metrics",
    "_prepare_run_output_dirs",
    "_raise_memory_error",
    "_resolve_train_seeds",
    "_slugify_path_component",
    "_write_multi_seed_summary",
    "clear_batch_status_line",
    "create_criterion",
    "ensure_dir",
    "get_or_create_fixed_test_visualization_sample_ids",
    "load_best_model",
    "print_model_info",
    "print_batch_status",
    "save_segmentation_visualization_panel",
    "save_checkpoint",
    "save_runtime_config",
    "set_seed",
    "should_log_batch",
    "validate_config",
]
