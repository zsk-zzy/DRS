

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from DataLoading import (
    SegmentationDataset,
    create_segmentation_dataloaders,
)
from metric.segmentation_metrics import SegmentationMetricMeter
from SkinDataAugmentor import SegmentationAugmentor
from utils.utils import (
    get_or_create_fixed_test_visualization_sample_ids,
    load_best_model,
    save_segmentation_visualization_panel,
)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _prediction_mask_to_metric_logits(preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert discrete prediction masks back to logits so metric code can stay unchanged."""

    if int(num_classes) <= 1:
        positive = torch.full_like(preds, 20.0, dtype=torch.float32)
        negative = torch.full_like(preds, -20.0, dtype=torch.float32)
        return torch.where(preds > 0, positive, negative).unsqueeze(1)

    batch_size, height, width = preds.shape
    logits = torch.full(
        (batch_size, int(num_classes), height, width),
        -20.0,
        device=preds.device,
        dtype=torch.float32,
    )
    return logits.scatter_(1, preds.unsqueeze(1), 20.0)


def _logits_to_mask(logits: torch.Tensor, num_classes: int, threshold: float) -> torch.Tensor:
   

    if int(num_classes) <= 1:
        return (torch.sigmoid(logits) >= threshold).long().squeeze(1)
    return torch.argmax(logits, dim=1).long()


def _extract_primary_logits(model_output: torch.Tensor | tuple | list) -> torch.Tensor:
   

    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, (tuple, list)) and len(model_output) > 0 and torch.is_tensor(model_output[0]):
        return model_output[0]
    raise TypeError(f"Model output must be logits tensor or tuple/list with logits first, got {type(model_output)!r}")


def _target_to_mask(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    if int(num_classes) <= 1:
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        return (targets >= 0.5).long()

    if targets.dim() == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    return targets.long()


def _print_batch_status(prefix: str, batch_idx: int, total_batches: int, **metrics) -> None:
    """Use a single refreshing console line for batch progress."""

    width = max(3, len(str(total_batches)))
    metrics_str = "  ".join(f"{key}={value}" for key, value in metrics.items())
    last_len = getattr(_print_batch_status, "_last_len", 0)
    message = f"{prefix} [{batch_idx + 1:0{width}d}/{total_batches}] {metrics_str}"

    padding = " " * max(0, last_len - len(message))
    is_last_batch = batch_idx == total_batches - 1
    end = "\n" if is_last_batch else "\r"
    print(f"{message}{padding}", end=end, flush=True)
    _print_batch_status._last_len = 0 if is_last_batch else len(message)


def _should_log_batch(batch_idx: int, total_batches: int, log_interval: int) -> bool:
    """Keep test logging cadence aligned with train/val."""

    if total_batches <= 0:
        return False
    interval = max(1, int(log_interval))
    return (batch_idx + 1) % interval == 0 or batch_idx == total_batches - 1


def _build_full_dataset_test_loader(config: Dict[str, Any]):
    """Build a test loader over the whole dataset without split."""

    dataset = SegmentationDataset(
        image_folder=config.get("image_path"),
        mask_folder=config.get("mask_path"),
        split="all",
        seed=int(config.get("split_seed", config.get("seed", 42))),
        val_split=0.0,
        test_split=0.0,
        config=config,
        verbose=True,
    )
    augmentor = SegmentationAugmentor(
        img_size=int(config.get("img_size", 256)),
        num_classes=int(config.get("num_classes", 1)),
        normalize=bool(config.get("normalize", True)),
        mean=config.get("mean"),
        std=config.get("std"),
        mask_threshold=config.get("mask_threshold"),
    )
    dataset.set_transform(augmentor.get_test_transform())
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[full] samples={len(dataset)}")
    return loader


def _safe_div(numerator: float, denominator: float, eps: float = 1e-7) -> float:
    return float(numerator / (denominator + eps))


def _json_safe_metric_value(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_metric_value(item) for item in value]
    return value


def _sanitize_sample_name(sample_id: str) -> str:
    return str(sample_id).replace("/", "_").replace("\\", "_").strip("_") or "sample"


def _denormalize_image_for_visualization(image_tensor: torch.Tensor, config: Dict[str, Any]) -> np.ndarray:
    image = image_tensor.detach().cpu().float().clone()
    if image.dim() == 4:
        image = image[0]
    if image.dim() != 3:
        raise ValueError(f"Expected CHW image tensor, got shape={tuple(image.shape)}")

    if bool(config.get("normalize", True)):
        channels = int(image.shape[0])
        mean_cfg = list(config.get("mean", [0.5, 0.5, 0.5]))
        std_cfg = list(config.get("std", [0.5, 0.5, 0.5]))
        if len(mean_cfg) < channels:
            mean_cfg = mean_cfg + [mean_cfg[-1] if mean_cfg else 0.5] * (channels - len(mean_cfg))
        if len(std_cfg) < channels:
            std_cfg = std_cfg + [std_cfg[-1] if std_cfg else 0.5] * (channels - len(std_cfg))
        mean = torch.tensor(mean_cfg[:channels], dtype=torch.float32).view(channels, 1, 1)
        std = torch.tensor(std_cfg[:channels], dtype=torch.float32).view(channels, 1, 1)
        image = image * std + mean

    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def _tensor_to_label_map(mask_tensor: torch.Tensor, num_classes: int) -> np.ndarray:
    data = mask_tensor.detach().cpu()
    if data.dim() == 4:
        data = data[0]
    if data.dim() == 3 and data.shape[0] == 1:
        data = data[0]
    if data.dim() == 3 and int(num_classes) > 1:
        data = torch.argmax(data, dim=0)
    if data.dim() != 2:
        raise ValueError(f"Expected 2D mask after squeeze/argmax, got shape={tuple(data.shape)}")
    return data.numpy()


def _target_binary_and_valid_map(
    target_mask: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    labels = _tensor_to_label_map(target_mask, num_classes=num_classes)
    if int(num_classes) <= 1:
        gt = labels.astype(np.float32) >= 0.5
    else:
        labels = np.round(labels).astype(np.int64)
        gt = labels > 0

    valid = np.ones_like(gt, dtype=bool)
    if ignore_index is not None:
        valid = np.round(labels).astype(np.int64) != int(ignore_index)
        gt = gt & valid
    return gt.astype(bool), valid.astype(bool)


def _prediction_binary_map(pred_mask: torch.Tensor, *, num_classes: int) -> np.ndarray:
    labels = _tensor_to_label_map(pred_mask, num_classes=num_classes)
    if int(num_classes) <= 1:
        return labels.astype(np.float32) > 0.5
    return labels.astype(np.int64) > 0


def _compute_sample_binary_metrics(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: Optional[int],
) -> Dict[str, float]:
    pred_fg = _prediction_binary_map(pred_mask, num_classes=num_classes)
    gt_fg, valid = _target_binary_and_valid_map(
        target_mask,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    pred_fg = pred_fg & valid
    valid_pixels = int(valid.sum())
    if valid_pixels <= 0:
        return {
            "dice": 0.0,
            "miou": 0.0,
            "precision": 0.0,
            "pre_percent": 0.0,
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "sens_percent": 0.0,
            "spe_percent": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
            "gt_area_ratio": 0.0,
            "pred_area_ratio": 0.0,
            "valid_pixels": 0.0,
        }

    tp = float(np.logical_and(pred_fg, gt_fg).sum())
    fp = float(np.logical_and(pred_fg, np.logical_not(gt_fg)).sum())
    fn = float(np.logical_and(np.logical_not(pred_fg), gt_fg).sum())
    tn = float(np.logical_and(np.logical_not(pred_fg), np.logical_not(gt_fg)).sum())

    dice = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    miou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    gt_area_ratio = _safe_div(float(gt_fg.sum()), float(valid_pixels))
    pred_area_ratio = _safe_div(float(pred_fg.sum()), float(valid_pixels))

    return {
        "dice": float(dice),
        "miou": float(miou),
        "precision": float(precision),
        "pre_percent": float(precision * 100.0),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "sens_percent": float(sensitivity * 100.0),
        "spe_percent": float(specificity * 100.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_area_ratio": float(gt_area_ratio),
        "pred_area_ratio": float(pred_area_ratio),
        "valid_pixels": float(valid_pixels),
    }


def _compute_union_bbox(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    padding_ratio: float = 0.16,
    min_padding: int = 12,
) -> Tuple[int, int, int, int]:
    height, width = gt_mask.shape
    union_mask = np.logical_or(gt_mask, pred_mask)
    if not union_mask.any():
        return 0, height, 0, width

    ys, xs = np.where(union_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad = max(min_padding, int(round(max(y1 - y0, x1 - x0) * float(padding_ratio))))
    y0 = max(0, y0 - pad)
    y1 = min(height, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(width, x1 + pad)
    return y0, y1, x0, x1


def _save_error_decomposition_panel(
    *,
    image_tensor: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    save_path: str,
    config: Dict[str, Any],
    sample_name: str,
    sample_metrics: Dict[str, float],
) -> str:
    num_classes = int(config.get("num_classes", 1))
    ignore_index = config.get("ignore_index")
    vis_dpi = max(150, int(config.get("visualization_dpi", config.get("panel_dpi", 300))))
    vis_pad_inches = float(config.get("visualization_pad_inches", 0.02))

    image_np = _denormalize_image_for_visualization(image_tensor, config=config)
    pred_fg = _prediction_binary_map(pred_mask, num_classes=num_classes)
    gt_fg, valid = _target_binary_and_valid_map(
        target_mask,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    pred_fg = pred_fg & valid

    tp = np.logical_and(pred_fg, gt_fg)
    fp = np.logical_and(pred_fg, np.logical_not(gt_fg))
    fn = np.logical_and(np.logical_not(pred_fg), gt_fg)

    error_overlay = image_np.copy()
    if np.any(tp):
        error_overlay[tp] = 0.58 * error_overlay[tp] + 0.42 * np.array([0.08, 0.80, 0.28], dtype=np.float32)
    if np.any(fp):
        error_overlay[fp] = 0.58 * error_overlay[fp] + 0.42 * np.array([1.00, 0.18, 0.18], dtype=np.float32)
    if np.any(fn):
        error_overlay[fn] = 0.58 * error_overlay[fn] + 0.42 * np.array([0.14, 0.43, 1.00], dtype=np.float32)
    error_overlay = np.clip(error_overlay, 0.0, 1.0)

    y0, y1, x0, x1 = _compute_union_bbox(gt_fg, pred_fg)
    zoom_img = image_np[y0:y1, x0:x1, :]
    zoom_gt = gt_fg[y0:y1, x0:x1]
    zoom_pred = pred_fg[y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 5, figsize=(23, 5), dpi=vis_dpi)
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt_fg.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_fg.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Pred Mask")
    axes[2].axis("off")

    axes[3].imshow(error_overlay)
    axes[3].set_title("TP / FP / FN")
    axes[3].axis("off")

    axes[4].imshow(zoom_img)
    if zoom_gt.shape[0] >= 2 and zoom_gt.shape[1] >= 2 and zoom_gt.any():
        axes[4].contour(zoom_gt.astype(np.float32), levels=[0.5], colors=["#2C7BE5"], linewidths=2.2, linestyles="-")
    if zoom_pred.shape[0] >= 2 and zoom_pred.shape[1] >= 2 and zoom_pred.any():
        axes[4].contour(zoom_pred.astype(np.float32), levels=[0.5], colors=["#FF6B3D"], linewidths=2.0, linestyles="--")
    axes[4].set_title("Boundary Zoom")
    axes[4].axis("off")

    fig.suptitle(
        f"{sample_name} | Dice={sample_metrics['dice']:.4f}  "
        f"mIoU={sample_metrics['miou']:.4f}  "
        f"PRE={sample_metrics.get('precision', sample_metrics.get('pre_percent', 0.0) / 100.0):.4f}",
        fontsize=13,
        y=0.99,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(save_path, dpi=vis_dpi, bbox_inches="tight", pad_inches=vis_pad_inches)
    plt.close(fig)
    return save_path


def _save_sample_metric_distribution_plot(sample_rows: Sequence[Dict[str, Any]], save_path: str) -> str:
    if not sample_rows:
        return save_path

    dice = np.asarray([float(item["dice"]) for item in sample_rows], dtype=np.float32)
    miou = np.asarray([float(item["miou"]) for item in sample_rows], dtype=np.float32)
    pre = np.asarray([float(item.get("pre_percent", item.get("precision", 0.0) * 100.0)) for item in sample_rows], dtype=np.float32)
    sens = np.asarray([float(item["sensitivity"]) for item in sample_rows], dtype=np.float32)
    spec = np.asarray([float(item["specificity"]) for item in sample_rows], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=220)
    bins = min(20, max(8, int(np.sqrt(max(1, len(dice))))))

    axes[0].hist(dice, bins=bins, color="#2C7BE5", alpha=0.85, edgecolor="white")
    axes[0].set_title("Dice Distribution")
    axes[0].set_xlabel("Dice")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.25, linestyle=":")

    axes[1].hist(miou, bins=bins, color="#00A884", alpha=0.85, edgecolor="white")
    axes[1].set_title("mIoU Distribution")
    axes[1].set_xlabel("mIoU")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.25, linestyle=":")

    box_values = [dice * 100.0, miou * 100.0, pre, sens * 100.0, spec * 100.0]
    box_labels = ["Dice", "mIoU", "PRE", "SEN", "SPE"]
    axes[2].boxplot(box_values, patch_artist=True)
    axes[2].set_xticks(range(1, len(box_labels) + 1), box_labels)
    axes[2].set_ylim(0.0, 102.0)
    axes[2].set_title("Per-sample Metric Spread")
    axes[2].set_ylabel("%")
    axes[2].grid(alpha=0.25, linestyle=":")

    fig.suptitle(
        f"Sample-level Metrics (n={len(sample_rows)}) | "
        f"Dice={float(dice.mean()):.4f}+/-{float(dice.std()):.4f}",
        fontsize=12.5,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _assign_area_groups(sample_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Tuple[float, float]]:
    rows = [dict(item) for item in sample_rows]
    if not rows:
        return rows, (0.0, 0.0)

    area_values = np.asarray([float(item.get("gt_area_ratio", 0.0)) for item in rows], dtype=np.float32)
    positive = area_values[area_values > 0.0]
    base = positive if positive.size >= 3 else area_values
    if base.size == 0:
        q1, q2 = 0.0, 0.0
    else:
        q1, q2 = [float(v) for v in np.quantile(base, [0.33, 0.66])]

    for idx, row in enumerate(rows):
        area = float(area_values[idx])
        if area <= q1:
            area_group = "small"
        elif area <= q2:
            area_group = "medium"
        else:
            area_group = "large"
        row["area_group"] = area_group
    return rows, (q1, q2)


def _save_area_stratified_plot(
    sample_rows: Sequence[Dict[str, Any]],
    *,
    save_path: str,
    stats_path: str,
) -> Tuple[str, str]:
    if not sample_rows:
        with open(stats_path, "w", encoding="utf-8") as handle:
            json.dump({"groups": {}, "quantiles": [0.0, 0.0], "samples": 0}, handle, indent=2, ensure_ascii=False)
        return save_path, stats_path

    rows, (q1, q2) = _assign_area_groups(sample_rows)
    groups = ("small", "medium", "large")
    dice_means: List[float] = []
    miou_means: List[float] = []
    counts: List[int] = []
    payload_groups: Dict[str, Any] = {}

    for group in groups:
        sub = [item for item in rows if item.get("area_group") == group]
        dice_vals = np.asarray([float(item["dice"]) for item in sub], dtype=np.float32) if sub else np.asarray([], dtype=np.float32)
        miou_vals = np.asarray([float(item["miou"]) for item in sub], dtype=np.float32) if sub else np.asarray([], dtype=np.float32)
        dice_mean = float(dice_vals.mean()) if dice_vals.size > 0 else 0.0
        miou_mean = float(miou_vals.mean()) if miou_vals.size > 0 else 0.0
        dice_means.append(dice_mean)
        miou_means.append(miou_mean)
        counts.append(len(sub))
        payload_groups[group] = {
            "count": int(len(sub)),
            "dice_mean": float(dice_mean),
            "miou_mean": float(miou_mean),
            "dice_std": float(dice_vals.std()) if dice_vals.size > 0 else 0.0,
            "miou_std": float(miou_vals.std()) if miou_vals.size > 0 else 0.0,
        }

    x = np.arange(len(groups), dtype=np.float32)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=220)
    bars1 = ax.bar(x - width / 2.0, dice_means, width=width, color="#2C7BE5", label="Dice")
    bars2 = ax.bar(x + width / 2.0, miou_means, width=width, color="#00A884", label="mIoU")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={c})" for g, c in zip(groups, counts)])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Area-stratified Segmentation Performance")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "samples": int(len(rows)),
        "quantiles": [float(q1), float(q2)],
        "groups": payload_groups,
    }
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return save_path, stats_path


def _save_sample_metrics_csv(sample_rows: Sequence[Dict[str, Any]], save_path: str) -> str:
    fieldnames = [
        "sample_id",
        "sample_name",
        "dice",
        "miou",
        "precision",
        "pre_percent",
        "accuracy",
        "sensitivity",
        "specificity",
        "sens_percent",
        "spe_percent",
        "gt_area_ratio",
        "pred_area_ratio",
        "area_group",
        "tp",
        "fp",
        "fn",
        "tn",
        "valid_pixels",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return save_path


class ModelPredictor:
    

    def __init__(
        self,
        args: Dict,
        test_loader=None,
        checkpoint_path: Optional[str] = None,
    ):
        self.request_args = dict(args)
        self.args = dict(args)
        self.device = self.args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.test_loader = test_loader
        self.checkpoint_path = checkpoint_path or self._default_checkpoint_path()
        self.prediction_dir = self.args.get("prediction_dir", r"results\predictions")
        # Visualization artifacts are grouped directly under results/<dataset>_<model>/.
        self.visualization_dir = os.path.join(self.prediction_dir, "panel")
        self.error_visualization_dir = os.path.join(self.prediction_dir, "error_decomposition")
        self.paper_visualization_dir = os.path.join(self.prediction_dir, "paper_figures")
        self.max_visualizations = int(self.args.get("max_test_visualizations", 4))

        self.model = None
        self._load_model()

    def _default_checkpoint_path(self) -> str:
       
        model_type = self.request_args.get("model_type", "DRS")
        save_dir = self.request_args.get("save_dir_model", r"results\checkpoint")
        return os.path.join(save_dir, f"best_model_{model_type}.pth")

    def _load_model(self) -> None:
        self.model, merged_config, checkpoint = load_best_model(
            config=self.request_args,
            model_path=self.checkpoint_path,
        )
        if self.request_args.get("prediction_dir"):
            merged_config["prediction_dir"] = self.request_args.get("prediction_dir")
        self.args = merged_config
        self.device = self.args.get("device", self.device)
        self.prediction_dir = self.args.get("prediction_dir", r"results\predictions")
        self.visualization_dir = os.path.join(self.prediction_dir, "panel")
        self.error_visualization_dir = os.path.join(self.prediction_dir, "error_decomposition")
        self.paper_visualization_dir = os.path.join(self.prediction_dir, "paper_figures")
        self.max_visualizations = int(self.args.get("max_test_visualizations", 4))
        self._clear_decoder_aux_cache_for_inference()

        os.makedirs(self.prediction_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.paper_visualization_dir, exist_ok=True)
        os.makedirs(self.error_visualization_dir, exist_ok=True)

        best_dice = checkpoint.get("best_dice")
        if best_dice is not None:
            print(
                f"Loaded checkpoint: {self.checkpoint_path}  "
                f"best_val_dice={best_dice:.4f}"
            )
        else:
            print(f"Loaded checkpoint: {self.checkpoint_path}")

    def _clear_decoder_aux_cache_for_inference(self) -> None:
        """Clear cached decoder aux tensors before inference to avoid stale references."""

        model_core = getattr(self.model, "module", self.model)
        if hasattr(model_core, "last_structure_prior"):
            model_core.last_structure_prior = None
        for decoder in (getattr(model_core, "decoder", None), getattr(model_core, "decoder_b", None)):
            if decoder is None:
                continue
            if hasattr(decoder, "last_decoder_feature"):
                decoder.last_decoder_feature = None
            for stage_name in ("up1", "up2", "up3", "up4"):
                up_module = getattr(decoder, stage_name, None)
                if up_module is not None and hasattr(up_module, "last_aux"):
                    up_module.last_aux = None

    def _ensure_test_loader(self):
    

        if self.test_loader is not None:
            return self.test_loader

        _, _, test_loader, _ = create_segmentation_dataloaders(
            image_path=self.args.get("image_path"),
            mask_path=self.args.get("mask_path"),
            batch_size=int(self.args.get("batch_size", 8)),
            img_size=int(self.args.get("img_size", 256)),
            normalize=bool(self.args.get("normalize", True)),
            num_workers=int(self.args.get("num_workers", 0)),
            seed=int(self.args.get("split_seed", self.args.get("seed", 42))),
            val_split=float(self.args.get("val_split", 0.1)),
            test_split=float(self.args.get("test_split", 0.1)),
            config=self.args,
        )
        self.test_loader = test_loader
        return self.test_loader

    def _select_visualization_sample_ids(self, test_loader) -> Set[str]:
        """Load or create one fixed set of representative test sample_ids."""

        if self.max_visualizations <= 0:
            return set()

        dataset = getattr(test_loader, "dataset", None)
        fixed_count = min(4, int(self.max_visualizations))
        selected_ids, json_path = get_or_create_fixed_test_visualization_sample_ids(
            test_dataset=dataset,
            config=self.args,
            num_samples=fixed_count,
        )
        selected_set = {str(item).strip() for item in selected_ids if str(item).strip()}
        if not selected_set:
            return set()

        dataset_samples = list(getattr(dataset, "samples", []) or [])
        dataset_id_set = {
            str(sample.get("sample_id", "")).strip()
            for sample in dataset_samples
            if str(sample.get("sample_id", "")).strip()
        }
        overlap = selected_set & dataset_id_set if dataset_id_set else selected_set
        if not overlap:
            print(
                "No fixed visualization sample_id matched this test split; "
                "falling back to sequential top-k."
            )
            return set()

        print(
            f"Using fixed visualization sample_ids ({len(overlap)}): "
            f"{os.path.abspath(json_path)}"
        )
        return overlap

    def _clear_previous_visualizations(self) -> None:
        """Remove stale test panels so each run keeps only the current fixed sample set."""

        if not os.path.isdir(self.visualization_dir):
            return

        for filename in os.listdir(self.visualization_dir):
            if not (
                filename.endswith("_panel.png")
            ):
                continue

            file_path = os.path.join(self.visualization_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def _clear_previous_paper_visualizations(self) -> None:
        if os.path.isdir(self.error_visualization_dir):
            for filename in os.listdir(self.error_visualization_dir):
                file_path = os.path.join(self.error_visualization_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        prefix = str(self.args.get("predict_artifact_prefix", "") or "")
        for filename in (
            f"{prefix}sample_metrics.csv",
            f"{prefix}sample_metric_distributions.png",
            f"{prefix}area_stratified_metrics.png",
            f"{prefix}area_stratified_metrics.json",
        ):
            file_path = os.path.join(self.paper_visualization_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def _export_paper_visualizations(self, sample_rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
        if not sample_rows:
            return {}

        rows_with_groups, _ = _assign_area_groups(sample_rows)
        prefix = str(self.args.get("predict_artifact_prefix", "") or "")
        csv_path = os.path.join(self.paper_visualization_dir, f"{prefix}sample_metrics.csv")
        dist_path = os.path.join(self.paper_visualization_dir, f"{prefix}sample_metric_distributions.png")
        area_plot_path = os.path.join(self.paper_visualization_dir, f"{prefix}area_stratified_metrics.png")
        area_stats_path = os.path.join(self.paper_visualization_dir, f"{prefix}area_stratified_metrics.json")

        _save_sample_metrics_csv(rows_with_groups, csv_path)
        _save_sample_metric_distribution_plot(rows_with_groups, dist_path)
        _save_area_stratified_plot(
            rows_with_groups,
            save_path=area_plot_path,
            stats_path=area_stats_path,
        )
        return {
            "sample_metrics_csv": csv_path,
            "distribution_plot": dist_path,
            "area_plot": area_plot_path,
            "area_stats": area_stats_path,
            "error_dir": self.error_visualization_dir,
        }

    def run(self) -> Dict[str, Any]:
        test_loader = self._ensure_test_loader()
        total_batches = len(test_loader)
        log_interval = int(self.args.get("batch_log_interval", 5))
        metric_meter = SegmentationMetricMeter(
            num_classes=int(self.args.get("num_classes", 1)),
            threshold=float(self.args.get("threshold", 0.5)),
            ignore_index=self.args.get("ignore_index"),
            include_background=bool(self.args.get("metric_include_background", False)),
        )

        self._clear_previous_visualizations()
        self._clear_previous_paper_visualizations()
        visualization_sample_ids = self._select_visualization_sample_ids(test_loader)

        total_samples = 0
        saved_visuals = 0
        saved_error_maps = 0
        sample_rows: List[Dict[str, Any]] = []
        num_classes = int(self.args.get("num_classes", 1))
        ignore_index = self.args.get("ignore_index")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, masks, sample_ids, _) in enumerate(test_loader):
                inputs = inputs.to(self.device, non_blocking=True)
                if num_classes <= 1:
                    masks = masks.to(self.device, non_blocking=True).float()
                else:
                    masks = masks.to(self.device, non_blocking=True).long()

                logits = _extract_primary_logits(self.model(inputs))
                preds = _logits_to_mask(
                    logits=logits,
                    num_classes=num_classes,
                    threshold=float(self.args.get("threshold", 0.5)),
                )
                targets = _target_to_mask(masks, num_classes=num_classes)

                metric_logits = _prediction_mask_to_metric_logits(
                    preds=preds,
                    num_classes=num_classes,
                )
                metric_meter.update(logits=metric_logits, targets=masks)
                batch_size = inputs.size(0)
                total_samples += batch_size

                for item_idx in range(batch_size):
                    sample_id = str(sample_ids[item_idx]).strip()
                    sample_name = _sanitize_sample_name(sample_id)
                    sample_metrics = _compute_sample_binary_metrics(
                        pred_mask=preds[item_idx],
                        target_mask=targets[item_idx],
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                    )
                    sample_rows.append(
                        {
                            "sample_id": sample_id,
                            "sample_name": sample_name,
                            **sample_metrics,
                        }
                    )

                    should_save = sample_id in visualization_sample_ids
                    if not visualization_sample_ids:
                        should_save = saved_visuals < self.max_visualizations
                    if not should_save:
                        continue

                    save_path = os.path.join(self.visualization_dir, f"{sample_name}_panel.png")
                    save_segmentation_visualization_panel(
                        image_tensor=inputs[item_idx].cpu(),
                        pred_mask=preds[item_idx].cpu(),
                        target_mask=targets[item_idx].cpu(),
                        save_path=save_path,
                        config=self.args,
                        panel_title=f"Segmentation | {sample_name}",
                    )

                    saved_visuals += 1
                    error_save_path = os.path.join(self.error_visualization_dir, f"{sample_name}_error_map.png")
                    _save_error_decomposition_panel(
                        image_tensor=inputs[item_idx].cpu(),
                        pred_mask=preds[item_idx].cpu(),
                        target_mask=targets[item_idx].cpu(),
                        save_path=error_save_path,
                        config=self.args,
                        sample_name=sample_name,
                        sample_metrics=sample_metrics,
                    )
                    saved_error_maps += 1

                if _should_log_batch(batch_idx, total_batches, log_interval):
                    metrics_now = metric_meter.compute()
                    _print_batch_status(
                        prefix="Test",
                        batch_idx=batch_idx,
                        total_batches=total_batches,
                        Dice=f"{metrics_now['dice']:.4f}",
                        mIoU=f"{metrics_now['miou']:.4f}",
                        PRE=f"{metrics_now.get('precision', metrics_now.get('pre_percent', 0.0) / 100.0):.4f}",
                        Accuracy=f"{metrics_now['accuracy']:.4f}",
                        SEN=f"{metrics_now.get('sens_percent', metrics_now['sensitivity'] * 100.0):.2f}%",
                        SPE=f"{metrics_now.get('spe_percent', metrics_now['specificity'] * 100.0):.2f}%",
                    )

        metrics = metric_meter.compute()
        metrics["samples"] = float(total_samples)
        metrics["saved_visualizations"] = float(saved_visuals)
        metrics["saved_error_maps"] = float(saved_error_maps)
        active_threshold = float(self.args.get("threshold", 0.5))
        paper_artifacts = self._export_paper_visualizations(sample_rows)

        summary_text = (
            f"Threshold={active_threshold:.3f}  "
            f"Test summary  Dice={metrics['dice']:.4f}  "
            f"mIoU={metrics['miou']:.4f}  "
            f"PRE={metrics.get('precision', metrics.get('pre_percent', 0.0) / 100.0):.4f}  "
            f"Accuracy={metrics['accuracy']:.4f}  "
            f"SEN={metrics.get('sens_percent', metrics['sensitivity'] * 100.0):.2f}%  "
            f"SPE={metrics.get('spe_percent', metrics['specificity'] * 100.0):.2f}%"
        )

        print(summary_text)
        print(f"Saved test visualization panels to: {os.path.abspath(self.visualization_dir)}")
        if paper_artifacts:
            print(f"Saved paper figures to: {os.path.abspath(self.paper_visualization_dir)}")

        prefix = str(self.args.get("predict_artifact_prefix", "") or "")
        summary_txt_path = os.path.join(self.prediction_dir, f"{prefix}test_summary.txt")
        with open(summary_txt_path, "a", encoding="utf-8") as f:
            f.write("Test result:\n")
            f.write(summary_text + "\n")
            f.write(f"Samples: {int(metrics.get('samples', 0))}\n")
            f.write(f"Saved test visualization panels to: {os.path.abspath(self.visualization_dir)}\n")
            f.write(f"Saved error decomposition panels: {int(metrics.get('saved_error_maps', 0))}\n")
            if paper_artifacts:
                f.write("Paper figures:\n")
                f.write(f"  sample_metrics_csv: {os.path.abspath(paper_artifacts['sample_metrics_csv'])}\n")
                f.write(f"  distribution_plot: {os.path.abspath(paper_artifacts['distribution_plot'])}\n")
                f.write(f"  area_plot: {os.path.abspath(paper_artifacts['area_plot'])}\n")
                f.write(f"  area_stats: {os.path.abspath(paper_artifacts['area_stats'])}\n")
                f.write(f"  error_maps_dir: {os.path.abspath(paper_artifacts['error_dir'])}\n")

        metrics_json_path = os.path.join(self.prediction_dir, f"{prefix}test_metrics.json")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {key: _json_safe_metric_value(value) for key, value in metrics.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Saved test summary to: {os.path.abspath(summary_txt_path)}")
        print(f"Saved test metrics json to: {os.path.abspath(metrics_json_path)}")

        return metrics

    def predict_single_image(self, image_path: str, save_prefix: Optional[str] = None) -> Dict[str, str]:

        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        augmentor = SegmentationAugmentor(
            img_size=int(self.args.get("img_size", 256)),
            num_classes=int(self.args.get("num_classes", 1)),
            normalize=bool(self.args.get("normalize", True)),
            mean=self.args.get("mean"),
            std=self.args.get("std"),
            mask_threshold=self.args.get("mask_threshold"),
        )
        input_tensor = augmentor.transform_image_only(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = _extract_primary_logits(self.model(input_tensor))
            pred_mask_small = _logits_to_mask(
                logits=logits,
                num_classes=int(self.args.get("num_classes", 1)),
                threshold=float(self.args.get("threshold", 0.5)),
            )
            pred_mask_small = pred_mask_small.float().unsqueeze(1)

        pred_mask = F.interpolate(pred_mask_small, size=(original_size[1], original_size[0]), mode="nearest")
        pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_mask_small = pred_mask_small.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

        prefix = save_prefix or os.path.splitext(os.path.basename(image_path))[0]
        mask_output = os.path.join(self.prediction_dir, f"{prefix}_mask.png")
        overlay_output = os.path.join(self.prediction_dir, f"{prefix}_overlay.png")

        mask_scale = 255 if int(self.args.get("num_classes", 1)) <= 1 else 1
        Image.fromarray(pred_mask * mask_scale).save(mask_output)

        resized_input = augmentor.transform_image_only(image)
        save_segmentation_visualization_panel(
            image_tensor=resized_input,
            pred_mask=torch.from_numpy(pred_mask_small),
            target_mask=None,
            save_path=overlay_output,
            config=self.args,
        )

        return {
            "mask_path": mask_output,
            "overlay_path": overlay_output,
        }


PREDICT_CONFIG: Dict[str, Any] = {
    "model_type": "DRS",
    "prediction_dir": "results/predictions",
    "batch_size": 8,
    "num_workers": 0,
    "threshold": 0.5,
    "max_test_visualizations": 4,
    "fixed_test_visualization_json": "",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "predict_full_dataset": True,
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate DRS on all labeled image-mask pairs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--output", default="results/predictions")
    args = parser.parse_args()
    config = dict(PREDICT_CONFIG, image_path=args.image_path, mask_path=args.mask_path,
                  prediction_dir=args.output)
    # Build the loader after checkpoint metadata restores image size and normalization.
    predictor = ModelPredictor(args=config, checkpoint_path=args.checkpoint)
    predictor.test_loader = _build_full_dataset_test_loader(predictor.args)
    predictor.run()


if __name__ == "__main__":
    main()
