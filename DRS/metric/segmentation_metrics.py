"""Segmentation metrics for Dice, mIoU, Precision, accuracy, sensitivity, and specificity."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch


def _prepare_binary_targets(targets: torch.Tensor) -> torch.Tensor:
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    elif targets.dim() == 4 and targets.shape[1] != 1:
        targets = targets[:, :1]
    return targets.float()


def _prepare_multiclass_targets(targets: torch.Tensor) -> torch.Tensor:
    if targets.dim() == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    return targets.long()


class SegmentationMetricMeter:
    """Accumulates segmentation statistics across batches."""

    def __init__(
        self,
        num_classes: int = 1,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        include_background: bool = False,
        eps: float = 1e-7,
    ) -> None:
        self.num_classes = int(num_classes)
        self.threshold = float(threshold)
        self.ignore_index = ignore_index
        self.include_background = bool(include_background)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        channels = 1 if self.num_classes <= 1 else self.num_classes
        self.tp = torch.zeros(channels, dtype=torch.float64)
        self.fp = torch.zeros(channels, dtype=torch.float64)
        self.fn = torch.zeros(channels, dtype=torch.float64)
        self.tn = torch.zeros(channels, dtype=torch.float64)
        self.correct = torch.tensor(0.0, dtype=torch.float64)
        self.total = torch.tensor(0.0, dtype=torch.float64)

    def find_best_threshold(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """在验证集 logits/targets 上搜索最优二分类阈值（最大化 Dice，要求 |pred_fg - gt_fg| / gt_fg <= 1.5）。"""
        probs = torch.sigmoid(logits.detach().float())
        tgt = _prepare_binary_targets(targets.detach()).float()
        if probs.dim() == 3:
            probs = probs.unsqueeze(1)

        gt_fg_ratio = float(tgt.mean())
        best_t, best_dice = 0.5, -1.0
        for t in [i / 20 for i in range(2, 19)]:  # 0.10 ~ 0.90
            preds = (probs >= t).float()
            pred_fg_ratio = float(preds.mean())
            # 过滤预测前景比例与真实相差过大的阈值
            if gt_fg_ratio > 1e-4 and pred_fg_ratio > gt_fg_ratio * 1.5:
                continue
            tp = (preds * tgt).sum()
            fp = (preds * (1 - tgt)).sum()
            fn = ((1 - preds) * tgt).sum()
            dice = float((2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7))
            if dice > best_dice:
                best_dice, best_t = dice, t
        return best_t

    def _prepare_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes <= 1:
            return (torch.sigmoid(logits) >= self.threshold).long().squeeze(1)
        return torch.argmax(logits, dim=1).long()

    def _prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.num_classes <= 1:
            targets = _prepare_binary_targets(targets)
            return (targets >= 0.5).long().squeeze(1)
        return _prepare_multiclass_targets(targets)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = self._prepare_predictions(logits.detach())
        targets = self._prepare_targets(targets.detach())

        if self.ignore_index is not None and self.num_classes > 1:
            valid_mask = targets != int(self.ignore_index)
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        self.correct += float(((preds == targets) & valid_mask).sum().item())
        self.total += float(valid_mask.sum().item())

        if self.num_classes <= 1:
            pred_fg = (preds == 1) & valid_mask
            target_fg = (targets == 1) & valid_mask
            pred_bg = (~pred_fg) & valid_mask
            target_bg = (~target_fg) & valid_mask

            self.tp[0] += float((pred_fg & target_fg).sum().item())
            self.fp[0] += float((pred_fg & target_bg).sum().item())
            self.fn[0] += float((pred_bg & target_fg).sum().item())
            self.tn[0] += float((pred_bg & target_bg).sum().item())
            return

        for class_idx in range(self.num_classes):
            pred_mask = (preds == class_idx) & valid_mask
            target_mask = (targets == class_idx) & valid_mask
            pred_not = (~pred_mask) & valid_mask
            target_not = (~target_mask) & valid_mask

            self.tp[class_idx] += float((pred_mask & target_mask).sum().item())
            self.fp[class_idx] += float((pred_mask & target_not).sum().item())
            self.fn[class_idx] += float((pred_not & target_mask).sum().item())
            self.tn[class_idx] += float((pred_not & target_not).sum().item())

    def _metric_indices(self) -> Iterable[int]:
        if self.num_classes <= 1:
            return [0]
        indices = list(range(self.num_classes))
        if not self.include_background and self.num_classes > 1:
            indices = indices[1:]
        return indices

    def compute(self) -> Dict[str, float]:
        indices = list(self._metric_indices())
        if not indices:
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
                "dice_per_class": [],
                "iou_per_class": [],
                "precision_per_class": [],
                "sensitivity_per_class": [],
                "specificity_per_class": [],
            }

        tp = self.tp[indices]
        fp = self.fp[indices]
        fn = self.fn[indices]
        tn = self.tn[indices]

        dice = (2.0 * tp + self.eps) / (2.0 * tp + fp + fn + self.eps)
        iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        total_valid = float(self.total.item())
        accuracy = float((self.correct / self.total).item()) if total_valid > 0 else 0.0
        sensitivity = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)
        mean_iou = float(iou.mean().item())
        mean_precision = float(precision.mean().item())
        mean_sensitivity = float(sensitivity.mean().item())
        mean_specificity = float(specificity.mean().item())

        return {
            "dice": float(dice.mean().item()),
            "miou": mean_iou,
            "precision": mean_precision,
            "pre_percent": mean_precision * 100.0,
            "accuracy": accuracy,
            "sensitivity": mean_sensitivity,
            "specificity": mean_specificity,
            "sens_percent": mean_sensitivity * 100.0,
            "spe_percent": mean_specificity * 100.0,
            "dice_per_class": [float(v) for v in dice.tolist()],
            "iou_per_class": [float(v) for v in iou.tolist()],
            "precision_per_class": [float(v) for v in precision.tolist()],
            "sensitivity_per_class": [float(v) for v in sensitivity.tolist()],
            "specificity_per_class": [float(v) for v in specificity.tolist()],
        }


def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 1,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    include_background: bool = False,
) -> Dict[str, float]:
    meter = SegmentationMetricMeter(
        num_classes=num_classes,
        threshold=threshold,
        ignore_index=ignore_index,
        include_background=include_background,
    )
    meter.update(logits, targets)
    return meter.compute()
