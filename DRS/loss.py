"""分割任务损失函数。"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 张量/权重工具
# ---------------------------------------------------------------------------
def _as_float_tensor(value, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """Convert optional scalar/list/tensor weights into float tensors."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone().float()
    elif isinstance(value, (list, tuple)):
        tensor = torch.tensor(value, dtype=torch.float32)
    else:
        tensor = torch.tensor([value], dtype=torch.float32)
    if tensor.numel() == 0:
        return None
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# ---------------------------------------------------------------------------
# 标签格式归一化
# ---------------------------------------------------------------------------
def _prepare_binary_targets(targets: torch.Tensor) -> torch.Tensor:
    """Normalize binary masks to [N, 1, H, W] float tensors."""

    if targets.dim() == 2:
        targets = targets.unsqueeze(0).unsqueeze(0)
    elif targets.dim() == 3:
        targets = targets.unsqueeze(1)
    elif targets.dim() == 4 and targets.shape[1] != 1:
        raise ValueError("Binary targets must have shape [N, H, W] or [N, 1, H, W].")
    return targets.float()


def _prepare_multiclass_targets(targets: torch.Tensor) -> torch.Tensor:
    """Normalize multiclass masks to [N, H, W] long tensors."""

    if targets.dim() == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    elif targets.dim() == 4 and targets.shape[1] > 1:
        targets = torch.argmax(targets, dim=1)
    elif targets.dim() == 2:
        targets = targets.unsqueeze(0)
    return targets.long()


# ---------------------------------------------------------------------------
# Dice 基础组件
# ---------------------------------------------------------------------------
def _binary_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Dice loss for binary segmentation."""

    probs = torch.sigmoid(logits)
    targets = _prepare_binary_targets(targets)

    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth + eps)
    return 1.0 - dice.mean()


def binary_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Binary Dice loss for DRS segmentation."""

    return _binary_dice_loss(logits, targets, smooth=smooth, eps=eps)


def binary_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight=None,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Project-standard binary supervised loss: weighted BCEWithLogits + Dice."""

    targets = _prepare_binary_targets(targets).to(device=logits.device, dtype=logits.dtype)
    pos_weight_tensor = _as_float_tensor(pos_weight, device=logits.device)
    if pos_weight_tensor is not None:
        pos_weight_tensor = pos_weight_tensor.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight_tensor)
    dice = binary_dice_loss(logits, targets, smooth=smooth)
    return float(bce_weight) * bce + float(dice_weight) * dice


def _multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    include_background: bool = True,
    smooth: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Dice loss for multiclass segmentation."""

    targets = _prepare_multiclass_targets(targets)
    probs = F.softmax(logits, dim=1)

    if ignore_index is not None:
        valid_mask = targets != int(ignore_index)
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0
    else:
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        safe_targets = targets

    one_hot = F.one_hot(safe_targets.clamp(min=0, max=num_classes - 1), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    valid_mask = valid_mask.unsqueeze(1)

    probs = probs * valid_mask
    one_hot = one_hot * valid_mask

    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(one_hot, dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth + eps)

    class_indices = list(range(num_classes))
    if not include_background and num_classes > 1:
        class_indices = class_indices[1:]

    if not class_indices:
        return logits.new_tensor(0.0)

    dice = dice[class_indices]
    return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# 主分割损失封装
# ---------------------------------------------------------------------------
class BinarySegmentationLoss(nn.Module):
    """Binary combined loss: BCEWithLogits + Dice."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight=None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

        pos_weight_tensor = _as_float_tensor(pos_weight)
        if pos_weight_tensor is not None:
            self.register_buffer("pos_weight", pos_weight_tensor)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return binary_segmentation_loss(
            logits,
            targets,
            bce_weight=self.bce_weight,
            dice_weight=self.dice_weight,
            pos_weight=self.pos_weight,
            smooth=self.smooth,
        )


class MultiClassSegmentationLoss(nn.Module):
    """Multiclass combined loss: CE + Dice."""

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights=None,
        ignore_index: Optional[int] = None,
        include_background: bool = False,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ignore_index = ignore_index
        self.include_background = bool(include_background)
        self.smooth = float(smooth)

        class_weights_tensor = _as_float_tensor(class_weights)
        if class_weights_tensor is not None:
            self.register_buffer("class_weights", class_weights_tensor)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _prepare_multiclass_targets(targets).to(device=logits.device, dtype=torch.long)
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ignore_index = -100 if self.ignore_index is None else int(self.ignore_index)
        ce = F.cross_entropy(logits, targets, weight=weight, ignore_index=ignore_index)
        dice = _multiclass_dice_loss(
            logits=logits,
            targets=targets,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            include_background=self.include_background,
            smooth=self.smooth,
        )
        return self.ce_weight * ce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# 统一入口损失（二分类/多分类自动适配）
# ---------------------------------------------------------------------------
class SegmentationLoss(nn.Module):
    """Unified combined loss wrapper for segmentation."""

    def __init__(
        self,
        num_classes: int = 1,
        bce_weight: float = 1.0,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight=None,
        class_weights=None,
        ignore_index: Optional[int] = None,
        include_background: bool = False,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.last_stats: Dict[str, float] = {}

        if self.num_classes <= 1:
            self.impl = BinarySegmentationLoss(
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                pos_weight=pos_weight,
                smooth=smooth,
            )
        else:
            self.impl = MultiClassSegmentationLoss(
                num_classes=self.num_classes,
                ce_weight=ce_weight,
                dice_weight=dice_weight,
                class_weights=class_weights,
                ignore_index=ignore_index,
                include_background=include_background,
                smooth=smooth,
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = self.impl(logits, targets)
        loss_value = float(total_loss.detach().item())
        self.last_stats = {
            "loss_main": loss_value,
            "loss_total": loss_value,
        }
        return total_loss

    def get_last_stats(self) -> Dict[str, float]:
        """Return the latest scalar loss statistics."""

        return dict(self.last_stats)


# ---------------------------------------------------------------------------
# 边界监督工具
# ---------------------------------------------------------------------------
def build_boundary_target(targets: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Build a binary boundary target from binary segmentation masks."""

    ks = max(3, int(kernel_size))
    if ks % 2 == 0:
        ks += 1

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    y = targets.float()

    dilated = F.max_pool2d(y, kernel_size=ks, stride=1, padding=ks // 2)
    eroded = -F.max_pool2d(-y, kernel_size=ks, stride=1, padding=ks // 2)
    return (dilated - eroded).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# 深监督内部计算
# ---------------------------------------------------------------------------
def _compute_deep_supervision_loss(
    criterion: nn.Module,
    targets: torch.Tensor,
    aux_logits: Sequence[Optional[torch.Tensor]],
    aux_weights: Sequence[float],
    reference: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute weighted auxiliary deep-supervision losses."""

    total_aux_loss = reference.new_tensor(0.0)
    aux_losses: list[torch.Tensor] = []

    for aux_idx, aux in enumerate(aux_logits):
        weight = float(aux_weights[aux_idx]) if aux_idx < len(aux_weights) else 0.0
        if weight <= 0:
            continue
        if aux is None:
            # Some model variants disable auxiliary heads for certain stages.
            # Keep weight indexing but skip missing outputs.
            continue

        aux_resized = aux
        if aux_resized.shape[-2:] != targets.shape[-2:]:
            aux_resized = F.interpolate(aux_resized, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        aux_loss = criterion(aux_resized, targets)
        aux_losses.append(aux_loss)
        total_aux_loss = total_aux_loss + weight * aux_loss

    return total_aux_loss, aux_losses


# ---------------------------------------------------------------------------
# 训练循环使用的公开监督接口
# ---------------------------------------------------------------------------
def compute_main_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """计算主分割损失（最终输出 logits 对应的损失）。"""

    return criterion(logits, targets)


def compute_deep_supervision_loss(
    criterion: nn.Module,
    targets: torch.Tensor,
    aux_logits: Optional[Sequence[Optional[torch.Tensor]]] = None,
    deep_supervision_weights: Optional[Sequence[float]] = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[float]]:
    """计算加权深监督辅助损失，并返回实际使用的权重。"""

    aux_logits_list = list(aux_logits or [])
    weights = [float(v) for v in (deep_supervision_weights if deep_supervision_weights is not None else (1.0, 0.4, 0.3, 0.2, 0.1))]
    target_len = max(1, 1 + len(aux_logits_list))
    if len(weights) < target_len:
        weights.extend([0.0] * (target_len - len(weights)))
    else:
        weights = weights[:target_len]
    aux_weights = weights[1:]

    weighted_aux_loss, aux_losses = _compute_deep_supervision_loss(
        criterion=criterion,
        targets=targets,
        aux_logits=aux_logits_list,
        aux_weights=aux_weights,
        reference=targets,
    )

    return weighted_aux_loss, aux_losses, weights


def compute_boundary_loss(
    boundary_logit: Optional[torch.Tensor],
    targets: torch.Tensor,
    boundary_kernel_size: int = 3,
) -> torch.Tensor:
    """计算未加权的边界监督损失。"""

    if boundary_logit is None:
        return targets.new_tensor(0.0)

    boundary_target = build_boundary_target(targets, kernel_size=boundary_kernel_size).to(
        device=boundary_logit.device,
        dtype=boundary_logit.dtype,
    )
    if boundary_target.shape[-2:] != boundary_logit.shape[-2:]:
        boundary_target = F.interpolate(boundary_target, size=boundary_logit.shape[-2:], mode="nearest")
    return F.binary_cross_entropy_with_logits(boundary_logit, boundary_target)


CombinedLoss = SegmentationLoss
