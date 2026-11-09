"""训练通用工具：运行配置、日志、checkpoint 和可视化输出。"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from loss import SegmentationLoss
from metric.segmentation_metrics import SegmentationMetricMeter
from utils.utils import save_segmentation_visualization_panel


def set_seed(seed: int) -> None:
    """固定 random/numpy/torch 随机源并开启确定性，保证实验可复现。"""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_run_dirs(config: Dict[str, Any], batch_tag: Optional[str] = None) -> None:
    """按 method/seed/时间戳建立本次运行目录，并把 config 落盘成 config.json。

    会写回 config 的 run_name / run_root / save_dir_model / save_dir_png /
    prediction_dir，供后续训练、checkpoint 和预测共用同一套输出路径。
    """

    tag = batch_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = str(config.get("method_name", "DRS"))
    run_name = f"{method_name}_seed{int(config.get('seed', 42))}_{tag}"
    run_root = Path(str(config.get("results_root", "results"))) / run_name
    config["run_name"] = run_name
    config["run_root"] = str(run_root)
    config["save_dir_model"] = str(run_root / "checkpoint")
    config["save_dir_png"] = str(run_root / "png")
    config["prediction_dir"] = str(run_root / "predictions")
    for key in ("save_dir_model", "save_dir_png", "prediction_dir"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)
    with (run_root / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def build_criterion(config: Dict[str, Any]) -> SegmentationLoss:
    """按配置构建分割损失（BCE/CE + Dice + 可选边界/类别权重）。"""

    return SegmentationLoss(
        num_classes=int(config.get("num_classes", 1)),
        bce_weight=float(config.get("bce_weight", 1.0)),
        ce_weight=float(config.get("ce_weight", 1.0)),
        dice_weight=float(config.get("dice_weight", 1.0)),
        pos_weight=config.get("pos_weight"),
        class_weights=config.get("class_weights"),
        ignore_index=config.get("ignore_index"),
        include_background=bool(config.get("metric_include_background", False)),
    )


def build_meter(config: Dict[str, Any]) -> SegmentationMetricMeter:
    """按配置构建分割指标累计器（Dice/mIoU/敏感度等）。"""

    return SegmentationMetricMeter(
        num_classes=int(config.get("num_classes", 1)),
        threshold=float(config.get("threshold", 0.5)),
        ignore_index=config.get("ignore_index"),
        include_background=bool(config.get("metric_include_background", False)),
    )


def resolve_unlabeled_config(config: Dict[str, Any]) -> None:
    """规范无标签比例配置；训练默认使用完整半监督流程。"""

    requested_unlabeled_ratio = float(config.get("unlabeled_ratio", 0.0) or 0.0)
    if not 0.0 <= requested_unlabeled_ratio < 1.0:
        raise ValueError(f"unlabeled_ratio must be in [0, 1), got {requested_unlabeled_ratio}")

    config["requested_unlabeled_ratio"] = requested_unlabeled_ratio
    config["effective_unlabeled_ratio"] = requested_unlabeled_ratio


def build_loader_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """生成数据加载配置，避免直接修改运行时 config。"""

    loader_config = dict(config)
    loader_config["unlabeled_ratio"] = float(config.get("effective_unlabeled_ratio", 0.0))
    return loader_config


def unpack_batch_views(batch):
    """Return weak/strong views and metadata.

    The primary weak-strong trainer consumes the six-item form
    ``(x_w, x_s, mask, sample_id, path, is_labeled)`` where x_w drives the
    pseudo-label and x_s is the DDV student input. Shorter forms are still
    accepted so single-view callers keep working.
    """

    if len(batch) == 6:
        weak, strong, masks, sample_ids, image_paths, is_labeled = batch
        return weak, strong, masks, sample_ids, image_paths, is_labeled
    if len(batch) == 5:
        inputs, masks, sample_ids, image_paths, is_labeled = batch
        return inputs, None, masks, sample_ids, image_paths, is_labeled
    if len(batch) == 4:
        inputs, masks, sample_ids, image_paths = batch
        return inputs, None, masks, sample_ids, image_paths, None
    raise ValueError(f"Expected dataloader batch with 4, 5 or 6 items, got {len(batch)}")


def unpack_batch(batch):
    """统一成 6 项结构：(inputs_weak, inputs_strong, masks, sample_ids, image_paths, is_labeled)。

    兼容三种 batch：
    - 6 项半监督 weak/strong：(img_weak, img_strong, mask, id, path, is_labeled)
    - 5 项半监督单视图：    (img, mask, id, path, is_labeled) -> strong=None
    - 4 项纯监督：          (img, mask, id, path)             -> strong=None, is_labeled=None
    """

    weak, augmented, masks, sample_ids, image_paths, is_labeled = unpack_batch_views(batch)
    return weak, augmented, masks, sample_ids, image_paths, is_labeled


def print_model_info(model: torch.nn.Module) -> None:
    """Print architecture and trainable parameter counts."""

    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    decoder_count = 2 if hasattr(model, "decoder_b") else 1
    print(f"Model: DRS (shared encoder + {decoder_count} decoder(s))")
    print(
        f"Architecture/inference params: {params:,} ({params / 1e6:.3f} M); "
        f"trainable={trainable:,}"
    )


def load_pretrained_checkpoint_if_needed(model: torch.nn.Module, config: Dict[str, Any]) -> None:
    """如果配置了 checkpoint，则用它初始化当前模型。"""

    checkpoint_path = str(
        config.get("pretrained_checkpoint_path") or config.get("warmup_checkpoint_path") or ""
    ).strip()
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"pretrained_checkpoint_path does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=config["device"], weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    strict = bool(config.get("checkpoint_strict_load", True))
    load_result = model.load_state_dict(state_dict, strict=strict)

    missing = list(getattr(load_result, "missing_keys", []) or [])
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    print(f"Loaded pretrained checkpoint: {checkpoint_path}")
    if missing or unexpected:
        print(f"Checkpoint load details: missing={missing}, unexpected={unexpected}")


def log_batch(prefix: str, batch_idx: int, total_batches: int, metrics: Dict[str, float]) -> None:
    """用单行刷新方式输出 batch 进度，避免训练日志刷屏。"""

    message = (
        f"{prefix} [{batch_idx + 1:03d}/{total_batches:03d}] "
        f"loss={metrics['loss']:.4f} dice={metrics['dice']:.4f} "
        f"PRE={metrics.get('precision', metrics.get('pre_percent', 0.0) / 100.0):.4f} "
        f"SEN={metrics.get('sens_percent', metrics.get('sensitivity', 0.0) * 100.0):.2f}% "
        f"SPE={metrics.get('spe_percent', metrics.get('specificity', 0.0) * 100.0):.2f}% "
        f"miou={metrics['miou']:.4f} acc={metrics['accuracy']:.4f}"
    )
    if "objective_loss" in metrics or "supervised_loss" in metrics:
        message += (
            f" obj_l={metrics.get('objective_loss', metrics['loss']):.4f}"
            f" sup_l={metrics.get('supervised_loss', metrics['loss']):.4f}"
        )
    end = "\n" if batch_idx + 1 >= total_batches else "\r"
    print(f"\r{message:<120}", end=end, flush=True)


def format_metric_summary(prefix: str, metrics: Dict[str, float]) -> str:
    """把一组指标格式化成统一的 Train/Val/Test 摘要字符串。"""

    return (
        f"{prefix}: "
        f"Dice={metrics.get('dice', 0.0):.4f} "
        f"mIoU={metrics.get('miou', 0.0):.4f} "
        f"PRE={metrics.get('precision', metrics.get('pre_percent', 0.0) / 100.0):.4f} "
        f"Acc={metrics.get('accuracy', 0.0):.4f} "
        f"SEN={metrics.get('sens_percent', metrics.get('sensitivity', 0.0) * 100.0):.2f}% "
        f"SPE={metrics.get('spe_percent', metrics.get('specificity', 0.0) * 100.0):.2f}%"
    )


def best_checkpoint_path(config: Dict[str, Any]) -> Path:
    """Resolve the best-checkpoint path so writers and readers cannot drift."""

    return Path(config["save_dir_model"]) / (
        f"best_model_{config.get('method_name', 'DRS')}.pth"
    )


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: Dict[str, Any],
    epoch: int,
    best_dice: float,
    val_metrics: Dict[str, Any],
    is_best: bool,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Path:
    """保存最佳 checkpoint，并按 save_freq 可选保存周期 checkpoint。"""

    checkpoint = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_dice": best_dice,
        "val_metrics": val_metrics,
        "config": dict(config),
    }
    if extra_state:
        checkpoint.update(extra_state)
    best_path = best_checkpoint_path(config)
    if is_best:
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint: {best_path}")
    save_freq = int(config.get("save_freq", 0))
    if save_freq > 0 and (epoch + 1) % save_freq == 0:
        path = Path(config["save_dir_model"]) / f"epoch_{epoch + 1}.pth"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    return best_path


def save_training_curve(history: Dict[str, list[float]], save_dir: str) -> Optional[Path]:
    """把训练/验证 loss 和主要分割指标画成训练曲线图片。"""

    if not history.get("train_loss"):
        print("Skip training curve because history is empty.")
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_dir) / "training_history.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), dpi=160)
    axes = axes.ravel()

    def plot_pair(index: int, train_key: str, val_key: str, title: str, ylabel: str) -> None:
        """在指定子图上绘制某个指标的训练/验证曲线。"""

        axes[index].plot(epochs, history.get(train_key, []), label="Train", color="tab:blue")
        axes[index].plot(epochs, history.get(val_key, []), label="Val", color="tab:orange")
        axes[index].set_title(title)
        axes[index].set_xlabel("Epoch")
        axes[index].set_ylabel(ylabel)
        axes[index].grid(True, linestyle=":", alpha=0.4)
        axes[index].legend()

    plot_pair(0, "train_loss", "val_loss", "Loss", "Loss")
    plot_pair(1, "train_dice", "val_dice", "Dice", "Dice")
    plot_pair(2, "train_miou", "val_miou", "mIoU", "mIoU")
    plot_pair(3, "train_pre_percent", "val_pre_percent", "PRE", "PRE")
    plot_pair(4, "train_accuracy", "val_accuracy", "Accuracy", "Accuracy")
    plot_pair(5, "train_sens_percent", "val_sens_percent", "SEN (%)", "SEN (%)")
    plot_pair(6, "train_spe_percent", "val_spe_percent", "SPE (%)", "SPE (%)")
    axes[7].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curve: {save_path}")
    return save_path


