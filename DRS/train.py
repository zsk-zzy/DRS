"""DRS — Dual-view Reliability-guided Supervision（双视图可靠性引导监督框架）。"""
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import torch
from semi.train_ops import run_multi_seed_experiments

# DDV-M、DDV-S、FRG 三因子和双向 RRS 始终启用。
# 默认 PH2 归一化，标注率 4%；更换数据集时同步修改 mean/std。
# 修改以下参数，或用 --config 加载 JSON 覆盖参数。
CONFIG: Dict[str, Any] = {
    # 框架名称
    "method_name": 'DRS',
    "model_type": 'DRS',
    # 数据路径与预处理（默认 PH2）
    "image_path": 'data/PH2/images',
    "mask_path": 'data/PH2/masks',
    "mean": [0.7535, 0.5766, 0.4886],
    "std": [0.1967, 0.1934, 0.1987],
    "mask_suffixes": ['', '_mask', '-mask', '_seg', '_segmentation', '_label', '_lesion', '_Segmentation'],
    "val_split": 0.1,
    "test_split": 0.2,
    "split_seed": 2026,
    "img_size": 256,
    "normalize": True,
    # 双流半监督批次
    "unlabeled_ratio": 0.96,
    "use_two_stream_batch_sampler": True,
    "labeled_batch_size": 4,
    "batch_size": 8,
    # 网络结构
    "features": [16, 24, 32, 40, 48],
    "num_classes": 1,
    "dropout": 0.0,
    # 有监督损失与评估
    "deep_supervision_weights": [1.0, 0.1, 0.2, 0.3, 0.4],
    "boundary_loss_weight": 0.1,
    "boundary_kernel_size": 3,
    "bce_weight": 1.0,
    "ce_weight": 1.0,
    "dice_weight": 1.0,
    "pos_weight": None,
    "class_weights": None,
    "threshold": 0.5,
    "metric_include_background": False,
    "ignore_index": None,
    # RRS 热身与优化器
    "rrs_warmup_epochs": 10,
    "num_epochs": 100,
    "initial_lr": 0.001,
    "weight_decay": 0.0001,
    "grad_accum_steps": 1,
    "max_grad_norm": 1.0,
    "use_amp": True,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "num_workers": 2,
    "persistent_workers": True,
    "eval_persistent_workers": False,
    "prefetch_factor": 1,
    "train_seeds": [42, 43, 44],
    "batch_log_interval": 20,
    "verbose": False,
    # 初始化、续训、独立评估与输出
    "pretrained_checkpoint_path": None,
    "resume_checkpoint_path": None,
    "checkpoint_strict_load": True,
    "test_only": False,
    "test_checkpoint_dir": None,
    "test_normalization": 'config',
    "save_freq": 10,
    "save_training_curve": True,
    "max_test_visualizations": 4,
    "results_root": 'results/DRS',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON file overriding CONFIG")
    parser.add_argument("--image-path")
    parser.add_argument("--mask-path")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--checkpoint-dir", help="Directory containing per-seed best checkpoints")
    args = parser.parse_args()
    config = dict(CONFIG)
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            overrides = json.load(handle)
        unknown = set(overrides) - set(CONFIG)
        if unknown:
            parser.error(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        config.update(overrides)
    for key in ("image_path", "mask_path"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.seeds:
        config["train_seeds"] = args.seeds
    if args.test_only:
        config["test_only"] = True
    if args.checkpoint_dir:
        config["test_checkpoint_dir"] = args.checkpoint_dir
    config.update(method_name="DRS", model_type="DRS")
    run_multi_seed_experiments(config)


if __name__ == "__main__":
    main()
