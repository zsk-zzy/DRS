# DRS

PyTorch implementation of **DRS: Dual-view Reliability-guided Supervision** for semi-supervised dermoscopic skin lesion segmentation.

**中文名称：双视图可靠性引导监督框架。**

DRS uses a shared **Structure-aware Local-Global Encoder (SLGE)** and two independent **Prior-guided Refinement after Fusion (PRSF)** decoders. The framework combines morphological and semantic perturbations with reliability-guided supervision to learn from a small labeled subset and a larger unlabeled subset.

The complete training pipeline includes:

- **DDV:** generates a morphological view through lesion-aware image deformation and a semantic view through prototype-based feature perturbation.
- **FRG:** combines prediction-response consistency, frequency reliability, and region-relation reliability into a shared pixel-wise reliability map.
- **RRS:** uses detached weak-view predictions to supervise the opposite decoder's perturbed predictions, weighted by the shared reliability map.

All components are enabled throughout training. Ground-truth supervision is applied to labeled weak-view predictions. Validation and inference use the mean logits of the two decoders.

## Requirements

The pinned dependencies require **Python 3.11 or later** ([NumPy 2.4.2 requirements](https://pypi.org/project/numpy/2.4.2/)). Their versions match the locally validated Python 3.12 environment. Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

A minimal environment includes:

```text
torch
torchvision
numpy
Pillow
matplotlib
```

For GPU training, install the matching PyTorch and torchvision wheels for your CUDA environment before installing the remaining requirements. For example, to use CUDA 12.8:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Other CPU/CUDA builds are listed in the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v2110). The code automatically selects CUDA when available and otherwise uses the CPU.

## Repository Structure

```text
DRS/
├── train.py
├── Predict.py
├── DataLoading.py
├── SkinDataAugmentor.py
├── loss.py
├── models/
│   ├── __init__.py
│   └── drs.py
├── semi/
│   ├── __init__.py
│   ├── ddv.py
│   ├── frg.py
│   ├── rrs.py
│   ├── schedules.py
│   └── train_ops.py
├── metric/
│   ├── __init__.py
│   └── segmentation_metrics.py
├── utils/
│   ├── __init__.py
│   ├── checkpoint_discovery.py
│   ├── training_utils.py
│   └── utils.py
├── docs/
│   ├── method.md
│   └── data_flow.md
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```

`models/drs.py` defines the shared encoder and dual decoders. The `semi/` directory implements the complete semi-supervised training pipeline. `utils/checkpoint_discovery.py` resolves checkpoints for each training seed, while `utils/training_utils.py` handles training setup, logging, checkpoint saving, and training curves.

Detailed method descriptions are available in [Method](docs/method.md) and [Data Flow](docs/data_flow.md).

## Dataset Download

Download images and segmentation masks from the dataset providers. This repository does **not** redistribute the original datasets.

| Dataset | Official website |
| --- | --- |
| ISIC2017 | [ISIC 2017 Challenge](https://challenge.isic-archive.com/landing/2017/) |
| ISIC2018 | [ISIC 2018 Challenge](https://challenge.isic-archive.com/landing/2018/) |
| PH2 | [PH2 Database](https://www.fc.up.pt/addi/ph2%20database.html) |

## Dataset Preparation

Set `image_path` and `mask_path` to the image and mask directories. Images and masks are matched by relative paths or filenames; common mask suffixes such as `_mask` and `_segmentation` are supported.

For a dataset with predefined training, validation, and test partitions, arrange the files as follows:

```text
data/DATASET/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

The loader uses the corresponding split directories directly. To preserve an official evaluation protocol, prepare all three partitions explicitly.

Flat image and mask directories are also supported. In this case, the loader creates a reproducible split using `split_seed`, `val_split`, and `test_split`. The current defaults reserve 10% of all samples for validation and 20% for testing.

The current loader expects image-mask pairs and simulates unlabeled training data by hiding a fraction of the training masks. Set `unlabeled_ratio` to control this fraction:

| Labeled training fraction | `unlabeled_ratio` |
| --- | --- |
| 1% | `0.99` |
| 2% | `0.98` |
| 4% | `0.96` |

The realized fraction may differ slightly because the sample count is an integer. Hidden training masks are excluded from supervised loss computation; validation and test masks remain available for evaluation.

## Preprocessing and Data Augmentation

Image preprocessing and paired weak/strong augmentation are implemented in `SkinDataAugmentor.py`.

Images and masks are resized to **256 × 256** pixels by default. Image resizing uses bilinear interpolation, while mask resizing uses nearest-neighbor interpolation to preserve label values. Masks are converted to binary targets for lesion segmentation.

During training, the image and mask share the same geometric transformations:

- random horizontal flipping;
- random vertical flipping;
- random rotation with angles selected from `[-30°, -15°, 15°, 30°]`.

Each training sample produces two spatially aligned views. The **weak view** applies the shared geometric transformations and normalization. The **strong view** additionally applies random brightness, contrast, saturation, and hue changes, Gamma adjustment, and Gaussian noise.

The DDV module applies the additional morphological and semantic perturbations inside the training loop. During validation and testing, preprocessing consists of resizing, tensor conversion, and image normalization.

Set `mean` and `std` to match the dataset and preprocessing used in your experiment. The default configuration uses the following PH2 normalization values:

```python
"mean": [0.7535, 0.5766, 0.4886],
"std": [0.1967, 0.1934, 0.1987],
```

## Training

Modify the relevant entries in `CONFIG` in `train.py`:

```python
CONFIG = {
    # Selected entries; retain the remaining defaults in train.py.
    "image_path": "data/PH2/images",
    "mask_path": "data/PH2/masks",
    "img_size": 256,
    "num_classes": 1,
    "features": [16, 24, 32, 40, 48],
    "unlabeled_ratio": 0.96,
    "batch_size": 8,
    "labeled_batch_size": 4,
    "num_epochs": 100,
    "rrs_warmup_epochs": 10,
    "train_seeds": [42, 43, 44],
    "results_root": "results/DRS",
}
```

Then run:

```bash
python train.py
```

The default configuration runs three seeds sequentially. Each training batch contains **4 labeled and 4 unlabeled samples**. Training uses AdamW, an initial learning rate of `1e-3`, weight decay of `1e-4`, and cosine learning-rate scheduling. The RRS loss weight follows a smoothstep warm-up over the first 10 epochs.

To specify paths and run a single seed from the command line:

```bash
python train.py --image-path data/PH2/images --mask-path data/PH2/masks --seeds 42
```

You can also override configuration entries with a JSON file:

```bash
python train.py --config config.local.json
```

For example, `config.local.json` may contain:

```json
{
  "image_path": "data/PH2/images",
  "mask_path": "data/PH2/masks",
  "unlabeled_ratio": 0.98,
  "train_seeds": [42],
  "results_root": "results/DRS_2percent"
}
```

The best model is selected by **validation Dice** and saved as:

```text
results/DRS/DRS_seed<seed>_<timestamp>/checkpoint/best_model_DRS.pth
```

Training automatically evaluates the selected checkpoint on the test split and writes a summary across the configured seeds.

To resume a DRS run, set `resume_checkpoint_path` and use the corresponding single training seed. This restores the model, optimizer, scheduler, AMP scaler, and saved sampler state. To initialize model weights without restoring training state, set `pretrained_checkpoint_path`.

## Testing

Evaluate a single checkpoint with:

```bash
python Predict.py --checkpoint path/to/best_model_DRS.pth --image-path path/to/test/images --mask-path path/to/test/masks --output results/evaluation
```

The image size and normalization settings are restored from the checkpoint. The command evaluates **all paired samples** in the supplied directories, so point it to the intended test partition.

To evaluate checkpoints for multiple seeds:

```bash
python train.py --test-only --checkpoint-dir results/DRS --image-path path/to/test/images --mask-path path/to/test/masks --seeds 42 43 44
```

The checkpoint search matches seeds in run-directory names. If multiple best checkpoints match the same seed, narrow `--checkpoint-dir` to the intended experiment. This mode evaluates the full supplied target dataset, without creating training or validation splits.

For `train.py --test-only`, `test_normalization="config"` uses the normalization values in the current configuration; set it to `"checkpoint"` to use the source checkpoint's normalization. When using configuration-based normalization, update `mean` and `std` for the target dataset.

Evaluation reports **Dice, mIoU, Precision, Accuracy, Sensitivity, and Specificity** and saves metric summaries and visualizations in the selected output directory.

## Model Complexity and Inference

Training prints the total and trainable parameter counts. To inspect the parameter count independently:

```bash
python -c "from train import CONFIG; from models.drs import build_model; model = build_model(CONFIG); print('Parameters:', sum(p.numel() for p in model.parameters()))"
```

At inference time, DRS performs one shared encoder pass and two decoder passes. Their logits are averaged before sigmoid activation and thresholding. DDV, FRG, and RRS are used only during training.

When measuring inference speed, use this evaluation forward pass and report the input resolution, batch size, device, precision, and timing procedure. Exclude model construction, data loading, host-to-device transfer, post-processing, metric computation, and file I/O from forward-only timing.

## Checkpoints and Logs

With the default `results_root`, a training run produces:

```text
results/DRS/
├── DRS_seed42_<timestamp>/
│   ├── config.json
│   ├── checkpoint/
│   │   ├── best_model_DRS.pth
│   │   └── epoch_<N>.pth
│   ├── png/
│   │   ├── training_history.json
│   │   └── training_history.png
│   └── predictions/
│       ├── test_summary.txt
│       ├── test_metrics.json
│       ├── panel/
│       ├── error_decomposition/
│       └── paper_figures/
│           └── sample_metrics.csv
├── DRS_seed43_<timestamp>/
├── DRS_seed44_<timestamp>/
└── DRS/
    └── multi_seed_summary_<timestamp>.txt
```

Periodic checkpoints are controlled by `save_freq`, training-curve images by `save_training_curve`, and the number of prediction panels by `max_test_visualizations`. Progress logs are printed to the console. Generated data, weights, outputs, and local configuration files are excluded from version control by `.gitignore`.

## Citation

The DRS citation will be added when the manuscript's bibliographic details are available.
