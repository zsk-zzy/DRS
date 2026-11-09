"""Segmentation-only data loading utilities."""

from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from PIL import Image
from SkinDataAugmentor import SegmentationAugmentor
from torch.utils.data import DataLoader, Dataset, Sampler


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_MASK_SUFFIXES = ("", "_mask", "-mask", "_seg", "_segmentation", "_label", "_lesion")
Sample = Dict[str, Any]


class _CyclingIndexPool:
    """Draw shuffled indices continuously without dropping stream leftovers."""

    def __init__(self, indices: Sequence[int], *, seed: int) -> None:
        self._source = [int(index) for index in indices]
        if not self._source:
            raise ValueError("Cycling index pool requires at least one index.")
        self._rng = random.Random(int(seed))
        self._order: List[int] = []
        self._cursor = 0

    def draw(self, count: int) -> List[int]:
        requested = int(count)
        if requested < 0:
            raise ValueError(f"draw count must be non-negative, got {requested}")
        result: List[int] = []
        while len(result) < requested:
            if self._cursor >= len(self._order):
                self._order = list(self._source)
                self._rng.shuffle(self._order)
                self._cursor = 0
            take = min(requested - len(result), len(self._order) - self._cursor)
            result.extend(self._order[self._cursor : self._cursor + take])
            self._cursor += take
        return result

    def state_dict(self) -> Dict[str, Any]:
        """Serialize cursor, current permutation, and private RNG state."""

        return {
            "source": list(self._source),
            "order": list(self._order),
            "cursor": int(self._cursor),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if list(state.get("source", [])) != self._source:
            raise ValueError("Cycling pool checkpoint indices do not match the current dataset split.")
        order = [int(index) for index in state.get("order", [])]
        if order and sorted(order) != sorted(self._source):
            raise ValueError("Cycling pool checkpoint permutation is invalid.")
        cursor = int(state.get("cursor", 0))
        if cursor < 0 or cursor > len(order):
            raise ValueError("Cycling pool checkpoint cursor is out of range.")
        if "rng_state" not in state:
            raise ValueError("Cycling pool checkpoint is missing rng_state.")
        self._order = order
        self._cursor = cursor
        self._rng.setstate(state["rng_state"])


class TwoStreamBatchSampler(Sampler[List[int]]):
    """Yield fixed-composition labeled/unlabeled batches.

    Both streams are independently shuffled and cycled. Stream leftovers are
    preserved between epochs, so keeping the historical number of optimizer
    steps does not permanently discard any unlabeled samples.
    """

    def __init__(
        self,
        *,
        labeled_indices: Sequence[int],
        unlabeled_indices: Sequence[int],
        batch_size: int,
        labeled_batch_size: int,
        num_batches: int,
        seed: int,
    ) -> None:
        self.batch_size = int(batch_size)
        self.labeled_batch_size = int(labeled_batch_size)
        self.unlabeled_batch_size = self.batch_size - self.labeled_batch_size
        self.num_batches = int(num_batches)
        if self.batch_size < 2:
            raise ValueError("Two-stream sampling requires batch_size >= 2.")
        if not 1 <= self.labeled_batch_size < self.batch_size:
            raise ValueError(
                "labeled_batch_size must be in [1, batch_size), got "
                f"{self.labeled_batch_size} for batch_size={self.batch_size}."
            )
        if self.num_batches <= 0:
            raise ValueError(f"num_batches must be positive, got {self.num_batches}")

        labeled = [int(index) for index in labeled_indices]
        unlabeled = [int(index) for index in unlabeled_indices]
        if not labeled:
            raise ValueError("Two-stream sampling requires labeled samples.")
        if not unlabeled:
            raise ValueError("Two-stream sampling requires unlabeled samples.")
        overlap = set(labeled).intersection(unlabeled)
        if overlap:
            raise ValueError(f"Labeled and unlabeled streams overlap at indices: {sorted(overlap)}")

        base_seed = int(seed)
        self._labeled_pool = _CyclingIndexPool(labeled, seed=base_seed * 3 + 1)
        self._unlabeled_pool = _CyclingIndexPool(unlabeled, seed=base_seed * 3 + 2)
        self._batch_rng = random.Random(base_seed * 3 + 3)

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            batch = self._labeled_pool.draw(self.labeled_batch_size)
            batch.extend(self._unlabeled_pool.draw(self.unlabeled_batch_size))
            self._batch_rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches

    def state_dict(self) -> Dict[str, Any]:
        """Capture exact labeled/unlabeled stream continuation state."""

        return {
            "batch_size": self.batch_size,
            "labeled_batch_size": self.labeled_batch_size,
            "num_batches": self.num_batches,
            "labeled_pool": self._labeled_pool.state_dict(),
            "unlabeled_pool": self._unlabeled_pool.state_dict(),
            "batch_rng_state": self._batch_rng.getstate(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        expected = {
            "batch_size": self.batch_size,
            "labeled_batch_size": self.labeled_batch_size,
            "num_batches": self.num_batches,
        }
        for name, value in expected.items():
            if int(state.get(name, -1)) != value:
                raise ValueError(
                    f"Two-stream sampler checkpoint {name} does not match: "
                    f"saved={state.get(name)!r}, current={value}."
                )
        self._labeled_pool.load_state_dict(dict(state.get("labeled_pool") or {}))
        self._unlabeled_pool.load_state_dict(dict(state.get("unlabeled_pool") or {}))
        if "batch_rng_state" not in state:
            raise ValueError("Two-stream sampler checkpoint is missing batch_rng_state.")
        self._batch_rng.setstate(state["batch_rng_state"])


def build_dataloader_kwargs(
    config: Optional[Dict],
    *,
    num_workers: int,
    shuffle: bool,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    persistent_workers: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build DataLoader kwargs with safe worker persistence for Windows.

    ``persistent_workers`` and ``prefetch_factor`` are valid only when
    ``num_workers > 0``. Keeping this logic in one place lets the main loader
    and the method-specific labeled/unlabeled loaders share the same data
    reading behavior without touching the training strategy.
    """

    cfg = dict(config or {})
    workers = max(0, int(num_workers))
    kwargs: Dict[str, Any] = {
        "shuffle": bool(shuffle),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
        "drop_last": bool(drop_last),
    }
    if workers > 0:
        if persistent_workers is None:
            persistent_workers = bool(
                cfg.get("persistent_workers", cfg.get("dataloader_persistent_workers", True))
            )
        kwargs["persistent_workers"] = bool(persistent_workers)
        prefetch_factor = cfg.get("prefetch_factor", cfg.get("dataloader_prefetch_factor", 2))
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return kwargs


def _normalize_path_key(value: str) -> str:
    return os.path.splitext(value.replace("\\", "/"))[0]


def _list_files(root_dir: str) -> List[str]:
    files: List[str] = []
    for current_root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                files.append(os.path.join(current_root, filename))
    files.sort()
    return files


def _build_mask_indices(mask_root: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    rel_index: Dict[str, str] = {}
    stem_index: Dict[str, List[str]] = {}

    for path in _list_files(mask_root):
        rel_key = _normalize_path_key(os.path.relpath(path, mask_root))
        rel_index[rel_key] = path

        stem = os.path.splitext(os.path.basename(path))[0]
        stem_index.setdefault(stem, []).append(path)

    return rel_index, stem_index


def _relative_candidate_keys(rel_key: str, suffixes: Sequence[str]) -> List[str]:
    dir_name = os.path.dirname(rel_key).replace("\\", "/").strip("/")
    stem = os.path.basename(rel_key)

    keys = [rel_key]
    for suffix in suffixes:
        suffix = str(suffix or "")
        if not suffix:
            continue
        new_stem = f"{stem}{suffix}"
        keys.append(f"{dir_name}/{new_stem}" if dir_name else new_stem)
    return keys


def _find_mask_for_image(
    image_path: str,
    image_root: str,
    rel_index: Dict[str, str],
    stem_index: Dict[str, List[str]],
    suffixes: Sequence[str],
) -> Optional[str]:
    rel_key = _normalize_path_key(os.path.relpath(image_path, image_root))
    stem = os.path.splitext(os.path.basename(image_path))[0]

    for candidate_key in _relative_candidate_keys(rel_key, suffixes):
        if candidate_key in rel_index:
            return rel_index[candidate_key]

    for candidate_stem in [stem] + [f"{stem}{suffix}" for suffix in suffixes if suffix]:
        matches = stem_index.get(candidate_stem, [])
        if len(matches) == 1:
            return matches[0]

    return None


def _read_mask_array(mask_path: str, num_classes: int) -> np.ndarray:
    """读取 mask，并在这里统一做颜色通道格式校验。"""

    with Image.open(mask_path) as mask:
        mask_array = np.asarray(mask)

    if mask_array.ndim == 3:
        color_channels = mask_array[..., :3]
        if num_classes <= 1:
            mask_array = color_channels.max(axis=2)
        else:
            if np.all(color_channels == color_channels[..., :1]):
                mask_array = color_channels[..., 0]
            else:
                raise ValueError(
                    f"Mask '{mask_path}' is color-coded. Multiclass segmentation expects single-channel class-index masks."
                )
    elif mask_array.ndim != 2:
        raise ValueError(f"Unsupported mask shape {mask_array.shape} for '{mask_path}'")

    return mask_array


def _looks_like_multiclass_labels(unique_values: np.ndarray) -> bool:
    """判断当前 mask 值是否更像多分类标签，而不是二分类前景/背景。"""

    if unique_values.size <= 2 or unique_values.size > 32:
        return False
    if not np.allclose(unique_values, np.round(unique_values)):
        return False

    label_set = {int(v) for v in np.round(unique_values).astype(np.int64).tolist()}
    return not label_set.issubset({0, 1, 255})


def _random_split(
    samples: List[Sample],
    split: str,
    seed: int,
    val_split: float,
    test_split: float,
) -> List[Sample]:
    if split == "all":
        return list(samples)

    if not samples:
        return []

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    total = len(indices)
    test_count = int(round(total * float(test_split)))
    val_count = int(round(total * float(val_split)))

    test_count = min(max(test_count, 0), total)
    val_count = min(max(val_count, 0), max(total - test_count, 0))

    test_indices = indices[:test_count]
    val_indices = indices[test_count:test_count + val_count]
    train_indices = indices[test_count + val_count:]

    split_map = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
        "trainval": train_indices + val_indices,
    }
    selected_indices = split_map.get(split, train_indices)
    return [samples[idx] for idx in selected_indices]


def _resolve_unlabeled_ratio(config: Optional[Dict]) -> float:
    cfg = config or {}
    raw_ratio = cfg.get("unlabeled_ratio", cfg.get("train_unlabeled_ratio", 0.0))
    ratio = 0.0 if raw_ratio is None else float(raw_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"unlabeled_ratio must be in [0, 1), got {ratio}")
    return ratio


def _augmentation_kwargs(config: Optional[Dict]) -> Dict:
    """Return only explicit overrides for ``SegmentationAugmentor``.

    Augmentation defaults belong to :mod:`SkinDataAugmentor`; keeping a second
    copy here would let the data loader silently drift away from that module.
    Explicit appearance overrides can still be supplied without duplicating
    every default in ``train.py``.
    """

    cfg = config or {}
    config_to_argument = {
        "aug_hflip_prob": "hflip_prob",
        "aug_vflip_prob": "vflip_prob",
        "aug_rotation_prob": "rotation_prob",
        "aug_rotation_angles": "rotation_angles",
        "strong_color_jitter_prob": "strong_color_jitter_prob",
        "strong_brightness": "strong_brightness",
        "strong_contrast": "strong_contrast",
        "strong_saturation": "strong_saturation",
        "strong_hue": "strong_hue",
        "strong_gamma_prob": "strong_gamma_prob",
        "strong_gamma_range": "strong_gamma_range",
        "strong_noise_prob": "strong_noise_prob",
        "strong_noise_sigma": "strong_noise_sigma",
    }
    return {
        argument_name: cfg[config_name]
        for config_name, argument_name in config_to_argument.items()
        if config_name in cfg
    }


def _mark_train_unlabeled_samples(samples: List[Sample], ratio: float, seed: int) -> Tuple[int, int]:
    for sample in samples:
        sample["is_labeled"] = True

    if ratio <= 0.0 or not samples:
        return len(samples), 0

    total = len(samples)
    unlabeled_count = int(round(total * ratio))
    unlabeled_count = min(max(unlabeled_count, 0), max(total - 1, 0))
    if unlabeled_count <= 0:
        return total, 0

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    unlabeled_indices = set(indices[:unlabeled_count])
    for idx, sample in enumerate(samples):
        sample["is_labeled"] = idx not in unlabeled_indices

    return total - unlabeled_count, unlabeled_count


def _build_segmentation_class_names(config: Optional[Dict]) -> List[str]:
    cfg = config or {}
    num_classes = int(cfg.get("num_classes", 1))
    class_names = cfg.get("class_names")
    if class_names:
        names = list(class_names)
        expected_length = 2 if num_classes <= 1 else num_classes
        if len(names) != expected_length:
            raise ValueError(
                f"class_names length mismatch: expected {expected_length}, got {len(names)}"
            )
        return names

    if num_classes <= 1:
        return ["background", "foreground"]
    return [f"class_{idx}" for idx in range(num_classes)]


def _inspect_segmentation_dataset(
    samples: List[Sample],
    requested_num_classes: int,
    config: Optional[Dict] = None,
) -> Dict[str, object]:
    """扫描数据集中的 mask，避免类别数或 mask 格式配置错误。"""

    if not samples:
        raise FileNotFoundError("No image-mask pairs were found for segmentation inspection.")

    if int(requested_num_classes) <= 1:
        suspicious_labels: Set[int] = set()
        for sample in samples:
            mask_array = _read_mask_array(sample["mask_path"], num_classes=1)
            if not np.isfinite(mask_array).all():
                raise ValueError(f"Mask contains non-finite values: {sample['mask_path']}")

            unique_values = np.unique(mask_array)
            if _looks_like_multiclass_labels(unique_values):
                suspicious_labels.update(int(v) for v in np.round(unique_values).astype(np.int64).tolist())

        if len(suspicious_labels) > 2:
            raise ValueError(
                "Current config uses num_classes=1, but masks look like multiclass labels "
                f"{sorted(suspicious_labels)}. Please set the correct num_classes."
            )

        class_config = dict(config or {})
        class_config["num_classes"] = 1
        return {
            "task_type": "binary",
            "num_classes": 1,
            "class_names": _build_segmentation_class_names(class_config),
        }

    observed_labels: Set[int] = set()
    for sample in samples:
        mask_array = _read_mask_array(sample["mask_path"], num_classes=requested_num_classes)
        if not np.isfinite(mask_array).all():
            raise ValueError(f"Mask contains non-finite values: {sample['mask_path']}")
        if not np.allclose(mask_array, np.round(mask_array)):
            raise ValueError(
                f"Mask '{sample['mask_path']}' contains non-integer values. Multiclass masks must store class ids."
            )

        unique_labels = np.unique(mask_array.astype(np.int64))
        observed_labels.update(int(v) for v in unique_labels.tolist())

    if not observed_labels:
        raise ValueError("No valid class ids were found in multiclass masks.")
    if min(observed_labels) < 0:
        raise ValueError(f"Mask labels must be >= 0, got {sorted(observed_labels)}")

    max_label = max(observed_labels)
    expected_labels = set(range(max_label + 1))
    if observed_labels != expected_labels:
        raise ValueError(
            "Multiclass masks must use contiguous class ids starting at 0. "
            f"Observed labels: {sorted(observed_labels)}"
        )

    inferred_num_classes = max_label + 1
    if int(requested_num_classes) != inferred_num_classes:
        raise ValueError(
            f"Configured num_classes={requested_num_classes}, but masks contain labels 0..{max_label}. "
            "Please keep them consistent."
        )

    class_config = dict(config or {})
    class_config["num_classes"] = inferred_num_classes
    return {
        "task_type": "multiclass",
        "num_classes": inferred_num_classes,
        "class_names": _build_segmentation_class_names(class_config),
    }


class SegmentationDataset(Dataset):
    """Dataset that returns aligned image-mask pairs for segmentation."""

    def __init__(
        self,
        image_folder: Optional[str] = None,
        mask_folder: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
        seed: int = 42,
        val_split: float = 0.1,
        test_split: float = 0.1,
        config: Optional[Dict] = None,
        verbose: bool = True,
    ):
        self.split = str(split)
        self.transform = transform
        self.seed = int(seed)
        self.val_split = float(val_split)
        self.test_split = float(test_split)
        self.config = config or {}
        self.verbose = bool(verbose)
        self.unlabeled_ratio = _resolve_unlabeled_ratio(self.config)
        self.return_labeled_flag = self.split == "train" and self.unlabeled_ratio > 0.0
        # 标记训练集含无标签样本；实际返回单视图还是多视图由训练方法选择。
        self.uses_weak_strong = self.split == "train" and self.return_labeled_flag
        self.num_classes = int(self.config.get("num_classes", 1))
        self.samples: List[Sample] = []
        self.labeled_count = 0
        self.unlabeled_count = 0

        if image_folder and mask_folder:
            self.samples = self._load_from_explicit_dirs(image_dir=image_folder, mask_dir=mask_folder)
        else:
            raise ValueError("You must provide both image_folder and mask_folder for segmentation loading.")

        if self.split == "train":
            self.labeled_count, self.unlabeled_count = _mark_train_unlabeled_samples(
                samples=self.samples,
                ratio=self.unlabeled_ratio,
                seed=self.seed,
            )
        else:
            for sample in self.samples:
                sample["is_labeled"] = True
            self.labeled_count = len(self.samples)

        if self.verbose:
            if self.return_labeled_flag:
                actual_ratio = self.unlabeled_count / max(1, len(self.samples))
                print(
                    f"[{self.split}] samples={len(self.samples)} "
                    f"labeled={self.labeled_count} unlabeled={self.unlabeled_count} "
                    f"unlabeled_ratio={actual_ratio:.3f}"
                )
            else:
                print(f"[{self.split}] samples={len(self.samples)}")


    def _build_samples_from_dirs(self, image_dir: str, mask_dir: str) -> List[Sample]:
        suffixes = self.config.get("mask_suffixes", list(DEFAULT_MASK_SUFFIXES))
        rel_index, stem_index = _build_mask_indices(mask_dir)

        samples: List[Sample] = []
        for image_path in _list_files(image_dir):
            mask_path = _find_mask_for_image(
                image_path=image_path,
                image_root=image_dir,
                rel_index=rel_index,
                stem_index=stem_index,
                suffixes=suffixes,
            )
            if mask_path is None:
                continue

            sample_id = _normalize_path_key(os.path.relpath(image_path, image_dir))
            samples.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "sample_id": sample_id,
                }
            )

        if not samples:
            raise FileNotFoundError(
                f"No valid image-mask pairs were found under image_dir={image_dir} and mask_dir={mask_dir}."
            )
        return samples

    def _load_from_explicit_dirs(self, image_dir: str, mask_dir: str) -> List[Sample]:
        split_image_dir = os.path.join(image_dir, self.split)
        split_mask_dir = os.path.join(mask_dir, self.split)
        if os.path.isdir(split_image_dir) and os.path.isdir(split_mask_dir):
            return self._build_samples_from_dirs(image_dir=split_image_dir, mask_dir=split_mask_dir)

        train_image_dir = os.path.join(image_dir, "train")
        train_mask_dir = os.path.join(mask_dir, "train")
        val_image_dir = os.path.join(image_dir, "val")
        val_mask_dir = os.path.join(mask_dir, "val")
        test_image_dir = os.path.join(image_dir, "test")
        test_mask_dir = os.path.join(mask_dir, "test")

        has_train_split = os.path.isdir(train_image_dir) and os.path.isdir(train_mask_dir)
        has_val_split = os.path.isdir(val_image_dir) and os.path.isdir(val_mask_dir)
        has_test_split = os.path.isdir(test_image_dir) and os.path.isdir(test_mask_dir)

        # 兼容官方只提供 train/test 的数据集（如 ISIC2016）：
        # train/val 从官方 train 里按 val_split 再切，test 直接使用官方 test。
        if has_train_split and has_test_split and not has_val_split and self.split in {"train", "val", "trainval"}:
            train_samples = self._build_samples_from_dirs(image_dir=train_image_dir, mask_dir=train_mask_dir)
            return _random_split(
                samples=train_samples,
                split=self.split,
                seed=self.seed,
                val_split=self.val_split,
                test_split=0.0,
            )

        all_samples = self._build_samples_from_dirs(image_dir=image_dir, mask_dir=mask_dir)
        return _random_split(
            samples=all_samples,
            split=self.split,
            seed=self.seed,
            val_split=self.val_split,
            test_split=self.test_split,
        )

    def set_transform(
        self,
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]],
    ) -> None:
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = str(sample["image_path"])
        mask_path = str(sample["mask_path"])
        sample_id = str(sample["sample_id"])
        is_labeled = bool(sample.get("is_labeled", True))

        with Image.open(image_path) as img:
            image = img.convert("RGB").copy()
        if is_labeled:
            with Image.open(mask_path) as m:
                mask = m.copy()
        else:
            mask = Image.new("L", image.size, 0)

        if self.transform is not None:
            transformed = self.transform(image, mask)
        else:
            augmentor = SegmentationAugmentor(
                img_size=int(self.config.get("img_size", 256)),
                num_classes=self.num_classes,
                normalize=bool(self.config.get("normalize", True)),
                mean=self.config.get("mean"),
                std=self.config.get("std"),
                mask_threshold=self.config.get("mask_threshold"),
                **_augmentation_kwargs(self.config),
            )
            transformed = augmentor.get_val_transform()(image, mask)

        if self.uses_weak_strong:
            # Semi transform returns (image_weak, image_strong, mask).
            image_weak, image_strong, mask_tensor = transformed
            return image_weak, image_strong, mask_tensor, sample_id, image_path, is_labeled

        image_tensor, mask_tensor = transformed
        if self.return_labeled_flag:
            return image_tensor, mask_tensor, sample_id, image_path, is_labeled
        return image_tensor, mask_tensor, sample_id, image_path


def create_segmentation_dataloaders(
    image_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    normalize: bool = True,
    img_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
    val_split: float = 0.1,
    test_split: float = 0.1,
    config: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    cfg = dict(config or {})
    cfg["img_size"] = int(img_size)
    cfg["normalize"] = bool(normalize)
    cfg["num_classes"] = int(cfg.get("num_classes", 1))
    cfg["image_path"] = image_path or cfg.get("image_path")
    cfg["mask_path"] = mask_path or cfg.get("mask_path")
    if not cfg.get("image_path") or not cfg.get("mask_path"):
        raise ValueError("create_segmentation_dataloaders requires both image_path and mask_path.")

    inspection_kwargs = {
        "image_folder": cfg.get("image_path"),
        "mask_folder": cfg.get("mask_path"),
        "seed": seed,
        "val_split": val_split,
        "test_split": test_split,
        "config": cfg,
        "verbose": False,
    }
    inspection_dataset = SegmentationDataset(split="all", **inspection_kwargs)
    dataset_info = _inspect_segmentation_dataset(
        samples=inspection_dataset.samples,
        requested_num_classes=cfg["num_classes"],
        config=cfg,
    )

    cfg["num_classes"] = int(dataset_info["num_classes"])
    cfg["class_names"] = list(dataset_info["class_names"])

    augmentor = SegmentationAugmentor(
        img_size=img_size,
        num_classes=cfg["num_classes"],
        normalize=normalize,
        mean=cfg.get("mean"),
        std=cfg.get("std"),
        mask_threshold=cfg.get("mask_threshold"),
        **_augmentation_kwargs(cfg),
    )

    common_kwargs = {
        "image_folder": cfg.get("image_path"),
        "mask_folder": cfg.get("mask_path"),
        "seed": seed,
        "val_split": val_split,
        "test_split": test_split,
        "config": cfg,
    }

    verbose = bool(cfg.get("verbose", True))
    train_dataset = SegmentationDataset(split="train", verbose=verbose, **common_kwargs)
    val_dataset = SegmentationDataset(split="val", verbose=verbose, **common_kwargs)
    test_dataset = SegmentationDataset(split="test", verbose=verbose, **common_kwargs)

    train_total = len(train_dataset)
    train_labeled = int(train_dataset.labeled_count)
    train_unlabeled = int(train_dataset.unlabeled_count)

    # 记录训练集有标签/无标签划分，便于训练开始时确认当前半监督比例。
    dataset_info["train_total_samples"] = int(train_total)
    dataset_info["train_labeled_samples"] = train_labeled
    dataset_info["train_unlabeled_samples"] = train_unlabeled
    dataset_info["train_requested_unlabeled_ratio"] = float(train_dataset.unlabeled_ratio)
    dataset_info["train_actual_labeled_ratio"] = float(train_labeled) / max(1, train_total)
    dataset_info["train_actual_unlabeled_ratio"] = float(train_unlabeled) / max(1, train_total)
    dataset_info["val_total_samples"] = int(len(val_dataset))
    dataset_info["test_total_samples"] = int(len(test_dataset))


    if train_dataset.uses_weak_strong:
        # 主方法复用现有 weak/strong 成对增强；两视图共享几何变换。
        train_dataset.set_transform(augmentor.get_semi_train_transform())
    else:
        train_dataset.set_transform(augmentor.get_train_transform())
    val_dataset.set_transform(augmentor.get_val_transform())
    test_dataset.set_transform(augmentor.get_test_transform())

    pin_memory = torch.cuda.is_available()
    train_loader_dataset = train_dataset
    dataset_info["train_loader_samples"] = int(train_total)

    use_two_stream_sampler = (
        bool(cfg.get("use_two_stream_batch_sampler", False))
        and train_dataset.uses_weak_strong
    )
    if use_two_stream_sampler:
        num_train_batches = (train_total + int(batch_size) - 1) // int(batch_size)
        sampler_seed = int(cfg.get("train_seed", cfg.get("seed", seed)))
        labeled_indices = [
            index
            for index, sample in enumerate(train_dataset.samples)
            if bool(sample.get("is_labeled", True))
        ]
        unlabeled_indices = [
            index
            for index, sample in enumerate(train_dataset.samples)
            if not bool(sample.get("is_labeled", True))
        ]
        labeled_batch_size = int(cfg.get("labeled_batch_size", 1))
        batch_sampler = TwoStreamBatchSampler(
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            batch_size=batch_size,
            labeled_batch_size=labeled_batch_size,
            num_batches=num_train_batches,
            seed=sampler_seed,
        )
        train_loader_kwargs = build_dataloader_kwargs(
            cfg,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=bool(
                cfg.get("persistent_workers", cfg.get("dataloader_persistent_workers", True))
            ),
        )
        train_loader_kwargs.pop("shuffle")
        train_loader_kwargs.pop("drop_last")
        train_loader = DataLoader(
            train_loader_dataset,
            batch_sampler=batch_sampler,
            **train_loader_kwargs,
        )
        dataset_info["train_batch_sampler"] = "two_stream_balanced"
        dataset_info["train_batches_per_epoch"] = int(num_train_batches)
        dataset_info["train_labeled_batch_size"] = int(labeled_batch_size)
        dataset_info["train_unlabeled_batch_size"] = int(batch_size) - int(labeled_batch_size)
        dataset_info["train_batch_sampler_seed"] = sampler_seed
    else:
        train_loader = DataLoader(
            train_loader_dataset,
            batch_size=batch_size,
            **build_dataloader_kwargs(
                cfg,
                num_workers=num_workers,
                shuffle=True,
                pin_memory=pin_memory,
                drop_last=False,
                persistent_workers=bool(
                    cfg.get("persistent_workers", cfg.get("dataloader_persistent_workers", True))
                ),
            ),
        )
        dataset_info["train_batch_sampler"] = "random_shuffle"
        dataset_info["train_batches_per_epoch"] = int(len(train_loader))
    # DRS 训练始终发出对齐的 (x_w, x_s, mask, id, path, label)
    # 六项 batch：weak 供伪标签，strong 供 DDV 学生分支。
    dataset_info["train_view_mode"] = (
        "weak_strong" if train_dataset.uses_weak_strong else
        "configured_views"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        **build_dataloader_kwargs(
            cfg,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=bool(
                cfg.get("eval_persistent_workers", cfg.get("dataloader_eval_persistent_workers", False))
            ),
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        **build_dataloader_kwargs(
            cfg,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=bool(
                cfg.get("eval_persistent_workers", cfg.get("dataloader_eval_persistent_workers", False))
            ),
        ),
    )

    return train_loader, val_loader, test_loader, dataset_info
