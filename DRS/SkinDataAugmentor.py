# -*- coding: utf-8 -*-
"""Segmentation augmentation utilities used by the current U-Net pipeline."""

from __future__ import annotations

import random
import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

__all__ = ["SegmentationAugmentor"]


class SegmentationAugmentor:
    """Paired image-mask augmentation for segmentation samples."""

    def __init__(
        self,
        img_size: int = 256,
        num_classes: int = 1,
        normalize: bool = True,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        mask_threshold: Optional[int] = None,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotation_prob: float = 0.5,
        rotation_angles: Sequence[float] = (-30.0, -15.0, 15.0, 30.0),
        strong_color_jitter_prob: float = 0.8,
        strong_brightness: float = 0.4,
        strong_contrast: float = 0.4,
        strong_saturation: float = 0.4,
        strong_hue: float = 0.1,
        strong_gamma_prob: float = 0.5,
        strong_gamma_range: Sequence[float] = (0.7, 1.5),
        strong_noise_prob: float = 0.5,
        strong_noise_sigma: Sequence[float] = (0.01, 0.05),
    ):
        self.img_size = int(img_size)
        self.num_classes = int(num_classes)
        self.normalize = bool(normalize)
        self.mean = list(mean if mean is not None else [0.5, 0.5, 0.5])
        self.std = list(std if std is not None else [0.5, 0.5, 0.5])
        self.mask_threshold = None if mask_threshold is None else float(mask_threshold)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotation_prob = float(rotation_prob)
        self.rotation_angles = [float(value) for value in rotation_angles]
        self.strong_color_jitter_prob = float(strong_color_jitter_prob)
        self.strong_brightness = float(strong_brightness)
        self.strong_contrast = float(strong_contrast)
        self.strong_saturation = float(strong_saturation)
        self.strong_hue = float(strong_hue)
        self.strong_gamma_prob = float(strong_gamma_prob)
        self.strong_gamma_range = tuple(float(value) for value in strong_gamma_range)
        self.strong_noise_prob = float(strong_noise_prob)
        self.strong_noise_sigma = tuple(float(value) for value in strong_noise_sigma)

        probability_names = (
            "hflip_prob",
            "vflip_prob",
            "rotation_prob",
            "strong_color_jitter_prob",
            "strong_gamma_prob",
            "strong_noise_prob",
        )
        for name in probability_names:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if self.rotation_prob > 0.0 and not self.rotation_angles:
            raise ValueError("rotation_angles cannot be empty when rotation_prob > 0.")
        for name in (
            "strong_gamma_range",
            "strong_noise_sigma",
        ):
            values = getattr(self, name)
            if len(values) != 2 or values[0] > values[1]:
                raise ValueError(f"{name} must contain an ordered [min, max] pair, got {values}.")
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain finite values, got {values}.")
    def _to_mask_array(self, mask: Image.Image) -> np.ndarray:
        """Convert a mask image into a single-channel array."""

        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            color_channels = mask_array[..., :3]
            if self.num_classes <= 1:
                mask_array = color_channels.max(axis=2)
            else:
                if np.all(color_channels == color_channels[..., :1]):
                    mask_array = color_channels[..., 0]
                else:
                    raise ValueError(
                        "Multiclass segmentation expects single-channel class-index masks, "
                        "but got a color mask. Please convert the mask to label ids first."
                    )
        elif mask_array.ndim != 2:
            raise ValueError(f"Unsupported mask shape: {mask_array.shape}")
        return mask_array

    def _compute_otsu_threshold(self, mask_array: np.ndarray) -> float:
        """Estimate a binary threshold automatically from the mask histogram."""

        flat = mask_array.astype(np.float32).reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return 0.0

        value_min = float(flat.min())
        value_max = float(flat.max())
        if value_max <= value_min:
            return 0.0 if value_max <= 0 else value_min / 2.0

        bin_count = 256
        if np.allclose(flat, np.round(flat)):
            value_span = int(round(value_max - value_min))
            if 1 < value_span < 256:
                bin_count = value_span + 1

        hist, bin_edges = np.histogram(flat, bins=bin_count, range=(value_min, value_max))
        hist = hist.astype(np.float64)
        if hist.sum() == 0:
            return (value_min + value_max) / 2.0

        probabilities = hist / hist.sum()
        cumulative_prob = np.cumsum(probabilities)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        cumulative_mean = np.cumsum(probabilities * bin_centers)
        global_mean = cumulative_mean[-1]

        between_class_var = np.zeros_like(cumulative_prob)
        valid = (cumulative_prob > 0.0) & (cumulative_prob < 1.0)
        between_class_var[valid] = (
            (global_mean * cumulative_prob[valid] - cumulative_mean[valid]) ** 2
        ) / (cumulative_prob[valid] * (1.0 - cumulative_prob[valid]))

        return float(bin_centers[int(np.argmax(between_class_var))])

    def _resolve_binary_threshold(self, mask_array: np.ndarray) -> float:
        """Pick a binary threshold automatically unless the user overrides it."""

        if self.mask_threshold is not None:
            return float(self.mask_threshold)

        unique_values = np.unique(mask_array)
        if unique_values.size <= 1:
            return 0.0
        if unique_values.size == 2:
            return float(unique_values.astype(np.float32).mean())
        return self._compute_otsu_threshold(mask_array)

    def _mask_to_tensor(self, mask: Image.Image) -> torch.Tensor:
        mask_array = self._to_mask_array(mask)

        if self.num_classes <= 1:
            threshold = self._resolve_binary_threshold(mask_array)
            binary = (mask_array > threshold).astype(np.float32)
            return torch.from_numpy(binary).unsqueeze(0)

        multiclass = mask_array.astype(np.int64)
        if multiclass.size > 0:
            min_label = int(multiclass.min())
            max_label = int(multiclass.max())
            if min_label < 0:
                raise ValueError(f"Mask labels must be >= 0, got {min_label}")
            if max_label >= self.num_classes:
                raise ValueError(
                    f"Mask label value {max_label} exceeds configured num_classes={self.num_classes}."
                )
        return torch.from_numpy(multiclass)

    def _finalize(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image_tensor = TF.to_tensor(image)
        if self.normalize:
            image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)
        mask_tensor = self._mask_to_tensor(mask)
        return image_tensor, mask_tensor

    def _resize_pair(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        image = TF.resize(image, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
        return image, mask

    def _apply_train(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep requested order:
        # Resize -> HFlip -> VFlip -> Rotation(degrees=30) -> ToTensor -> Normalize.
        image, mask = self._resize_pair(image, mask)

        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < self.vflip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if random.random() < self.rotation_prob:
            angle = random.choice(self.rotation_angles)
            image = TF.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST, fill=0)

        return self._finalize(image, mask)

    def _apply_eval(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._resize_pair(image, mask)
        return self._finalize(image, mask)

    def _apply_strong_photometric(self, image: Image.Image) -> Image.Image:
        """Non-geometric strong perturbations for the strong view.

        Only photometric changes and noise are applied here so the strong view
        stays spatially aligned with the weak view (and thus the pseudo label).
        Geometric transforms are handled once in ``_apply_weak_strong``.
        """
        # Color jitter (brightness/contrast/saturation/hue).
        if random.random() < self.strong_color_jitter_prob:
            brightness = 1.0 + random.uniform(-self.strong_brightness, self.strong_brightness)
            contrast = 1.0 + random.uniform(-self.strong_contrast, self.strong_contrast)
            saturation = 1.0 + random.uniform(-self.strong_saturation, self.strong_saturation)
            hue = random.uniform(-self.strong_hue, self.strong_hue)
            image = TF.adjust_brightness(image, max(0.1, brightness))
            image = TF.adjust_contrast(image, max(0.1, contrast))
            image = TF.adjust_saturation(image, max(0.1, saturation))
            image = TF.adjust_hue(image, hue)
        # Gamma.
        if random.random() < self.strong_gamma_prob:
            gamma = random.uniform(*self.strong_gamma_range)
            image = TF.adjust_gamma(image, gamma)
        # Gaussian noise.
        if random.random() < self.strong_noise_prob:
            sigma = random.uniform(*self.strong_noise_sigma)
            image_np = np.asarray(image).astype(np.float32) / 255.0
            noise = np.random.normal(loc=0.0, scale=sigma, size=image_np.shape).astype(np.float32)
            image_np = np.clip(image_np + noise, 0.0, 1.0)
            image = Image.fromarray((image_np * 255.0).astype(np.uint8))
        return image

    def _apply_weak_strong(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build aligned weak/strong views sharing one set of geometric params.

        weak  = geometry + normalize
        strong = geometry + non-geometric photometric perturbations + normalize
        mask  = follows geometry only (shared by both views).

        Returns (image_weak, image_strong, mask).
        """
        # 1) One shared geometric transform (resize -> hflip -> vflip -> rotate),
        #    sampled once so weak/strong/mask stay pixel aligned.
        image, mask = self._resize_pair(image, mask)
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < self.vflip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if random.random() < self.rotation_prob:
            angle = random.choice(self.rotation_angles)
            image = TF.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST, fill=0)

        # 2) weak = geometry only; strong = geometry + extra photometric noise.
        image_strong = self._apply_strong_photometric(image)

        image_weak_tensor = TF.to_tensor(image)
        image_strong_tensor = TF.to_tensor(image_strong)
        if self.normalize:
            image_weak_tensor = TF.normalize(image_weak_tensor, mean=self.mean, std=self.std)
            image_strong_tensor = TF.normalize(image_strong_tensor, mean=self.mean, std=self.std)
        mask_tensor = self._mask_to_tensor(mask)
        return image_weak_tensor, image_strong_tensor, mask_tensor


    def get_train_transform(self) -> Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]:
        return self._apply_train

    def get_semi_train_transform(
        self,
    ) -> Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Paired weak/strong transform for semi-supervised training."""

        return self._apply_weak_strong


    def get_val_transform(self) -> Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]:
        return self._apply_eval

    def get_test_transform(self) -> Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]:
        return self._apply_eval

    def transform_image_only(self, image: Image.Image) -> torch.Tensor:
        """Resize and normalize an image for single-image inference."""

        image = TF.resize(image, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
        image_tensor = TF.to_tensor(image)
        if self.normalize:
            image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)
        return image_tensor
