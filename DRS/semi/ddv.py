"""DDV perturbations for the weak-strong DRS training pair.

Both strong branches start from ``x_s`` (geometry shared with the weak view;
appearance/color/gamma/noise applied in the dataloader). DDV-M deforms the lesion neighbourhood
and inverse-warps decoder A back to the strong coordinates; DDV-S applies an
uncertainty-guided prototype semantic decision-boundary perturbation to the
shared bottleneck feature before decoder B.
Neither branch re-applies appearance augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SRMViewOutput:
    """One DRS forward plus training-only feature/evidence tensors."""

    logits: torch.Tensor
    aux_logits: Tuple[torch.Tensor, ...]
    boundary_logit: Optional[torch.Tensor]
    decoder_feature: torch.Tensor
    structure_prior: torch.Tensor
    decoder_mid_feature: Optional[torch.Tensor] = None
    bottleneck_feature: Optional[torch.Tensor] = None
    feature_relative_change: Optional[torch.Tensor] = None

    @property
    def supervised_output(self):
        return self.logits, list(self.aux_logits), self.boundary_logit


@dataclass(frozen=True)
class MorphologyPerturbation:
    """Lesion-local image deformation and its undivided-coordinate inverse grid."""

    image: torch.Tensor
    local_weight: torch.Tensor
    boundary_region: torch.Tensor
    lesion_center_region: torch.Tensor
    forward_grid: torch.Tensor
    inverse_grid: torch.Tensor
    valid_mask: torch.Tensor
    displacement_pixels: torch.Tensor
    inverse_error_pixels: torch.Tensor
    localized_samples: torch.Tensor
    boundary_normal_disp_mean: torch.Tensor
    boundary_normal_disp_p90: torch.Tensor
    inward_disp_ratio: torch.Tensor
    outward_disp_ratio: torch.Tensor


@dataclass(frozen=True)
class DDVPerturbationParameters:
    """Fixed boundary-normal morphology and prototype-boundary settings."""

    # DDV-M: include moderately confident lesion pixels; the contour band is
    # displaced by at most four pixels along the estimated lesion normal.
    morphology_confidence_threshold: float = 0.65
    morphology_dilation_radius: int = 5
    morphology_band_sigma: float = 3.0
    morphology_max_displacement_pixels: float = 4.0
    morphology_min_displacement_pixels: float = 2.0
    morphology_inverse_iterations: int = 6
    # DDV-S: adapt the prototype-midpoint pull to estimated prototype quality
    # and cap its relative change in the uncertain feature region.
    semantic_prototype_foreground_threshold: float = 0.8
    semantic_prototype_background_threshold: float = 0.2
    semantic_lambda_min: float = 0.10
    semantic_lambda_max: float = 0.25
    semantic_min_prototype_mass: float = 1.0
    semantic_max_uncertain_relative_change: float = 0.10


DDV_PERTURBATION_PARAMETERS = DDVPerturbationParameters()
# 仅作几何有效性筛选：正反变换坐标误差不得超过半个输入像素。
MORPHOLOGY_INVERSE_MAX_ERROR_PIXELS = 0.5


@torch.no_grad()
def _morphology_alignment_validity(forward_grid, inverse_grid, base_grid):
    """检查双线性采样支撑域及正反变换闭环误差；grid 必须是未 clamp 的坐标。"""
    height, width = base_grid.shape[1:3]
    bounds = base_grid.new_tensor([1 - 1 / width, 1 - 1 / height])
    # align_corners=False：严格使用像素中心范围，排除边界复制的插值证据。
    inverse_inside = (inverse_grid.abs() <= bounds + 1e-6).all(dim=-1)
    forward_inside = (forward_grid.abs() <= bounds + 1e-6).all(dim=-1)
    support = F.grid_sample(
        forward_inside.unsqueeze(1).float(), inverse_grid,
        mode="bilinear", padding_mode="zeros", align_corners=False)
    reconstructed = F.grid_sample(
        forward_grid.permute(0, 3, 1, 2), inverse_grid,
        mode="bilinear", padding_mode="border", align_corners=False).permute(0, 2, 3, 1)
    residual_pixels = (reconstructed - base_grid) * base_grid.new_tensor([width / 2, height / 2])
    error = residual_pixels.square().sum(dim=-1).sqrt().unsqueeze(1)
    valid = (inverse_inside.unsqueeze(1) & (support >= 1 - 1e-6)
             & torch.isfinite(error) & (error <= MORPHOLOGY_INVERSE_MAX_ERROR_PIXELS))
    return valid.float(), error


def _gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("Gaussian sigma must be positive.")
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=tensor.device, dtype=tensor.dtype)
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(torch.finfo(tensor.dtype).eps)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, *kernel_2d.shape).repeat(tensor.shape[1], 1, 1, 1)
    return F.conv2d(tensor, kernel, padding=radius, groups=tensor.shape[1])


def _identity_grid(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def lesion_aware_morphology_perturbation(
    image: torch.Tensor,
    weak_probability: torch.Tensor,
    *,
    confidence_threshold: float = DDV_PERTURBATION_PARAMETERS.morphology_confidence_threshold,
    dilation_radius: int = DDV_PERTURBATION_PARAMETERS.morphology_dilation_radius,
    band_sigma: float = DDV_PERTURBATION_PARAMETERS.morphology_band_sigma,
    max_displacement_pixels: float = DDV_PERTURBATION_PARAMETERS.morphology_max_displacement_pixels,
    min_displacement_pixels: float = DDV_PERTURBATION_PARAMETERS.morphology_min_displacement_pixels,
    inverse_iterations: int = DDV_PERTURBATION_PARAMETERS.morphology_inverse_iterations,
) -> MorphologyPerturbation:
    """DDV-M: boundary-normal morphological perturbation on the strong image.

    ``weak_probability`` is the detached weak prediction; it only locates the
    lesion and its contour. A smooth, spatially varying displacement is then
    built on the narrow contour band and applied along the estimated lesion normal:

        displacement = alpha(x, y) * W_boundary * delta * normal

    ``grid_sample`` uses reverse sampling: outward sampling flow contracts lesion
    content, while inward sampling flow expands it. The lesion center and
    the background stay essentially fixed. This is pure geometry: the branch
    carries no colour/gamma/noise of its own, which lives in the strong
    view. The deformation is still a geometric intervention on ``image`` and is
    inverted by the same undivided-coordinate inverse grid as before.
    """

    if image.ndim != 4 or weak_probability.ndim != 4 or weak_probability.shape[1] != 1:
        raise ValueError(
            "image must be NCHW and weak_probability must be N1HW."
        )
    if image.shape[0] != weak_probability.shape[0] or image.shape[-2:] != weak_probability.shape[-2:]:
        raise ValueError("image and weak_probability must have matching batch and spatial shapes.")
    confidence_threshold = float(confidence_threshold)
    dilation_radius = int(dilation_radius)
    max_displacement_pixels = float(max_displacement_pixels)
    min_displacement_pixels = float(min_displacement_pixels)
    inverse_iterations = int(inverse_iterations)
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie in [0, 1].")
    if dilation_radius < 0:
        raise ValueError("dilation_radius must be non-negative.")
    if max_displacement_pixels < 0.0:
        raise ValueError("max_displacement_pixels must be non-negative.")
    if min_displacement_pixels < 0.0:
        raise ValueError("min_displacement_pixels must be non-negative.")
    if min_displacement_pixels > max_displacement_pixels:
        raise ValueError("min_displacement_pixels must not exceed max_displacement_pixels.")
    if inverse_iterations < 1:
        raise ValueError("inverse_iterations must be positive.")

    with torch.no_grad(), torch.autocast(device_type=image.device.type, enabled=False):
        probability = weak_probability.detach().float().clamp(0.0, 1.0)
        high_confident_foreground = (probability >= confidence_threshold).float()
        kernel_size = 2 * dilation_radius + 1
        dilated = F.max_pool2d(
            high_confident_foreground,
            kernel_size=kernel_size,
            stride=1,
            padding=dilation_radius,
        )
        eroded = 1 - F.max_pool2d(
            1 - high_confident_foreground,
            kernel_size=kernel_size,
            stride=1,
            padding=dilation_radius,
        )
        boundary_region = (dilated - eroded).clamp(0.0, 1.0)
        lesion_center_region = eroded * high_confident_foreground

        batch, _, height, width = image.shape
        # Normal field from the blurred mask gradient. The blurred mask also
        # serves as a signed-distance proxy: it is positive inside the lesion,
        # near zero on the contour, and zero on the background.
        blurred_mask = _gaussian_blur(high_confident_foreground, band_sigma)
        value_min = blurred_mask.flatten(1).amin(dim=1).view(batch, 1, 1, 1)
        value_max = blurred_mask.flatten(1).amax(dim=1).view(batch, 1, 1, 1)
        signed_distance = 2.0 * (blurred_mask - value_min) / (
            value_max - value_min
        ).clamp_min(1e-6) - 1.0
        # torch.gradient 按 dim=(2,3) 返回 (gradient_y, gradient_x)，而 flow 的通道
        # 约定是 channel0 = x、channel1 = y，所以必须按 (x, y) 拼接。
        gradient_y, gradient_x = torch.gradient(signed_distance, dim=(2, 3))
        # 该法是 outward normal：blurred mask 在病灶内为正、外为负，所以指向
        # signed_distance 增大的方向（−∇）即为由病灶指向背景的外法向。
        normal = torch.cat((gradient_x, gradient_y), dim=1)
        normal = -normal / normal.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        # Only the narrow contour band is displaced; the center is preserved by
        # construction and the background holds an all-zero displacement.
        band = _gaussian_blur(boundary_region, band_sigma).clamp(0.0, 1.0) * boundary_region
        local_weight = band
        # Keep one direction per sample so the contour expands or contracts,
        # while varying the magnitude smoothly between different contour arcs.
        direction = torch.where(
            torch.rand(batch, 1, 1, 1, device=image.device) < 0.5,
            -1.0,
            1.0,
        ).to(dtype=torch.float32)
        # 按内外侧分别加权：外侧像素承担主要收缩/膨胀，中心基本不变。
        side_weight = 0.35 + 0.65 * (0.5 + 0.5 * signed_distance * direction)
        amplitude = min_displacement_pixels + torch.rand(
            batch, 1, 1, 1, device=image.device, dtype=torch.float32
        ) * (max_displacement_pixels - min_displacement_pixels)
        # A low-resolution random field gives neighbouring boundary pixels
        # similar amplitudes. Center it on the boundary so the sampled base
        # strength is retained, and limit variation by the existing bounds.
        spacing = max(4, int(round(4.0 * float(band_sigma))))
        coarse_height = max(2, (height + spacing - 1) // spacing)
        coarse_width = max(2, (width + spacing - 1) // spacing)
        spatial_noise = F.interpolate(
            torch.rand(batch, 1, coarse_height, coarse_width,
                       device=image.device, dtype=torch.float32),
            size=(height, width), mode="bilinear", align_corners=False)
        boundary_count = boundary_region.flatten(1).sum(dim=1).view(batch, 1, 1, 1)
        boundary_mean = (
            (spatial_noise * boundary_region).flatten(1).sum(dim=1)
            .view(batch, 1, 1, 1)
        )
        boundary_mean = boundary_mean / boundary_count.clamp_min(1.0)
        centered_noise = (spatial_noise - boundary_mean) * boundary_region
        noise_scale = centered_noise.abs().flatten(1).amax(dim=1).view(batch, 1, 1, 1)
        normalized_noise = centered_noise / noise_scale.clamp_min(1e-6)
        variation_budget = 0.5 * torch.minimum(
            amplitude - min_displacement_pixels,
            max_displacement_pixels - amplitude)
        amplitude_map = amplitude + variation_budget * normalized_noise
        normal_disp = normal * direction * amplitude_map
        displacement_pixels = normal_disp * band * side_weight
        normalized_flow = torch.stack(
            (
                displacement_pixels[:, 0] * (2.0 / width),
                displacement_pixels[:, 1] * (2.0 / height),
            ),
            dim=1,
        )
        base_grid = _identity_grid(
            batch,
            height,
            width,
            device=image.device,
            dtype=torch.float32,
        )
        forward_grid_unclamped = base_grid + normalized_flow.permute(0, 2, 3, 1)
        forward_grid = forward_grid_unclamped.clamp(-1.0, 1.0)
        inverse_grid = base_grid.clone()
        inverse_grid_unclamped = base_grid.clone()
        for _ in range(inverse_iterations):
            sampled_flow = F.grid_sample(
                normalized_flow,
                inverse_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            inverse_grid_unclamped = base_grid - sampled_flow.permute(0, 2, 3, 1)
            inverse_grid = inverse_grid_unclamped.clamp(-1.0, 1.0)
        valid_mask, inverse_error = _morphology_alignment_validity(
            forward_grid_unclamped, inverse_grid_unclamped, base_grid)
        localized_samples = high_confident_foreground.flatten(1).any(dim=1).float()
        # 未定位到高置信病灶时没有产生 morphology view，因此该样本不能
        # 为 D_M 或 B→A RRS 提供伪造的“有效”证据。
        valid_mask = valid_mask * localized_samples.view(batch, 1, 1, 1)
        deformed = F.grid_sample(
            image.detach().float(),
            forward_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).to(dtype=image.dtype)
        # normal is outward. Because grid_sample reverse-samples the image,
        # outward sampling flow contracts lesion content (inward_disp_ratio);
        # inward sampling flow expands it (outward_disp_ratio).
        signed_along_normal = (displacement_pixels * normal).sum(dim=1, keepdim=True)
        boundary_support = (local_weight > 1e-6) & valid_mask.gt(0.0)
        directional_support = boundary_support & (signed_along_normal != 0.0)
        boundary_pixels = directional_support.flatten(1).sum(dim=1).clamp_min(1).float()
        contraction_pixels = (directional_support & (signed_along_normal > 0)).flatten(1).sum(dim=1).float()
        expansion_pixels = (directional_support & (signed_along_normal < 0)).flatten(1).sum(dim=1).float()
        inward_ratio = (contraction_pixels / boundary_pixels).view(batch, 1, 1, 1)
        outward_ratio = (expansion_pixels / boundary_pixels).view(batch, 1, 1, 1)
        boundary_normal_displacement = displacement_pixels.square().sum(dim=1, keepdim=True).sqrt()
        # Per-sample mean/p90 use only valid boundary support, never the full
        # image. Empty support returns zero and cannot produce NaN.
        boundary_disp_mean = (
            (boundary_normal_displacement * boundary_support).flatten(1).sum(dim=1)
            / boundary_support.flatten(1).sum(dim=1).clamp_min(1).float()
        ).view(batch, 1, 1, 1)
        boundary_disp_p90 = torch.stack([
            torch.quantile(boundary_normal_displacement[i][boundary_support[i]], 0.90)
            if bool(boundary_support[i].any()) else boundary_normal_displacement.new_zeros(())
            for i in range(batch)
        ]).view(batch, 1, 1, 1)

    return MorphologyPerturbation(
        image=deformed,
        local_weight=local_weight,
        boundary_region=boundary_region,
        lesion_center_region=lesion_center_region,
        forward_grid=forward_grid,
        inverse_grid=inverse_grid,
        valid_mask=valid_mask,
        displacement_pixels=displacement_pixels.square().sum(dim=1, keepdim=True).sqrt(),
        inverse_error_pixels=inverse_error,
        localized_samples=localized_samples,
        boundary_normal_disp_mean=boundary_disp_mean,
        boundary_normal_disp_p90=boundary_disp_p90,
        inward_disp_ratio=inward_ratio,
        outward_disp_ratio=outward_ratio,
    )


def _inverse_warp_logit(logit: torch.Tensor, inverse_grid: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logit.float())
    aligned_probability = F.grid_sample(
        probability,
        inverse_grid.to(device=logit.device, dtype=torch.float32),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    eps = torch.finfo(aligned_probability.dtype).eps
    return torch.logit(aligned_probability.clamp(eps, 1.0 - eps)).to(dtype=logit.dtype)


def inverse_warp_srm_view(
    view: SRMViewOutput,
    inverse_grid: torch.Tensor,
) -> SRMViewOutput:
    """Align morphology-view predictions back to the shared strong coordinates."""

    aligned_aux = tuple(_inverse_warp_logit(value, inverse_grid) for value in view.aux_logits)
    aligned_boundary = (
        _inverse_warp_logit(view.boundary_logit, inverse_grid)
        if torch.is_tensor(view.boundary_logit)
        else None
    )
    return SRMViewOutput(
        logits=_inverse_warp_logit(view.logits, inverse_grid),
        aux_logits=aligned_aux,
        boundary_logit=aligned_boundary,
        decoder_feature=view.decoder_feature,
        structure_prior=view.structure_prior,
        decoder_mid_feature=view.decoder_mid_feature,
        bottleneck_feature=view.bottleneck_feature,
        feature_relative_change=view.feature_relative_change,
    )


@torch.no_grad()
def boundary_prototype_statistics(
    weak_probability: torch.Tensor,
    feature: torch.Tensor,
    *,
    foreground_threshold: float = DDV_PERTURBATION_PARAMETERS.semantic_prototype_foreground_threshold,
    background_threshold: float = DDV_PERTURBATION_PARAMETERS.semantic_prototype_background_threshold,
    min_prototype_mass: float = DDV_PERTURBATION_PARAMETERS.semantic_min_prototype_mass,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-sample foreground/background prototypes on a feature map.

    Returns ``(prototype_fg, prototype_bg, prototype_mid, uncertainty, valid)``.
    The three prototypes keep the channel axis and the singleton
    spatial axes of the feature, i.e. ``[B, C, 1, 1]``, and ``valid`` is
    ``[B, 1, 1, 1]``; ``uncertainty`` is ``4 * P_w * (1 - P_w)``.
    Invalid samples must be skipped instead of producing NaN.
    """

    if feature.ndim != 4:
        raise ValueError("feature must be a BCHW tensor.")
    if weak_probability.ndim != 4 or weak_probability.shape[1] != 1:
        raise ValueError("weak_probability must have shape [B,1,H,W].")
    if weak_probability.shape[0] != feature.shape[0]:
        raise ValueError("weak pseudo mask and strong feature must share the batch size.")
    if not 0.0 <= background_threshold < foreground_threshold <= 1.0:
        raise ValueError("Prototype thresholds must satisfy 0 <= background < foreground <= 1.")

    pseudo = weak_probability.detach().float().clamp(0.0, 1.0)
    source = feature.detach().float()
    feature_hw = source.shape[-2:]
    resized = (
        pseudo
        if pseudo.shape[-2:] == feature_hw
        else F.interpolate(pseudo, size=feature_hw, mode="bilinear", align_corners=False)
    )
    foreground_mask = (resized > foreground_threshold).float()
    background_mask = (resized < background_threshold).float()
    # 空间维求和、通道维保留：prototype 始终是 [B, C, 1, 1]，不能把 B*C 展平。
    foreground_mass = foreground_mask.sum((2, 3), keepdim=True)
    background_mass = background_mask.sum((2, 3), keepdim=True)
    # 缺少有效 fg/bg prototype 的样本必须整体跳过，分母不能是 0。
    valid = (
        (foreground_mass >= float(min_prototype_mass))
        & (background_mass >= float(min_prototype_mass))
    )
    prototype_fg = (source * foreground_mask).sum((2, 3), keepdim=True) / foreground_mass.clamp_min(1.0)
    prototype_bg = (source * background_mask).sum((2, 3), keepdim=True) / background_mass.clamp_min(1.0)
    valid_mask = valid.view(-1, 1, 1, 1).to(dtype=torch.float32)
    prototype_fg = torch.nan_to_num(prototype_fg) * valid_mask
    prototype_bg = torch.nan_to_num(prototype_bg) * valid_mask
    prototype_mid = 0.5 * (prototype_fg + prototype_bg)
    uncertainty = 4.0 * resized * (1.0 - resized)
    return prototype_fg, prototype_bg, prototype_mid, uncertainty, valid_mask


def make_semantic_feature_perturbation(
    weak_probability: torch.Tensor,
    *,
    lambda_min: float = DDV_PERTURBATION_PARAMETERS.semantic_lambda_min,
    lambda_max: float = DDV_PERTURBATION_PARAMETERS.semantic_lambda_max,
    foreground_threshold: float = DDV_PERTURBATION_PARAMETERS.semantic_prototype_foreground_threshold,
    background_threshold: float = DDV_PERTURBATION_PARAMETERS.semantic_prototype_background_threshold,
    max_uncertain_relative_change: float = (
        DDV_PERTURBATION_PARAMETERS.semantic_max_uncertain_relative_change),
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """DDV-S: uncertainty-guided prototype semantic decision-boundary perturbation.

    On the strong bottleneck feature F_s the detached weak mask is split into a
    high-confidence foreground (``P_w > 0.8``) and background
    (``P_w < 0.2``) prototype. The perturbation

        F_s' = F_s + c * lambda * U * Q * (p_mid - F_s),
        U = 4 * P_w * (1 - P_w)

    ``Q`` is prototype separation divided by separation plus its estimated
    sampling error; ``c`` caps the feature change in the uncertain region.
    High-confidence regions stay nearly untouched. Samples without both
    prototypes are skipped.

    Diagnostics for the last applied perturbation are kept on
    ``perturb.last_stats`` (a plain dict of detached float tensors) so the
    training loop can log them without changing the model context contract;
    the returned callable keeps ``DRS``'s ``bottleneck_perturbation``
    signature unchanged.
    """

    pseudo = weak_probability.detach().float()
    lambda_min = float(lambda_min)
    lambda_max = float(lambda_max)
    max_uncertain_relative_change = float(max_uncertain_relative_change)
    if not 0.0 <= lambda_min <= lambda_max:
        raise ValueError("lambda range must satisfy 0 <= lambda_min <= lambda_max.")
    if not 0.0 <= max_uncertain_relative_change <= 1.0:
        raise ValueError("max_uncertain_relative_change must lie in [0, 1].")

    def perturb(feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if feature.shape[0] != pseudo.shape[0] or feature.device != pseudo.device:
            raise ValueError("Weak pseudo mask and strong feature must share batch and device.")
        (prototype_fg, prototype_bg, prototype_mid, uncertainty,
         valid_mask) = boundary_prototype_statistics(
            pseudo,
            feature,
            foreground_threshold=foreground_threshold,
            background_threshold=background_threshold,
        )
        lam = torch.empty(
            feature.shape[0], 1, 1, 1, device=feature.device, dtype=torch.float32
        ).uniform_(lambda_min, lambda_max)
        if uncertainty.shape[-2:] != feature.shape[-2:]:
            uncertainty = F.interpolate(
                uncertainty, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        source = feature.detach().float()
        with torch.no_grad():
            resized_pseudo = (
                pseudo if pseudo.shape[-2:] == feature.shape[-2:]
                else F.interpolate(pseudo, size=feature.shape[-2:],
                                   mode="bilinear", align_corners=False)
            )
            fg_mass = (resized_pseudo > foreground_threshold).float().sum((2, 3), keepdim=True)
            bg_mass = (resized_pseudo < background_threshold).float().sum((2, 3), keepdim=True)
            gap = (prototype_fg - prototype_bg).square().mean(dim=1, keepdim=True).sqrt()
            spatial_spread = (
                source.var(dim=(2, 3), unbiased=False, keepdim=True)
                .mean(dim=1, keepdim=True).sqrt()
            )
            standard_error = spatial_spread * (
                fg_mass.clamp_min(1).reciprocal() + bg_mass.clamp_min(1).reciprocal()
            ).sqrt()
            prototype_quality = (gap / (gap + standard_error + 1e-6)) * valid_mask
            focus = uncertainty * valid_mask
            displacement = prototype_mid - source
            raw_step = lam * focus * prototype_quality
            raw_shift = raw_step * displacement
            source_mass = (source.abs() * focus).flatten(1).sum(dim=1).clamp_min(1e-6)
            raw_relative_change = (raw_shift.abs() * focus).flatten(1).sum(dim=1) / source_mass
            cap_scale = (
                max_uncertain_relative_change / raw_relative_change.clamp_min(1e-6)
            ).clamp(max=1.0).view(-1, 1, 1, 1) * valid_mask
            step = raw_step * cap_scale
            local_step = step * displacement
            uncertain_relative_change = (
                (local_step.abs() * focus).flatten(1).sum(dim=1) / source_mass
            )
        perturbed = feature + local_step.to(dtype=feature.dtype)
        with torch.no_grad():
            sample_count = feature.shape[0]
            # 逐样本统计，使训练循环能像其它 DDV 诊断量一样按 unlabeled 选择器取值，
            # 而不是只能拿到整 batch 的标量。
            valid = valid_mask.flatten().gt(0.0)
            distances = (prototype_fg - prototype_bg).abs().mean(dim=1).flatten()
            prototype_distance = torch.where(valid, distances, torch.zeros_like(distances))
            relative_change = (
                (perturbed.detach().float() - source).abs().flatten(1).mean(dim=1)
                / (source.abs().flatten(1).mean(dim=1) + 1e-6)
            )
            perturb.last_stats = {
                "prototype_valid_ratio": valid_mask.flatten().detach().view(sample_count),
                "uncertainty_mean": uncertainty.detach().flatten(1).mean(dim=1).view(sample_count),
                "prototype_fg_bg_distance": prototype_distance.detach().view(sample_count),
                "prototype_quality": prototype_quality.detach().flatten().view(sample_count),
                "semantic_feature_change": relative_change.detach().view(sample_count),
                "semantic_uncertain_feature_change": (
                    uncertain_relative_change.detach().view(sample_count)),
                "semantic_strength_scale": cap_scale.detach().flatten().view(sample_count),
                "semantic_perturbation_strength": (
                    local_step.detach().abs().flatten(1).mean(dim=1).view(sample_count)),
            }
        return perturbed, relative_change.mean().detach()

    perturb.last_stats = None
    return perturb


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def forward_srm_view(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    bottleneck_perturbation: Optional[Callable[[torch.Tensor], object]] = None,
    decoder_branch: Optional[str] = None,
) -> SRMViewOutput:
    """Run one shared-model view and capture tensors from that exact call."""

    kwargs = {"return_context": True}
    if bottleneck_perturbation is not None:
        kwargs["bottleneck_perturbation"] = bottleneck_perturbation
    if decoder_branch is not None:
        kwargs["decoder_branch"] = decoder_branch
    context = model(image, **kwargs)
    return srm_view_from_context(context, image)


def assert_dual_decoder(model: torch.nn.Module) -> None:
    base = _unwrap_model(model)
    names = [name for name, _ in base.named_children() if name.startswith("decoder")]
    if names != ["decoder", "decoder_b"]:
        raise RuntimeError("The main method requires shared encoder + decoder/decoder_b.")


def forward_dual_srm_views(model: torch.nn.Module, image: torch.Tensor):
    """Encode once and capture both clean decoder outputs with their own features."""
    context = model(image, return_context=True)
    return (
        srm_view_from_context(context["branch_a"], image),
        srm_view_from_context(context["branch_b"], image),
    )


def srm_view_from_context(context, image: torch.Tensor) -> SRMViewOutput:
    """Validate and wrap tensors belonging to one decoder forward."""
    if not isinstance(context, dict) or "output" not in context:
        raise RuntimeError("DRS must return context when return_context=True.")
    output = context["output"]
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise RuntimeError("DRS output must be (logits, aux_logits, boundary_logit).")
    logits, aux_logits, boundary_logit = output
    decoder_feature = context.get("decoder_feature")
    decoder_mid_feature = context.get("decoder_mid_feature")
    bottleneck_feature = context.get("bottleneck_feature")
    structure_prior = context.get("structure_prior")
    relative_change = context.get("feature_relative_change")
    if not all(torch.is_tensor(value) for value in (logits, decoder_feature, structure_prior)):
        raise RuntimeError("DDV requires tensor logits, decoder_feature, and structure_prior.")
    if logits.ndim != 4 or logits.shape[0] != image.shape[0]:
        raise ValueError(f"Invalid segmentation logits shape: {tuple(logits.shape)}.")
    if logits.shape[-2:] != image.shape[-2:]:
        raise ValueError("Segmentation logits and input must have the same spatial size.")
    if decoder_feature.ndim != 4 or decoder_feature.shape[0] != image.shape[0]:
        raise ValueError(f"Invalid decoder feature shape: {tuple(decoder_feature.shape)}.")
    if structure_prior.ndim != 4 or structure_prior.shape[:2] != (image.shape[0], 1):
        raise ValueError(f"Invalid structure prior shape: {tuple(structure_prior.shape)}.")
    optional = (decoder_mid_feature, bottleneck_feature)
    if any(value is not None and not torch.is_tensor(value) for value in optional):
        raise RuntimeError("Optional SRM context features must be tensors.")
    tensors = [logits, decoder_feature, structure_prior]
    tensors.extend(value for value in optional if torch.is_tensor(value))
    tensors.extend(value for value in (aux_logits or ()) if torch.is_tensor(value))
    if torch.is_tensor(boundary_logit):
        tensors.append(boundary_logit)
    if any(value.device != image.device for value in tensors):
        raise ValueError("Every DDV output must stay on the input device.")
    if relative_change is not None:
        if not torch.is_tensor(relative_change) or relative_change.numel() != 1:
            raise ValueError("feature_relative_change must be a scalar tensor.")
        relative_change = relative_change.detach().float()

    return SRMViewOutput(
        logits=logits,
        aux_logits=tuple(aux_logits or ()),
        boundary_logit=boundary_logit,
        decoder_feature=decoder_feature,
        structure_prior=structure_prior,
        decoder_mid_feature=decoder_mid_feature,
        bottleneck_feature=bottleneck_feature,
        feature_relative_change=relative_change,
    )


__all__ = [
    "DDVPerturbationParameters",
    "DDV_PERTURBATION_PARAMETERS",
    "MorphologyPerturbation",
    "SRMViewOutput",
    "assert_dual_decoder",
    "forward_srm_view",
    "forward_dual_srm_views",
    "inverse_warp_srm_view",
    "lesion_aware_morphology_perturbation",
    "make_semantic_feature_perturbation",
]
