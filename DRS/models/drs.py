"""DRS: shared SRM encoder and two independent PRSF decoders.

This file merges the required U-Net blocks, SLGE encoder, and PRSF decoder into
one importable module. `build_model()` always builds the full DRS variant.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):
    """Downscaling with a fixed DRS encoder block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_block,
    ):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscale + skip fusion + PRSF refinement."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        upsample_block=None,
        upsample_block_kwargs: dict | None = None,
        skip_fusion_mode: str = "concat",
        low_channels: int | None = None,
        skip_channels: int | None = None,
    ):
        super().__init__()

        self.last_aux = None
        self.skip_fusion_mode = str(skip_fusion_mode or "concat").lower()
        if self.skip_fusion_mode not in {"concat", "add"}:
            raise ValueError(f"skip_fusion_mode must be 'concat' or 'add', got: {skip_fusion_mode}")
        if upsample_block is None:
            raise ValueError("DRS Up requires a PRSF upsample_block.")

        if upsample_block_kwargs is None:
            upsample_block_kwargs = {}

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            low_channels = int(low_channels) if low_channels is not None else in_channels // 2
            skip_channels = int(skip_channels) if skip_channels is not None else in_channels // 2
        else:
            low_channels = int(low_channels) if low_channels is not None else in_channels
            skip_channels = int(skip_channels) if skip_channels is not None else in_channels // 2
            self.up = nn.ConvTranspose2d(
                low_channels,
                low_channels,
                kernel_size=2,
                stride=2,
            )

        if self.skip_fusion_mode == "concat":
            self.skip_align = nn.Identity()
            fused_channels = low_channels + skip_channels
        else:
            self.skip_align = (
                nn.Identity()
                if low_channels == skip_channels
                else nn.Conv2d(skip_channels, low_channels, kernel_size=1, bias=False)
            )
            fused_channels = low_channels

        custom_kwargs = {
            key: value
            for key, value in upsample_block_kwargs.items()
            if key not in {"low_channels", "skip_channels"}
        }
        self.custom_up = upsample_block(
            in_channels=fused_channels,
            out_channels=out_channels,
            **custom_kwargs,
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        structure_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.last_aux = None

        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(
                x1,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        if self.skip_fusion_mode == "concat":
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1 + self.skip_align(x2)

        out = self.custom_up(x, structure_prior=structure_prior)
        if isinstance(out, tuple):
            feat, aux = out
            self.last_aux = aux
            return feat
        self.last_aux = None
        return out


class OutConv(nn.Module):
    """Final 1x1 projection to the required output channels."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# -----------------------------------------------------------------------------
# SLGE: Structure-aware Local-Global Encoder
# -----------------------------------------------------------------------------
def _eca_kernel_size(channels: int, gamma: int = 2, b: int = 1) -> int:
    channels = max(1, int(channels))
    k = int(abs((torch.log2(torch.tensor(float(channels))).item() + b) / gamma))
    k = max(3, k)
    if k % 2 == 0:
        k += 1
    return k


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU() if act else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AsymmetricDilatedDWBranch(nn.Module):
    """Asymmetric depthwise branch: (1x3 DW) -> (3x1 DW), with dilation."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        c = int(channels)
        d = max(1, int(dilation))
        self.conv_1x3 = ConvBNAct(
            c,
            c,
            kernel_size=(1, 3),
            padding=(0, d),
            dilation=(1, d),
            groups=c,
            act=True,
        )
        self.conv_3x1 = ConvBNAct(
            c,
            c,
            kernel_size=(3, 1),
            padding=(d, 0),
            dilation=(d, 1),
            groups=c,
            act=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_1x3(x)
        y = self.conv_3x1(y)
        return y


class MultiDilatedDSConv(nn.Module):
    """Asymmetric multi-dilation local mixer with input-adaptive branch weighting."""

    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 3)) -> None:
        super().__init__()
        c = int(channels)
        self.branches = nn.ModuleList(
            [AsymmetricDilatedDWBranch(c, dilation=d) for d in dilations]
        )
        self.branch_weights = nn.Parameter(torch.ones(len(self.branches)))
        hidden = max(4, c // 8)
        self.branch_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, len(self.branches), kernel_size=1, bias=True),
        )
        self.pw = ConvBNAct(c, c, kernel_size=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.branches) == 0:
            return self.pw(x)

        static_logits = self.branch_weights.view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
        dynamic_logits = self.branch_router(x).to(dtype=x.dtype)
        weights = torch.softmax(static_logits + dynamic_logits, dim=1)

        y = torch.zeros_like(x)
        for branch_idx, branch in enumerate(self.branches):
            branch_weight = weights[:, branch_idx:branch_idx + 1]
            y = y + branch_weight * branch(x)
        return self.pw(y)


class LMMB(nn.Module):
    """
    Local multi-scale receptive-field module.

    stage_mode:
        lite        -> DW3x3 + PW1x1
        full/global -> DW3x3 + MultiDilatedDSConv(1,4,5) + PW1x1
    """

    def __init__(
        self,
        channels: int,
        stage_mode: str = "full",
        dilations: tuple[int, ...] = (1, 3, 5),
        res_init: float = -1.0,
    ) -> None:
        super().__init__()
        c = int(channels)
        mode = str(stage_mode or "full").strip().lower()
        if mode not in {"lite", "full", "global"}:
            raise ValueError(f"AsymMSRFBlock stage_mode must be lite/full/global, got: {stage_mode}")
        dilation_tuple = tuple(int(d) for d in dilations)
        use_multiscale = mode in {"full", "global"} or (mode == "lite" and len(dilation_tuple) > 1)

        if use_multiscale:
            self.block = nn.Sequential(
                ConvBNAct(
                    c,
                    c,
                    kernel_size=3,
                    padding=1,
                    dilation=1,
                    groups=c,
                    act=True,
                ),
                MultiDilatedDSConv(c, dilations=dilation_tuple),
                ConvBNAct(c, c, kernel_size=1, act=True),
            )
        else:
            self.block = nn.Sequential(
                ConvBNAct(
                    c,
                    c,
                    kernel_size=3,
                    padding=1,
                    dilation=1,
                    groups=c,
                    act=True,
                ),
                ConvBNAct(c, c, kernel_size=1, act=True),
            )

        self.res_scale_logit = nn.Parameter(torch.tensor(float(res_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.block(x)
        scale = torch.sigmoid(self.res_scale_logit).to(device=x.device, dtype=x.dtype)
        return x + scale * delta


class SpatialChannelRecalibrationAttention(nn.Module):
    """Unified Spatial-Channel Recalibration Attention (SCRA)."""

    def __init__(
        self,
        channels: int,
        spatial_init: float = -1.0,
        channel_init: float = -2.4,
    ) -> None:
        super().__init__()
        c = int(channels)
        spatial_mid = max(4, c // 32)
        kernel_size = _eca_kernel_size(c)

        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(2, spatial_mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(spatial_mid),
            nn.GELU(),
        )
        self.spatial_local_dw = nn.Sequential(
            nn.Conv2d(
                spatial_mid,
                spatial_mid,
                kernel_size=3,
                padding=1,
                groups=spatial_mid,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_mid),
            nn.GELU(),
        )
        self.spatial_out = nn.Conv2d(spatial_mid, 1, kernel_size=1, bias=True)
        self.channel_stat_logits = nn.Parameter(torch.ones(2))
        self.channel_gate = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.spatial_scale_logit = nn.Parameter(torch.tensor(float(spatial_init)))
        self.channel_scale_logit = nn.Parameter(torch.tensor(float(channel_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_stats = x.float()
        avg_map = torch.mean(spatial_stats, dim=1, keepdim=True)
        max_map = torch.amax(spatial_stats, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_map, max_map], dim=1).to(dtype=x.dtype)
        spatial_seed = self.spatial_reduce(spatial_input)
        local_feat = self.spatial_local_dw(spatial_seed)
        spatial_feat = local_feat
        spatial_gate = torch.sigmoid(self.spatial_out(spatial_feat))

        stats_source = x.float()
        avg = stats_source.mean(dim=(-2, -1), keepdim=True)
        maxv = stats_source.amax(dim=(-2, -1), keepdim=True)
        stat_weights = torch.softmax(self.channel_stat_logits, dim=0).to(device=x.device, dtype=stats_source.dtype)
        fused = (
            stat_weights[0] * avg
            + stat_weights[1] * maxv
        )
        channel_desc = fused.squeeze(-1).permute(0, 2, 1)
        channel_gate = torch.sigmoid(self.channel_gate(channel_desc)).permute(0, 2, 1).unsqueeze(-1)
        channel_gate = channel_gate.to(device=x.device, dtype=x.dtype)

        spatial_scale = torch.sigmoid(self.spatial_scale_logit).to(device=x.device, dtype=x.dtype)
        channel_scale = torch.sigmoid(self.channel_scale_logit).to(device=x.device, dtype=x.dtype)
        gate = (
            1.0
            + spatial_scale * (spatial_gate - 0.5)
            + channel_scale * (channel_gate - 0.5)
        )
        return x * gate


class GlobalContextCompensation(nn.Module):
    """Global Context Compensation (GCC) for the deepest stage."""

    def __init__(self, channels: int, context_init: float = -2.0) -> None:
        super().__init__()
        c = int(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.context_proj = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.scale_logit = nn.Parameter(torch.tensor(float(context_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context_gate = torch.tanh(self.context_proj(self.pool(x))).to(device=x.device, dtype=x.dtype)
        context_scale = torch.sigmoid(self.scale_logit).to(device=x.device, dtype=x.dtype)
        return x * (1.0 + context_scale * context_gate)


class StructurePriorHead(nn.Module):
    """Generate structure logit/prior from local feature edge response."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(4, c // 4)
        self.local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.head = nn.Sequential(
            ConvBNAct(c, hidden, kernel_size=1, act=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge = torch.abs(local - self.local_pool(local))
        # Blend edge with local semantics to stabilize structure prediction.
        edge = edge + 0.5 * local
        structure_logit = self.head(edge)
        structure_prior = torch.sigmoid(structure_logit).to(dtype=local.dtype)
        return structure_logit, structure_prior


class SLGE(nn.Module):
    """
    SLGE: Structure-aware Local-Global Encoder block.

    Input x ->
        1x1 projection ->
        LMMB ->
        StructurePriorHead (for decoder prior cache) ->
        SCRA ->
        GCC (global only) ->
        1x1 fuse -> projection-anchored residual output.

    The structure probability is cached as
    `last_structure_prior` for decoder-side usage when available.
"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        min_channels: int = 2,
        low_init: float = -2.0,
        shape_init: float = -2.0,
        context_init: float = -2.0,
        stage_mode: str = "full",
        use_global: bool = False,
        global_pool_size: int = 8,
        dilations: tuple[int, ...] = (1, 4, 5),
        use_lmmb: bool = True,
        use_scra: bool = True,
        use_gcc: bool = True,
    ) -> None:
        super().__init__()
        _ = min_channels  # Kept for compatibility with previous encoder block interface.
        _ = low_init
        _ = use_global
        _ = global_pool_size

        inner_channels = int(out_channels)
        if inner_channels <= 0:
            raise ValueError(f"inner_channels must be positive, got {inner_channels}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        mode = str(stage_mode or "full").strip().lower()
        if mode not in {"lite", "full", "global"}:
            raise ValueError(f"SLGE stage_mode must be one of lite/full/global, got: {stage_mode}")
        self.stage_mode = mode
        self.use_structure_prior = True
        self.use_global_context = self.stage_mode == "global"
        self.use_lmmb = bool(use_lmmb)
        self.use_scra = bool(use_scra)
        self.use_gcc = bool(use_gcc)

        self.proj = ConvBNAct(self.in_channels, inner_channels, kernel_size=1, act=True)
        self.local_module = (
            LMMB(
                channels=inner_channels,
                stage_mode=self.stage_mode,
                dilations=tuple(int(d) for d in dilations),
                res_init=-1.0,
            )
            if self.use_lmmb
            else nn.Identity()
        )
        self.structure_module = StructurePriorHead(inner_channels)
        self.sc_attention = (
            SpatialChannelRecalibrationAttention(
                channels=inner_channels,
                spatial_init=-1.0,
            )
            if self.use_scra
            else nn.Identity()
        )
        self.global_context = (
            GlobalContextCompensation(inner_channels, context_init=float(context_init))
            if self.use_global_context and self.use_gcc
            else None
        )

        self.fuse = ConvBNAct(inner_channels, self.out_channels, kernel_size=1, act=True)
        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )

        self.res_scale_logit = nn.Parameter(torch.tensor(-0.5))
        self.last_structure_prior: torch.Tensor | None = None
        self.zero_structure_template: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        proj = self.proj(x)
        local = self.local_module(proj)
        _, structure_prior = self.structure_module(local)

        feat = self.sc_attention(local)
        if self.global_context is not None:
            feat = self.global_context(feat)

        fused = self.fuse(feat)
        res_scale = torch.sigmoid(self.res_scale_logit).to(device=fused.device, dtype=fused.dtype)
        out = identity + res_scale * fused

        self.last_structure_prior = structure_prior if self.use_structure_prior else None
        return out

# -----------------------------------------------------------------------------
# PRSF: Prior-guided Refinement after Fusion
# -----------------------------------------------------------------------------
class PRSFConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU() if act else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSConv(nn.Module):
    """Depthwise-separable conv: DW3x3 -> PW1x1."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.dw = PRSFConvBNAct(c, c, kernel_size=3, padding=1, groups=c, act=True)
        self.pw = PRSFConvBNAct(c, c, kernel_size=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class GroupMaskEnhancer(nn.Module):
    """
    Per-group prior-guided enhancement:
        1) region-gated modulation
        2) structure-gated modulation
        3) lightweight conv refinement (DSConv or MultiDilatedDSConv)
    """

    def __init__(
        self,
        channels_per_group: int,
        region_init: float | None = None,
        structure_init: float | None = None,
        use_multidilated_dsconv: bool = False,
        enable_prior_modulation: bool = True,
    ) -> None:
        super().__init__()
        cpg = int(channels_per_group)
        if cpg <= 0:
            raise ValueError(f"channels_per_group must be positive, got {channels_per_group}")
        default_region_init = -1.2
        # Region is primary and structure is auxiliary by design.
        self.region_scale_logit = nn.Parameter(torch.tensor(default_region_init if region_init is None else float(region_init)))
        self.structure_scale_logit = nn.Parameter(torch.tensor(-2.2 if structure_init is None else float(structure_init)))
        self.use_multidilated_dsconv = bool(use_multidilated_dsconv)
        self.enable_prior_modulation = bool(enable_prior_modulation)
        self.refine = (
            MultiDilatedDSConv(cpg, dilations=(1, 3, 5))
            if self.use_multidilated_dsconv
            else DSConv(cpg)
        )

    def forward(
        self,
        group_feat: torch.Tensor,
        region_mask: torch.Tensor,
        structure_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if structure_mask is None:
            structure_mask = torch.zeros_like(region_mask)

        region_scale = torch.sigmoid(self.region_scale_logit).to(
            device=group_feat.device,
            dtype=group_feat.dtype,
        )
        structure_scale = torch.sigmoid(self.structure_scale_logit).to(
            device=group_feat.device,
            dtype=group_feat.dtype,
        )

        x = group_feat
        if self.enable_prior_modulation:
            region_hint = region_scale * (2.0 * region_mask - 1.0)
            structure_hint = structure_scale * (2.0 * structure_mask - 1.0)
            x = x * (1.0 + region_hint)
            x = x * (1.0 + structure_hint)
        x = self.refine(x)
        return x


class PRSF(nn.Module):
    """
    PRSF: Prior-guided Refinement after Fusion module.

    Input:
        fused_feat, structure_prior(optional)

    Flow:
        fused_feat -> input_proj -> aux_mask_head -> region_prior
        fused_feat split into 4 groups
        each group applies region modulation -> structure modulation -> refine
        (DSConv or MultiDilatedDSConv)
        concat groups -> channel shuffle -> grouped_out
        out = grouped_out + fused0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        region_init: float | None = None,
        structure_init: float | None = None,
        use_multidilated_dsconv: bool = False,
        enable_prior_modulation: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.group_count = 4
        if self.out_channels % self.group_count != 0:
            raise ValueError(
                f"PRSF expects out_channels divisible by {self.group_count},"
                f"got out_channels={self.out_channels}"
            )
        self.channels_per_group = self.out_channels // self.group_count

        # Always project to the residual anchor space (Cout) for a unified PRSF path.
        self.input_proj = PRSFConvBNAct(self.in_channels, self.out_channels, kernel_size=1, act=False)
        self.aux_mask_head = nn.Conv2d(self.out_channels, 1, kernel_size=1, bias=True)
        self.last_region_prior: torch.Tensor | None = None
        self.last_structure_prior_map: torch.Tensor | None = None
        self.group_mask_enhancers = nn.ModuleList(
            [
                GroupMaskEnhancer(
                    self.channels_per_group,
                    region_init=region_init,
                    structure_init=structure_init,
                    use_multidilated_dsconv=use_multidilated_dsconv,
                    enable_prior_modulation=enable_prior_modulation,
                )
                for _ in range(self.group_count)
            ]
        )

    @staticmethod
    def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
        b, c, h, w = x.shape
        g = max(1, int(groups))
        if c % g != 0:
            return x
        cpg = c // g
        x = x.view(b, g, cpg, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x

    def _align_structure_prior(
        self,
        structure_prior: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Align structure prior to the current decoder feature shape.
        """
        if structure_prior is None:
            return torch.zeros(
                (reference.shape[0], 1, reference.shape[2], reference.shape[3]),
                device=reference.device,
                dtype=reference.dtype,
            )

        sp = structure_prior
        if sp.dim() == 3:
            sp = sp.unsqueeze(1)
        if sp.dim() != 4:
            raise ValueError(f"structure_prior must be 3D/4D tensor, got shape={tuple(sp.shape)}")
        if sp.shape[1] != 1:
            sp = sp.mean(dim=1, keepdim=True)
        if sp.shape[-2:] != reference.shape[-2:]:
            sp = F.interpolate(sp, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        return sp.to(device=reference.device, dtype=reference.dtype)

    def forward(
        self,
        x: torch.Tensor,
        structure_prior: torch.Tensor | None = None,
    ):
        fused0 = self.input_proj(x)

        aux_mask_logit = self.aux_mask_head(fused0)
        region_prior = torch.sigmoid(aux_mask_logit).to(dtype=fused0.dtype)

        structure_prior_map = self._align_structure_prior(structure_prior, fused0)
        self.last_region_prior = region_prior
        self.last_structure_prior_map = structure_prior_map

        b, _, h, w = fused0.shape
        fused_grouped = fused0.reshape(b, self.group_count, self.channels_per_group, h, w)
        group_feats = []
        for group_idx in range(self.group_count):
            group_feat = fused_grouped[:, group_idx, :, :, :]
            group_feat = self.group_mask_enhancers[group_idx](
                group_feat,
                region_prior,
                structure_prior_map,
            )
            group_feats.append(group_feat)
        grouped_out = torch.cat(group_feats, dim=1)
        grouped_out = self._channel_shuffle(grouped_out, groups=self.group_count)

        out = grouped_out + fused0
        return out, aux_mask_logit

# -----------------------------------------------------------------------------
# DRS
# -----------------------------------------------------------------------------
class PRSFDecoder(nn.Module):
    """Prior-guided PRSF decoder (up1-4 + boundary head + final conv).

    The forward consumes encoder features ``x1..x5`` and per-stage structure
    priors ``s1..s4`` and returns ``(logits, aux_logits, boundary_logit)`` to
    preserve the original DRS return contract.
    """

    def __init__(
        self,
        *,
        enc_channels: Sequence[int],
        bottleneck_ch: int,
        dec_channels: Sequence[int],
        out_channel: int,
        bilinear: bool = True,
        skip_fusion_mode: str = "add",
        prsf_use_multidilated_dsconv: bool = True,
        prsf_enable_prior_modulation: bool = True,
        prsf_enable_aux_mask_supervision: bool = True,
        use_boundary_head: bool = True,
    ) -> None:
        super().__init__()
        enc1_ch, enc2_ch, enc3_ch, enc4_ch = map(int, enc_channels)
        dec1_ch, dec2_ch, dec3_ch, dec4_ch = map(int, dec_channels)
        bottleneck_ch = int(bottleneck_ch)

        self.skip_fusion_mode = str(skip_fusion_mode)
        self.prsf_use_multidilated_dsconv = bool(prsf_use_multidilated_dsconv)
        self.prsf_enable_prior_modulation = bool(prsf_enable_prior_modulation)
        self.prsf_enable_aux_mask_supervision = bool(prsf_enable_aux_mask_supervision)
        self.use_boundary_head = bool(use_boundary_head)
        # Last decoder feature before the final projection; exposed for
        # region-reliability consumers (Stage 3). Reset each forward.
        self.last_decoder_feature: torch.Tensor | None = None
        # Existing up3 output retained as optional training context.
        self.last_decoder_mid_feature: torch.Tensor | None = None

        # PRSF handles the decoder refinement internally, so no original
        # post-upsample conv block is attached.
        upsample_block = PRSF

        # Decoder stage-specific prior init (explicitly controlled per up block).
        up1_kwargs = {
            "region_init": -0.6,
            "structure_init": -1.2,
            "use_multidilated_dsconv": self.prsf_use_multidilated_dsconv,
            "enable_prior_modulation": self.prsf_enable_prior_modulation,
        }
        up2_kwargs = {
            "region_init": -0.8,
            "structure_init": -1.4,
            "use_multidilated_dsconv": self.prsf_use_multidilated_dsconv,
            "enable_prior_modulation": self.prsf_enable_prior_modulation,
        }
        up3_kwargs = {
            "region_init": -1.0,
            "structure_init": -1.6,
            "use_multidilated_dsconv": self.prsf_use_multidilated_dsconv,
            "enable_prior_modulation": self.prsf_enable_prior_modulation,
        }
        up4_kwargs = {
            "region_init": -1.2,
            "structure_init": -1.9,
            "use_multidilated_dsconv": self.prsf_use_multidilated_dsconv,
            "enable_prior_modulation": self.prsf_enable_prior_modulation,
        }

        self.up1 = Up(
            bottleneck_ch + enc4_ch,
            dec1_ch,
            bilinear,
            upsample_block=upsample_block,
            upsample_block_kwargs=up1_kwargs,
            skip_fusion_mode=self.skip_fusion_mode,
            low_channels=bottleneck_ch,
            skip_channels=enc4_ch,
        )
        self.up2 = Up(
            dec1_ch + enc3_ch,
            dec2_ch,
            bilinear,
            upsample_block=upsample_block,
            upsample_block_kwargs=up2_kwargs,
            skip_fusion_mode=self.skip_fusion_mode,
            low_channels=dec1_ch,
            skip_channels=enc3_ch,
        )
        self.up3 = Up(
            dec2_ch + enc2_ch,
            dec3_ch,
            bilinear,
            upsample_block=upsample_block,
            upsample_block_kwargs=up3_kwargs,
            skip_fusion_mode=self.skip_fusion_mode,
            low_channels=dec2_ch,
            skip_channels=enc2_ch,
        )
        self.up4 = Up(
            dec3_ch + enc1_ch,
            dec4_ch,
            bilinear,
            upsample_block=upsample_block,
            upsample_block_kwargs=up4_kwargs,
            skip_fusion_mode=self.skip_fusion_mode,
            low_channels=dec3_ch,
            skip_channels=enc1_ch,
        )

        self.boundary_head = (
            nn.Conv2d(dec4_ch, 1, kernel_size=1, bias=True)
            if self.use_boundary_head
            else None
        )
        self.final_conv = OutConv(dec4_ch, int(out_channel))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        s1: torch.Tensor | None = None,
        s2: torch.Tensor | None = None,
        s3: torch.Tensor | None = None,
        s4: torch.Tensor | None = None,
    ):
        self.last_decoder_mid_feature = None
        self.last_decoder_feature = None
        x = self.up1(x5, x4, structure_prior=s4)
        aux1 = getattr(self.up1, "last_aux", None)
        x = self.up2(x, x3, structure_prior=s3)
        aux2 = getattr(self.up2, "last_aux", None)
        x = self.up3(x, x2, structure_prior=s2)
        aux3 = getattr(self.up3, "last_aux", None)
        self.last_decoder_mid_feature = x
        x = self.up4(x, x1, structure_prior=s1)
        aux4 = getattr(self.up4, "last_aux", None)
        self.last_decoder_feature = x
        boundary_logit = self.boundary_head(x) if self.boundary_head is not None else None
        # Keep aux order consistent with shallow -> deep outputs.
        aux_logits = [aux for aux in (aux4, aux3, aux2, aux1) if torch.is_tensor(aux)]
        if not self.prsf_enable_aux_mask_supervision:
            # Keep PRSF forward path unchanged, only disable aux supervision targets.
            aux_logits = []
        logits = self.final_conv(x)
        return logits, aux_logits, boundary_logit


class DRS(nn.Module):
    """主方法固定为一个 SLGE 编码器和两个独立 PRSF 解码器。"""

    def __init__(
        self,
        in_channel: int = 3,
        out_channel: int = 1,
        features: Sequence[int] = (64, 128, 256, 512, 1024),
        bilinear: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        features = list(features)
        if len(features) not in (4, 5):
            raise ValueError(f"Expected 4 or 5 feature widths, got {len(features)}: {features}")

        factor = 2 if bilinear else 1

        self.in_channel = int(in_channel)
        self.out_channel = int(out_channel)
        self.features = features
        self.bilinear = bool(bilinear)

        self.skip_fusion_mode = "add"
        self.prsf_use_multidilated_dsconv = True
        self.prsf_enable_prior_modulation = True
        self.prsf_enable_aux_mask_supervision = True
        self.use_boundary_head = True
        self.use_lmmb = True
        self.use_scra = True
        self.use_gcc = True

        enc1_ch, enc2_ch, enc3_ch, enc4_ch = map(int, features[:4])

        if len(features) == 5:
            bottleneck_ch = int(features[4])
            dec1_ch, dec2_ch, dec3_ch, dec4_ch = enc4_ch, enc3_ch, enc2_ch, enc1_ch
        else:
            bottleneck_ch = (enc4_ch * 2) // factor
            dec1_ch, dec2_ch, dec3_ch, dec4_ch = enc4_ch // factor, enc3_ch // factor, enc2_ch // factor, enc1_ch

        # Encoder stage split:
        # inc/down1 -> SLGE-Lite, down2/down3 -> SLGE-Full, bottleneck -> SLGE-Global.
        shallow_slge_block = lambda in_channels, out_channels: SLGE(
            in_channels,
            out_channels,
            low_init=-3.0,
            shape_init=-2.0,
            context_init=-2.5,
            stage_mode="lite",
            use_global=False,
            dilations=(1,),
            use_lmmb=self.use_lmmb,
            use_scra=self.use_scra,
            use_gcc=self.use_gcc,
        )
        deep_slge_block = lambda in_channels, out_channels: SLGE(
            in_channels,
            out_channels,
            low_init=-2.5,
            shape_init=-1.8,
            context_init=-1.8,
            stage_mode="full",
            use_global=False,
            dilations=(1, 3, 5),
            use_lmmb=self.use_lmmb,
            use_scra=self.use_scra,
            use_gcc=self.use_gcc,
        )
        # Bottleneck stage directly uses SLGE-Global.
        bottleneck_slge_block = lambda in_channels, out_channels: SLGE(
            in_channels,
            out_channels,
            stage_mode="global",
            use_global=True,
            dilations=(1, 4, 5),
            use_lmmb=self.use_lmmb,
            use_scra=self.use_scra,
            use_gcc=self.use_gcc,
        )
        inc_block = shallow_slge_block
        down1_block = shallow_slge_block
        down2_block = deep_slge_block
        down3_block = deep_slge_block
        bottleneck_block = bottleneck_slge_block

        self.inc = inc_block(self.in_channel, enc1_ch)
        self.down1 = Down(enc1_ch, enc2_ch, conv_block=down1_block)
        self.down2 = Down(enc2_ch, enc3_ch, conv_block=down2_block)
        self.down3 = Down(enc3_ch, enc4_ch, conv_block=down3_block)
        self.down4 = Down(enc4_ch, bottleneck_ch, conv_block=bottleneck_block)

        self.dropout_layer = nn.Dropout2d(p=float(dropout)) if float(dropout) > 0 else nn.Identity()

        # Aggregated structure prior cache (set in forward).
        self.last_structure_prior: torch.Tensor | None = None
        decoder_kwargs = dict(
            enc_channels=(enc1_ch, enc2_ch, enc3_ch, enc4_ch),
            bottleneck_ch=bottleneck_ch,
            dec_channels=(dec1_ch, dec2_ch, dec3_ch, dec4_ch),
            out_channel=self.out_channel,
            bilinear=bilinear,
            skip_fusion_mode=self.skip_fusion_mode,
            prsf_use_multidilated_dsconv=self.prsf_use_multidilated_dsconv,
            prsf_enable_prior_modulation=self.prsf_enable_prior_modulation,
            prsf_enable_aux_mask_supervision=self.prsf_enable_aux_mask_supervision,
            use_boundary_head=self.use_boundary_head,
        )
        self.decoder = PRSFDecoder(**decoder_kwargs)
        self.decoder_b = PRSFDecoder(**decoder_kwargs)

    @staticmethod
    def _extract_structure_prior(stage_module: nn.Module) -> torch.Tensor | None:
        """Fetch cached structure prior from a stage block if available."""
        direct = getattr(stage_module, "last_structure_prior", None)
        if direct is not None:
            return direct

        maxpool_conv = getattr(stage_module, "maxpool_conv", None)
        if maxpool_conv is not None and len(maxpool_conv) > 0:
            tail = maxpool_conv[-1]
            return getattr(tail, "last_structure_prior", None)
        return None

    @staticmethod
    def _unify_structure_prior(sp: torch.Tensor | None) -> torch.Tensor | None:
        if sp is None:
            return None
        if sp.dim() == 3:
            sp = sp.unsqueeze(1)
        if sp.dim() != 4:
            return None
        if sp.shape[1] != 1:
            sp = sp.mean(dim=1, keepdim=True)
        return sp

    @staticmethod
    def _stage_smooth_structure(sp: torch.Tensor | None, stage: int) -> torch.Tensor | None:
        if sp is None:
            return None
        if int(stage) <= 2:
            return sp
        kernel_size = 3 if int(stage) == 3 else 5
        padding = kernel_size // 2
        return F.avg_pool2d(sp, kernel_size=kernel_size, stride=1, padding=padding)

    def _with_training_context(
        self,
        output,
        decoder: PRSFDecoder,
        bottleneck_feature: torch.Tensor,
        feature_relative_change: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Attach tensors from this exact forward without changing defaults.

        The returned tensors remain connected to autograd. Training code owns
        every detach decision; ordinary inference never requests this context.
        """

        decoder_feature = decoder.last_decoder_feature
        decoder_mid_feature = decoder.last_decoder_mid_feature
        if decoder_feature is None:
            raise RuntimeError("Decoder feature was not produced by the current forward.")
        if decoder_mid_feature is None:
            raise RuntimeError("Intermediate decoder feature was not produced by the current forward.")
        if self.last_structure_prior is None:
            raise RuntimeError("SRM structure prior was not produced by the current forward.")
        context = {
            "output": output,
            "decoder_feature": decoder_feature,
            "decoder_mid_feature": decoder_mid_feature,
            "bottleneck_feature": bottleneck_feature,
            "structure_prior": self.last_structure_prior,
        }
        if feature_relative_change is not None:
            context["feature_relative_change"] = feature_relative_change
        return context

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_context: bool = False,
        bottleneck_perturbation: Callable[[torch.Tensor], Any] | None = None,
        decoder_branch: str | None = None,
    ):
        if decoder_branch not in (None, "a", "b"):
            raise ValueError("decoder_branch must be a, b, or None (mean prediction).")
        x1 = self.inc(x)
        s1 = self._stage_smooth_structure(self._unify_structure_prior(self._extract_structure_prior(self.inc)), 1)
        x2 = self.down1(x1)
        s2 = self._stage_smooth_structure(self._unify_structure_prior(self._extract_structure_prior(self.down1)), 2)
        x3 = self.down2(x2)
        s3 = self._stage_smooth_structure(self._unify_structure_prior(self._extract_structure_prior(self.down2)), 3)
        x4 = self.down3(x3)
        s4 = self._stage_smooth_structure(self._unify_structure_prior(self._extract_structure_prior(self.down3)), 4)
        x5 = self.down4(x4)
        x5 = self.dropout_layer(x5)
        clean_bottleneck_shape = x5.shape
        clean_bottleneck_device = x5.device
        clean_bottleneck_dtype = x5.dtype
        feature_relative_change = None
        # Training-only hook for DDV. The callback contains no parameters
        # and the perturbed tensor is decoded by this same decoder instance.
        # In eval mode it is deliberately ignored, even if passed by mistake.
        if self.training and bottleneck_perturbation is not None:
            perturbed = bottleneck_perturbation(x5)
            if isinstance(perturbed, tuple):
                if len(perturbed) != 2:
                    raise ValueError(
                        "bottleneck_perturbation tuple must be (feature, relative_change)."
                    )
                x5, feature_relative_change = perturbed
            else:
                x5 = perturbed
            if not torch.is_tensor(x5) or x5.ndim != 4:
                raise TypeError("bottleneck_perturbation must return a BCHW tensor.")
            if x5.shape != clean_bottleneck_shape:
                raise ValueError(
                    "bottleneck_perturbation must preserve BCHW shape: "
                    f"expected {tuple(clean_bottleneck_shape)}, got {tuple(x5.shape)}."
                )
            if x5.device != clean_bottleneck_device or x5.dtype != clean_bottleneck_dtype:
                raise ValueError(
                    "bottleneck_perturbation must preserve feature device and dtype."
                )
            if not bool(torch.isfinite(x5.detach()).all()):
                raise ValueError("bottleneck_perturbation produced NaN or Inf.")
        # Cache the shallowest (highest-resolution) structure prior for
        # Stage-3 reliability consumers. Harmless when None.
        self.last_structure_prior = s1

        decoder_inputs = dict(x5=x5, x4=x4, x3=x3, x2=x2, x1=x1, s1=s1, s2=s2, s3=s3, s4=s4)

        if decoder_branch is None:
            output_a = self.decoder(**decoder_inputs)
            output_b = self.decoder_b(**decoder_inputs)
            output = (
                0.5 * (output_a[0] + output_b[0]),
                [0.5 * (a + b) for a, b in zip(output_a[1], output_b[1])],
                0.5 * (output_a[2] + output_b[2]),
            )
            if return_context:
                return {
                    "output": output,
                    "branch_a": self._with_training_context(output_a, self.decoder, x5),
                    "branch_b": self._with_training_context(output_b, self.decoder_b, x5),
                }
            return output
        decoder = self.decoder_b if decoder_branch == "b" else self.decoder
        output = decoder(**decoder_inputs)
        if return_context:
            output = self._with_training_context(
                output,
                decoder,
                bottleneck_feature=x5,
                feature_relative_change=feature_relative_change,
            )
        return output


def _build_features(
    features: Sequence[int] | None = None,
    base_channels: int = 64,
) -> Sequence[int]:
    if features is not None:
        vals = list(features)
        if len(vals) == 4:
            return vals
        if len(vals) == 5:
            return vals
        raise ValueError(f"Expected 4 or 5 feature widths, got {len(vals)}: {vals}")
    base = int(base_channels)
    return [base, base * 2, base * 4, base * 8, base * 16]


def build_model(
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> DRS:
    """创建主方法 DRS；架构固定为共享编码器和两个解码器。"""

    cfg = dict(config or {})
    cfg.update(kwargs)


    model = DRS(
        in_channel=int(cfg.get("in_channel", cfg.get("in_channels", 3))),
        out_channel=int(cfg.get("out_channel", cfg.get("out_channels", cfg.get("num_classes", 1)))),
        features=_build_features(
            features=cfg.get("features"),
            base_channels=int(cfg.get("base_channels", 64)),
        ),
        bilinear=bool(cfg.get("bilinear", True)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    return model


