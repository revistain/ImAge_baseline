# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from backbone.dinov2.attention import Attention, MemEffAttention
from backbone.dinov2.drop_path import DropPath
from backbone.dinov2.layer_scale import LayerScale
from backbone.dinov2.mlp import Mlp

logger = logging.getLogger("dinov2")

try:
    from xformers.ops import fmha
    from xformers.ops import scaled_index_add, index_select_cat

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

class VanillaAdapter(nn.Module):
    def __init__(
        self,
        fc_in_channels: int,
        in_channels: int,
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect=skip_connect

        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> List[Tensor]:
        x0 = self.D_fc1(x)
        x0 = self.act(x0)
        outputs = self.D_fc2(x0)

        if self.skip_connect:
            outputs+=x
        return outputs


class FiLMAdapter(nn.Module):
    """Layer-index conditioned shared adapter (FiLM: Feature-wise Linear Modulation).
    단일 인스턴스를 모든 블록이 공유하며, layer_idx(timestep)로 동작을 구분한다.
    """
    def __init__(self, fc_in_channels: int, in_channels: int, num_layers: int = 12) -> None:
        super().__init__()
        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)
        self.act   = nn.GELU()
        # FiLM: 레이어별 scale(γ) & shift(β) — 항등 변환에서 시작
        self.film_gamma = nn.Embedding(num_layers, in_channels)
        self.film_beta  = nn.Embedding(num_layers, in_channels)
        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_beta.weight)

    def forward(self, x: Tensor, layer_idx: int) -> Tensor:
        t     = torch.tensor(layer_idx, dtype=torch.long, device=x.device)
        x0    = self.act(self.D_fc1(x))
        # γ, β: [hidden] → broadcast over [B, N, hidden]
        x0    = self.film_gamma(t) * x0 + self.film_beta(t)
        return self.D_fc2(x0)

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        use_adapter = True,
        use_film_adapter: bool = False,  # 공유 FiLMAdapter 사용 여부
        layer_idx: int = 0,              # 블록 위치 (timestep)
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )

        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path
        self.use_adapter       = use_adapter
        self.use_film_adapter  = use_film_adapter
        self.layer_idx         = layer_idx

        # 독립 adapter: film 모드가 아닐 때만 생성
        if self.use_adapter and not self.use_film_adapter:
            self.adapter = VanillaAdapter(dim, dim // 2)

        drop_path = 0.
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor, modality=None, return_attn=False,
                film_adapter=None, return_flow_info=False) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x))[0])

        def _adapter_out(x_normed: Tensor) -> Tensor:
            if self.use_film_adapter and film_adapter is not None:
                return film_adapter(x_normed, self.layer_idx)
            return self.adapter(x_normed)

        def ffn_residual_func(x: Tensor, return_adapter=False):
            if self.use_adapter and modality == True:
                x_normed = self.norm2(x)
                x_a = _adapter_out(x_normed)
                if return_adapter:
                    return self.ls2(self.mlp(x_normed) + 0.2 * x_a), x_a
                return self.ls2(self.mlp(x_normed) + 0.2 * x_a)
            # adapter 비활성 시에도 return_adapter=True면 (residual, None) 반환
            residual = self.ls2(self.mlp(self.norm2(x)))
            if return_adapter:
                return residual, None
            return residual

        need_adapter_out = return_attn or return_flow_info

        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(
                x, residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            return x, None
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path2(ffn_residual_func(x))
            return x, None
        else:
            x = x + attn_residual_func(x)
            if need_adapter_out:
                ffn_out, x_a = ffn_residual_func(x, return_adapter=True)
                x = x + ffn_out
                if return_attn:
                    return x, ffn_out   # 기존 호환
                return x, x_a           # return_flow_info: raw adapter output
            else:
                x = x + ffn_residual_func(x)
                return x, None

def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs

class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor], modality=None) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:
            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list, modality=None, return_attn=False,
                film_adapter=None, return_flow_info=False):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list, modality=modality, return_attn=return_attn,
                                   film_adapter=film_adapter, return_flow_info=return_flow_info)
        elif isinstance(x_or_x_list, list):
            assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list, modality=modality)
        else:
            raise AssertionError

