# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa
from xformers.ops import AttentionBias, memory_efficient_attention

from gatr.primitives.invariants import _load_inner_product_factors

# Masked out attention logits are set to this constant (a finite replacement for -inf):
_MASKED_OUT = float("-inf")

# Force the use of xformers attention, even when no xformers attention mask is provided:
FORCE_XFORMERS = False


def sdp_attention(
    q_mv: Tensor,
    k_mv: Tensor,
    v_mv: Tensor,
    q_s: Tensor,
    k_s: Tensor,
    v_s: Tensor,
    attn_mask : Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Equivariant geometric attention based on scaled dot products.

    Expects both multivector and scalar queries, keys, and values as inputs.
    Then this function computes multivector and scalar outputs in the following way:

    ```
    attn_weights[..., i, j] = softmax_j[
        pga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
        + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
    ]
    out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
    out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
    ```

    Parameters
    ----------
    q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
        Queries, multivector part.
    k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
        Keys, multivector part.
    v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
        Values, multivector part.
    q_s : Tensor with shape (..., num_items_out, num_s_channels_in)
        Queries, scalar part.
    k_s : Tensor with shape (..., num_items_in, num_s_channels_in)
        Keys, scalar part.
    v_s : Tensor with shape (..., num_items_in, num_s_channels_out)
        Values, scalar part.
    attn_mask : None or Tensor with shape (..., num_items, num_items)
        Optional attention mask

    Returns
    -------
    outputs_mv : Tensor with shape (..., num_items_out, num_mv_channels_out, 16)
        Result, multivector part
    outputs_s : Tensor with shape (..., num_items_out, num_s_channels_out)
        Result, scalar part
    """

    # Construct queries and keys by concatenating relevant MV components and aux scalars
    q = torch.cat([rearrange(q_mv * _load_inner_product_factors(), "... c x -> ... (c x)"), q_s], -1)
    k = torch.cat([rearrange(k_mv, "... c x -> ... (c x)"), k_s], -1)

    num_channels_out = v_mv.shape[-2]
    v = torch.cat([rearrange(v_mv, "... c x -> ... (c x)"), v_s], -1)

    v_out = scaled_dot_product_attention(q, k, v)

    v_out_mv = rearrange(v_out[..., : num_channels_out * 16], "... (c x) -> ...  c x", x=16)
    v_out_s = v_out[..., num_channels_out * 16 :]

    return v_out_mv, v_out_s


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Union[AttentionBias, Tensor]] = None,
) -> Tensor:
    """Execute (vanilla) scaled dot-product attention.

    Dynamically dispatch to xFormers if attn_mask is an instance of xformers.ops.AttentionBias
    or FORCE_XFORMERS is set, use torch otherwise.

    Parameters
    ----------
    query : Tensor
        of shape [batch, head, item, d]
    key : Tensor
        of shape [batch, head, item, d]
    value : Tensor
        of shape [batch, head, item, d]
    attn_mask : Optional[Union[AttentionBias, Tensor]]
        Attention mask

    Returns
    -------
    Tensor
        of shape [batch, head, item, d]
    """
    if FORCE_XFORMERS or isinstance(attn_mask, AttentionBias):
        query = query.transpose(1, 2)  # [batch, head, item, d] -> [batch, item, head, d]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = memory_efficient_attention(
            query.contiguous(), key.contiguous(), value, attn_bias=attn_mask
        )
        out = out.transpose(1, 2)  # [batch, item, head, d] -> [batch, head, item, d]
        return out
    return torch_sdpa(query, key, value, attn_mask=attn_mask)
