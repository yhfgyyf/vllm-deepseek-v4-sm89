# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native V4.1 operations, isolated from the original DSV4 cache contract."""

from .cache_utils import compute_global_topk_indices_and_lens
from .fused_indexer_q import fused_indexer_q_rope_quant
from .fused_q_rope_swa import fused_q_rope_swa_insert
from .indexer_k_store import indexer_k_norm_rope_store
from .inverse_rope import fused_inv_rope_fp8_quant
from .quant_utils import MXFP4_BLOCK_SIZE

__all__ = [
    "MXFP4_BLOCK_SIZE",
    "compute_global_topk_indices_and_lens",
    "fused_indexer_q_rope_quant",
    "fused_inv_rope_fp8_quant",
    "fused_q_rope_swa_insert",
    "indexer_k_norm_rope_store",
]
