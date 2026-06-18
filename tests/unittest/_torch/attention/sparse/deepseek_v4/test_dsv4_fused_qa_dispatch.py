# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DeepSeek-V4 fused q_a *dispatch* routing decision.

These exercise only the pure decision function ``_select_dsv4_fused_qa_path`` (no fused op, no model),
so — unlike the op/numeric tests in ``test_dsv4_fused_qa.py`` — they are NOT gated on the SM100 family
and run on ANY GPU node (incl. non-SM100). They lock the three-way routing, including the
``1CTA_MIN_M > MIN_M`` case that previously mis-routed ``[MIN_M, 1CTA_MIN_M)`` to the non-fused path
instead of the 2-CTA kernel.

The routing policy itself needs no GPU, but importing it pulls in the ``tensorrt_llm`` package, which
currently requires a CUDA GPU at import time (a repo-wide constraint, not specific to this feature):
without one it raises ``RuntimeError("No CUDA GPUs are available")``, NOT ``ImportError``. So
``pytest.importorskip`` (which only catches ``ImportError``) would let that error fail collection;
the guard below catches both and skips the module cleanly instead.

Caveat: on a true no-GPU node, ``tests/unittest/conftest.py`` also imports ``tensorrt_llm`` and fails
the SAME way *before* this module is reached, so in practice these tests are GPU-only like the rest of
``tests/unittest`` (verified). The guard below still makes this module self-consistent. The real,
delivered win over the prior state is that the routing matrix is no longer behind the SM100-FAMILY skip,
so it runs on ANY GPU node, including non-SM100, where the SM100-gated op tests correctly skip.
"""

import pytest

try:
    from tensorrt_llm._torch.modules.attention import _select_dsv4_fused_qa_path
except (ImportError, RuntimeError) as exc:  # RuntimeError: no CUDA GPU at tensorrt_llm import
    pytest.skip(
        f"tensorrt_llm import requires a CUDA GPU ({type(exc).__name__}: {exc}); "
        "the routing logic runs on any GPU node",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "enabled,num_tokens,min_m,onecta_min_m,expected",
    [
        # feature off -> always non-fused, regardless of token count
        (False, 1000, 256, 1, "nonfused"),
        (False, 1, 256, 256, "nonfused"),
        # default (1CTA_MIN_M == MIN_M): large M -> 2-CTA, decode -> non-fused (single-CTA never)
        (True, 256, 256, 256, "2cta"),
        (True, 1000, 256, 256, "2cta"),
        (True, 255, 256, 256, "nonfused"),
        (True, 1, 256, 256, "nonfused"),
        # single-CTA enabled (1CTA_MIN_M < MIN_M): [1CTA_MIN_M, MIN_M) -> single-CTA
        (True, 1, 256, 1, "1cta"),
        (True, 32, 256, 1, "1cta"),
        (True, 255, 256, 1, "1cta"),
        (True, 256, 256, 1, "2cta"),
        (True, 64, 256, 64, "1cta"),
        (True, 63, 256, 64, "nonfused"),
        # 1CTA_MIN_M > MIN_M (documented "disable single-CTA"): the bug case.
        # [MIN_M, 1CTA_MIN_M) MUST stay 2-CTA, NOT fall back to non-fused; single-CTA never selected.
        (True, 300, 256, 512, "2cta"),
        (True, 256, 256, 512, "2cta"),
        (True, 511, 256, 512, "2cta"),
        (True, 600, 256, 512, "2cta"),
        (True, 200, 256, 512, "nonfused"),
        # 1CTA_MIN_M == 1 boundary: M=1 fuses (single-CTA), nothing below.
        (True, 1, 2, 1, "1cta"),
    ],
)
def test_select_dsv4_fused_qa_path(enabled, num_tokens, min_m, onecta_min_m, expected):
    assert _select_dsv4_fused_qa_path(enabled, num_tokens, min_m, onecta_min_m) == expected


def test_select_dsv4_fused_qa_path_single_cta_never_when_disabled():
    """With 1CTA_MIN_M >= MIN_M the single-CTA window is empty: no token count routes to '1cta'."""
    paths = {_select_dsv4_fused_qa_path(True, m, 256, 512) for m in range(0, 800)}
    assert "1cta" not in paths
    assert paths == {"2cta", "nonfused"}
