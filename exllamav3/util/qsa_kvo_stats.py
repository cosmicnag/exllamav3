from __future__ import annotations
import atexit
import os
import torch

# Phase-0 instrumentation for the QSA KV host-offload path (EXL3_QSA_KVO_STATS=1).
#
# Every judgement in the offload plan rests on one claim: a QSA row reads a constant amount of
# K/V (4 * block_topk + tail, padded to the kernel's K_pad) no matter how long the context is.
# This module is what makes that claim checkable instead of assumed. It records, per attention
# layer, how many sparse rows were served, how many cache positions those rows actually selected
# (counted on-device, so no synchronisation is added to the hot path), the resulting host->device
# read volume, and how much was moved by explicit prefill staging (phase 2).
#
# Counting is deliberately split into an exact term and a bound:
#   - the eager/prefill path builds its index list in python, so the valid (non -1) count is
#     accumulated exactly into a device-side counter;
#   - the graph-captured decode path builds indices inside the capture, where python cannot see
#     them, so only the padded K_pad bound is recorded. The bound is the transfer figure the plan
#     budgets against anyway, so both numbers are reported side by side.

_ENABLED = os.environ.get("EXL3_QSA_KVO_STATS", "0") != "0"


def enabled() -> bool:
    return _ENABLED


class _LayerStats:
    __slots__ = ("sparse_calls", "rows", "k_pad_max", "bound_bytes", "counter",
                 "decode_calls", "decode_rows", "stage_calls", "stage_bytes")

    def __init__(self):
        self.sparse_calls = 0
        self.rows = 0
        self.k_pad_max = 0
        self.bound_bytes = 0
        self.counter = None      # device tensor, exact count of selected cache positions
        self.decode_calls = 0
        self.decode_rows = 0
        self.stage_calls = 0
        self.stage_bytes = 0


_layers: dict[int, _LayerStats] = {}
_reported = False


def _layer(layer_idx: int) -> _LayerStats:
    st = _layers.get(layer_idx)
    if st is None:
        st = _layers[layer_idx] = _LayerStats()
    return st


def record_sparse(layer_idx: int, rows: int, k_pad: int, kv_heads: int, head_dim: int,
                  indices: torch.Tensor | None = None, decode: bool = False):
    """One sparse-attention launch over `rows` query rows, each reading up to `k_pad` cache
    positions of K and V. `indices` (the (rows, k_pad) int32 selection, -1 padded) is counted
    exactly when the caller has it in hand."""
    if not _ENABLED:
        return
    st = _layer(layer_idx)
    st.sparse_calls += 1
    st.rows += rows
    st.k_pad_max = max(st.k_pad_max, k_pad)
    # K and V, fp16, one head_dim vector per kv head per selected position
    st.bound_bytes += rows * k_pad * kv_heads * head_dim * 2 * 2
    if decode:
        st.decode_calls += 1
        st.decode_rows += rows
    if indices is not None:
        n = (indices >= 0).sum()
        if st.counter is None:
            st.counter = torch.zeros((), dtype = torch.long, device = indices.device)
        st.counter += n


def record_stage(layer_idx: int, nbytes: int):
    """One explicit host->VRAM staging copy (phase 2 prefill arena)."""
    if not _ENABLED:
        return
    st = _layer(layer_idx)
    st.stage_calls += 1
    st.stage_bytes += nbytes


def report():
    global _reported
    if not _ENABLED or _reported or not _layers:
        return
    _reported = True
    try:
        _report()
    except Exception as e:
        # atexit runs while CUDA may already be tearing down; a stats dump is never worth
        # turning a clean exit into a traceback
        print(f"QSA KV offload stats unavailable: {type(e).__name__}: {e}")


def _report():

    MIB = 1024 ** 2
    rowspec = "{:>5}  {:>9}  {:>11}  {:>7}  {:>13}  {:>13}  {:>8}  {:>11}"
    print()
    print("QSA KV offload stats (EXL3_QSA_KVO_STATS)")
    print(rowspec.format("layer", "sp.calls", "sparse rows", "K_pad",
                         "sel/row (act)", "host MiB (bd)", "stages", "staged MiB"))
    print("-" * 96)

    tot = _LayerStats()
    tot_sel = 0
    any_exact = False
    for layer_idx in sorted(_layers):
        st = _layers[layer_idx]
        sel = int(st.counter.item()) if st.counter is not None else None
        per_row = f"{sel / st.rows:.1f}" if sel is not None and st.rows else "-"
        if sel is not None:
            any_exact = True
            tot_sel += sel
        print(rowspec.format(
            layer_idx, st.sparse_calls, st.rows, st.k_pad_max, per_row,
            f"{st.bound_bytes / MIB:.1f}", st.stage_calls, f"{st.stage_bytes / MIB:.1f}"))
        tot.sparse_calls += st.sparse_calls
        tot.rows += st.rows
        tot.k_pad_max = max(tot.k_pad_max, st.k_pad_max)
        tot.bound_bytes += st.bound_bytes
        tot.decode_calls += st.decode_calls
        tot.decode_rows += st.decode_rows
        tot.stage_calls += st.stage_calls
        tot.stage_bytes += st.stage_bytes
    print("-" * 96)
    print(rowspec.format(
        "all", tot.sparse_calls, tot.rows, tot.k_pad_max,
        f"{tot_sel / tot.rows:.1f}" if any_exact and tot.rows else "-",
        f"{tot.bound_bytes / MIB:.1f}", tot.stage_calls, f"{tot.stage_bytes / MIB:.1f}"))

    if tot.decode_rows:
        print(f"\ndecode: {tot.decode_calls} sparse launches over {tot.decode_rows} rows")
    print(f"total host read bound: {tot.bound_bytes / 1024 ** 3:.2f} GiB"
          f"{'' if not any_exact else f'  (exact selections: {tot_sel})'}")
    print()


atexit.register(report)
