from __future__ import annotations
from typing_extensions import override
import mmap
import os
import numpy as np
import torch
from .qsa import CacheLayer_qsa
from ..ext import exllamav3_ext as ext
from ..model.model_tp_cuda import (
    cuda_host_register,
    cuda_host_unregister,
    CUDA_HOST_REGISTER_PORTABLE,
    CUDA_HOST_REGISTER_MAPPED,
)

# EXL3_QSA_KVO_RAW=0 keeps the indexer's raw key plane in VRAM (the phase-1 placement). It is
# offloaded by default: it is read only around the write head -- the pool kernel rebuilds just the
# blocks an append touches -- so it costs a few KiB per step over PCIe while freeing 3 KiB per
# token of VRAM, five times what the K/V planes leave behind
_OFFLOAD_RAW = os.environ.get("EXL3_QSA_KVO_RAW", "1") != "0"

_announced = False


class CacheLayer_qsa_offload(CacheLayer_qsa):
    """
    QSA cache layer with the K/V planes in pinned, device-mapped host memory and only the
    indexer's side planes in VRAM.

    The gather kernels reach K/V through a base pointer plus computed offsets and never touch a
    stride or a device property of those tensors, so handing them a zero-copy CUDA alias of host
    memory needs no kernel change: `sparse_attend`, `get_kv` and `ext.paged_kv_cache_update` all
    keep working, and `build_bc_attn`'s `layer.k.device == module.device` check passes because
    the alias really is a cuda:N tensor.

    What stays in VRAM is `pooled`, and that is not an optimisation but a requirement: the indexer
    scores EVERY block of the pooled plane on every step, so that plane's traffic is linear in
    context length (732 MiB per token at 1M across the full-attention layers). Everything else is
    read in bounded amounts -- K/V through a top-k selection whose size is fixed by the indexer
    budget, `raw_k` only around the write head, where the pool kernel rebuilds the blocks an append
    touched -- so their traffic is constant in context length. That is the whole reason this trade
    works, and why `pooled` is the one plane it cannot include.

    The slab is anonymous mmap memory registered with cudaHostRegister(PORTABLE | MAPPED) while
    the layer's device is current, not a `pin_memory = True` tensor. Both give pinned, mapped
    host memory, but torch resolves a device pointer's owning device through
    cudaPointerGetAttributes, which reports whichever device was current when the region was
    pinned -- and torch's caching host allocator recycles freed blocks across devices, so a slab
    freed on one device (autosplit rolls a layer back and retries on the next device) comes back
    bound to the wrong one and the alias is rejected. Registering the mapping ourselves ties each
    slab to its layer's device deterministically. mmap also guarantees the page alignment
    cudaHostRegister wants, and hands back zeroed pages without a 2 GiB memset.
    """

    def __init__(
        self,
        config,
        attention,
        cache_id: int,
        max_num_tokens: int,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)
        # The pinned host allocation backing self.k / self.v. The device aliases do NOT own it,
        # so these references are what keep them valid
        self.host_map = None      # mmap object (owns the pages)
        self.host_slab = None     # CPU tensor over it (what pinned_cuda_view aliases)
        self.host_ptr = 0         # registered base address, for unregister
        self.offload_raw = _OFFLOAD_RAW
        # CPU-side views of the same storage the device aliases point at, for copy_page
        self.host_k = None
        self.host_v = None
        self.host_raw_k = None

    @override
    def alloc(self, device: torch.device):
        global _announced

        dev = torch.device(device)
        assert dev.type == "cuda", \
            "QSA KV offload requires a CUDA device (the K/V aliases are device pointers)."
        self.device = device

        n = int(np.prod(self.shape)) if self.shape else 0
        nr = int(np.prod(self.raw_k_shape)) if self.offload_raw else 0
        total = 2 * n + nr

        if total:
            nbytes = total * torch.half.itemsize
            if not _announced:
                _announced = True
                print(f" -- QSA KV offload: pinning {nbytes / 1024 ** 3:.2f} GiB of host memory "
                      f"per attention layer; the first touch of each is slow (page-table setup).")
            # One slab for every offloaded plane: pinning is per-allocation work, and they are all
            # allocated and freed together
            idx = dev.index if dev.index is not None else 0
            self.host_map = mmap.mmap(-1, nbytes)
            self.host_slab = torch.frombuffer(self.host_map, dtype = torch.half, count = total)
            self.host_ptr = self.host_slab.data_ptr()
            with torch.cuda.device(idx):
                cuda_host_register(self.host_ptr, nbytes,
                                   CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED)
            alias = ext.pinned_cuda_view(self.host_slab, idx)
        else:
            alias = None

        if self.shape is None:
            self.k = None
            self.v = None
        else:
            self.k = alias[:n].view(self.shape)
            self.v = alias[n : 2 * n].view(self.shape)
            self.host_k = self.host_slab[:n].view(self.shape)
            self.host_v = self.host_slab[n : 2 * n].view(self.shape)

        if self.offload_raw:
            self.raw_k = alias[2 * n :].view(self.raw_k_shape)
            self.host_raw_k = self.host_slab[2 * n :].view(self.raw_k_shape)
        else:
            self.raw_k = torch.zeros(self.raw_k_shape, dtype = torch.half, device = device)
        self.pooled = torch.zeros(self.pooled_shape, dtype = torch.half, device = device)

    @override
    def free(self):
        # Order matters: the aliases are non-owning views of the slab's storage, so every
        # reference to them has to go before the slab does
        self.k = None
        self.v = None
        self.raw_k = None
        self.pooled = None
        self.host_k = None
        self.host_v = None
        self.host_raw_k = None
        self.device = None
        if self.host_ptr:
            cuda_host_unregister(self.host_ptr)
            self.host_ptr = 0
        self.host_slab = None
        if self.host_map is not None:
            try:
                self.host_map.close()
            except BufferError:
                # Something still holds a buffer view of the mapping; dropping the reference
                # leaves the unmap to the collector, which is fine now that it is unregistered
                pass
            self.host_map = None

    @override
    def copy_page(self, source: "CacheLayer_qsa_offload", from_page: int, to_page: int,
                  num_tokens: int):
        # Host-resident planes are copied on the CPU rather than by a device kernel reading and
        # writing the same memory across PCIe. It measures ~20% faster, but the reason to do it is
        # that it leaves the PCIe link alone: that link is what the offload spends on the
        # attention gather, and a page copy has no business competing for it.
        #
        # The device stream has to be drained first. A page being read may still be waiting on a
        # write from the current forward, and a page being written may still be under a read, and
        # a host memcpy is outside the stream's ordering entirely
        assert self.shape == source.shape
        torch.cuda.current_stream(torch.device(self.device)).synchronize()
        nt = num_tokens
        if self.shape is not None:
            self.host_k[to_page, :nt].copy_(source.host_k[from_page, :nt])
            self.host_v[to_page, :nt].copy_(source.host_v[from_page, :nt])
        if self.offload_raw:
            self.host_raw_k[to_page, :nt].copy_(source.host_raw_k[from_page, :nt])
        else:
            self.raw_k[to_page, :nt].copy_(source.raw_k[from_page, :nt], non_blocking = True)
        nb = (num_tokens + self.compress_ratio - 1) // self.compress_ratio
        self.pooled[to_page, :nb].copy_(source.pooled[from_page, :nb], non_blocking = True)

    @override
    def storage_size(self):
        # VRAM only. Autosplit and the TP planner size device allocations from this, and counting
        # the host-resident planes here would have them refuse a context that fits perfectly well
        n = np.prod(self.pooled_shape)
        if not self.offload_raw:
            n += np.prod(self.raw_k_shape)
        return n * torch.half.itemsize

    def host_size(self):
        """Bytes of pinned host memory this layer holds. Not part of the CacheLayer interface;
        used for reporting, since storage_size() deliberately hides it."""
        n = 2 * np.prod(self.shape) if self.shape else 0
        if self.offload_raw:
            n += np.prod(self.raw_k_shape)
        return n * torch.half.itemsize

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_qsa_offload,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens
            }
        }
