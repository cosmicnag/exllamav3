
<p align="center">
  <img src="doc/logo.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

[Installation](#installation) · [Supported models](#architecture-support) · [Examples](#examples) · [Quantization](#exl3-quantization) · [Community](#community)

---

## Fork: QSA KV offload · branch `deploy/exllamav3-kvo`

> [!IMPORTANT]
> This is a fork of [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), kept for serving **Qwen3.8-Flash-Next (EXL3)** at very long context. Everything below this section is upstream's README and applies unchanged — the only fork-specific content is this section.

**Branch** [`deploy/exllamav3-kvo`](https://github.com/cosmicnag/exllamav3/tree/deploy/exllamav3-kvo) · **Base** upstream `dev` @ `e3b52f47` · **Delta** 2 commits, **Python only**

### What it adds

QSA attention normally keeps three planes in VRAM per cache token. This fork moves the two that are read in *bounded* amounts into pinned, device-mapped host memory, read zero-copy over PCIe:

| plane | placement | VRAM |
|---|---|---|
| `k`, `v` (gathered) | pinned host RAM | — |
| indexer `raw_k` | pinned host RAM (`EXL3_QSA_KVO_RAW=0` keeps it in VRAM) | — |
| indexer `pooled` | **VRAM — not offloaded** | 0.75 KiB/token |
| | | **0.75 vs 27.75 KiB/token** |

The trade only works because of *which* planes move. `pooled` is scored in full on every step, so its traffic is linear in context length and it has to stay on the device — at 1M context that plane alone is ~732 MiB of traffic per step. K/V are read through a top-k selection whose size is fixed by the indexer budget, and `raw_k` only around the write head, where the pool kernel rebuilds just the blocks an append touches. Both are constant in context length, so the PCIe cost is constant while the VRAM saving is linear — which is what buys the very long contexts.

The gather kernels reach K/V through a base pointer plus computed offsets and never read a stride or device property of those tensors, so handing them a zero-copy CUDA alias of host memory needs no kernel change: `sparse_attend`, `get_kv` and `ext.paged_kv_cache_update` all keep working.

### Requirements

- **Linux + CUDA.** The slab is anonymous `mmap` memory registered with `cudaHostRegister(PORTABLE | MAPPED)` while the layer's device is current — not a `pin_memory=True` tensor.
- A **QSA-attention model**, i.e. Qwen3.8-Flash-Next EXL3 (`qwen4_exp`).
- **fp16 cache layer.** Not compatible with a quantized cache, and mutually exclusive with `--cpu_cache_size` (K/V already lives in host memory).
- **Host RAM** for the slab, allocated eagerly from full cache capacity rather than per used page: ~6.4 GiB at 262K across the model's 12 QSA layers.

### Getting running

```sh
git clone -b deploy/exllamav3-kvo https://github.com/cosmicnag/exllamav3.git
cd exllamav3
# install a CUDA-enabled torch first (see Installation below), then:
pip install -e .
```

Then set the switch **before the model config is built**, i.e. in the environment of the process that loads the model:

```sh
export EXL3_QSA_KV_OFFLOAD=1
```

TabbyAPI users can pass `-kvo` as a model arg instead. **Omitting it silently loads with no offload** — the default is `0`.

#### Knobs

| env | default | what it does |
|---|---|---|
| `EXL3_QSA_KV_OFFLOAD` | `0` | master switch; the `-kvo` CLI flag is equivalent |
| `EXL3_QSA_KVO_RAW` | `1` | also offload the indexer's raw key plane; `0` keeps it in VRAM (the phase-1 placement) |
| `EXL3_QSA_KVO_ARENA` | 32 MiB | per-device staging arena for the history-page walk. A smaller arena only means more buckets — the PCIe volume is identical either way — but the kernel re-scans the selection per pass. Raise it when there is VRAM headroom to spare: the 32 MiB default stages ~32 of 1024 pages per pass and costs ~17% prefill at 200K (3,238 T/s); 512 MiB amortizes the walk and recovers it (~3,905 T/s) |
| `EXL3_QSA_KVO_STAGE` | `1` | `0` forces the direct per-row gather straight from host memory (for A/B) |
| `EXL3_QSA_KVO_STAGE_ROWS` | 1024 | query rows per kernel launch. The launch's partial buffers are linear in this and are charged against exactly the VRAM the offload exists to free, so a long prefill chunk is served a block at a time |
| `EXL3_QSA_KVO_STATS` | `0` | `1` prints a per-layer table at shutdown: sparse rows served, positions selected, host→device MiB, staged MiB |

**Verify the claim it rests on:** run with `EXL3_QSA_KVO_STATS=1` and watch `sel/row` — it should stay flat as context grows. That column is the constant-read assumption; if it moves, the design's premise is wrong for your model.

### Caveats

- **Not upstream-supported.** This branch is a rebase surface — expect it to be rebased onto upstream `dev` rather than merged. Upstream publishes the patch as a file only; there is no PR channel.
- Offloading reduces QSA VRAM cost, it does not remove it: `pooled` stays on the device and is the floor.
- The host slab is committed up front from full cache capacity, so the RAM cost is paid whether or not the context is used.
- For two concurrent 262K sessions the slab is ~13.5 GiB (capacity is a shared pool across sequences), not ~6.4 GiB.

### For agents

- Branch `deploy/exllamav3-kvo` = upstream `dev` @ `e3b52f47` + `db8fefec` (the port) + `0ceb06ef` (`page_size` lookup fix in `QSAIndexer.sparse_attend` for quant caches).
- Touched files: `exllamav3/cache/qsa_offload.py` (new), `exllamav3/util/qsa_kvo_stats.py` (new), `exllamav3/modules/attention_fn/qsa_triton.py`, `exllamav3/modules/qsa_indexer.py`, `exllamav3/modules/attn.py`, `exllamav3/model/config.py`, `exllamav3/model_init.py`, `exllamav3/generator/cpu_cache.py`.
- **The 2 commits are pure Python.** If your build is a *non-editable* install of base `e3b52f47`, copying those 8 files into site-packages *is* the deploy — no CUDA extension rebuild. Confirm with `git diff --name-only e3b52f47..deploy/exllamav3-kvo | grep -v '\.py$'` (empty) and by checking that no `ext.*` symbol appears only on the `+` side of the Python diff.
- `direct_url.json` in site-packages may name a different source tree than the one actually installed; diff the tree instead of trusting it.
- Failure modes: `--kv_offload` + `--cpu_cache_size` raises at startup; a quantized cache raises in `attn.py`; without `EXL3_QSA_KV_OFFLOAD=1` you get a silent fp16 load that OOMs at any cache size the offload was needed for.

---

ExLlamaV3 is an inference library for running local LLMs on modern consumer GPUs, with flexible quantization and parallel inference.

- **Quantization** - [EXL3](doc/exl3.md), based on QTIP, plus 2–8 bit cache quantization.
- **Parallel inference** - Flexible tensor-parallel and expert-parallel inference for consumer hardware setups.
- **CPU offloading** - Allows large MoE models to run with limited GPU resources. AVX2 and AVX512 support.  
- **Generation** - Continuous, dynamic batching, speculative decoding, multimodal support.
- **Integrations** - Broad [HF model support](#architecture-support), a [Transformers plugin](examples/transformers_integration.py), and an OpenAI-compatible API via [TabbyAPI](https://github.com/theroyallab/tabbyAPI/).

> [!TIP]
> **Looking for a server?** [TabbyAPI](https://github.com/theroyallab/tabbyAPI/) is the official and recommended backend server. It provides an OpenAI-compatible API for local or remote inference, HF model downloading, embedding model support, and HF Jinja2 chat templates. Its startup script manages and installs prerequisites to help you get started.

<p align="center">
  <img src="doc/qb_kld.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

## Installation

Start by making sure you have the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/) installed (CUDA 12.4 or later) since the Torch dependency is not automatically handled by `pip`. Then pick a method below:

### Prebuilt wheel · recommended

Pick a wheel from the [releases page](https://github.com/turboderp-org/exllamav3/releases), then e.g.:

```sh
pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.6/exllamav3-0.0.6+cu128.torch2.8.0-cp313-cp313-linux_x86_64.whl
```

### Install from PyPI

```sh
pip install exllamav3
```
Note that the PyPI package does not contain a prebuilt extension and requires the CUDA toolkit and build prerequisites (i.e. VS Build Tools on Windows, gcc on Linux, `python-dev` headers etc.).

### Build from source

<details>
<summary>Source installation with uv or pip</summary>


`exllamav3` declares a minimum `torch` version (>= 2.6.0) and CUDA version (>= 12.4), but beyond that the user is free to select a version of `torch` that is compatible with their environment.

`torch` can be installed in three ways (from least to most effort):
1. **with `uv`, setting only `--extra cuXXX`** installs `torch` automatically with the specified CUDA version, `torch` version is selected by `uv` from compatible versions in the specific index associated with the chosen CUDA version (options 1 and 2)
2. **with `uv`, creating a thin project that depends on `exllamav3[cuXXX]` and pins a specific `torch` version** — like (1) but `torch` is pinned in the thin project's `pyproject.toml`, see [pinning a specific PyTorch version (optional)](#pinning-a-specific-pytorch-version-optional) for details
3. Manually with `uv pip` or `pip` (options 3 and 4)

The flavor extras (`--extra`) are `cu124`, `cu126`, `cu128`, `cu129`, `cu130`, and `cu132` — pick the one matching your installed CUDA build. Both `uv sync` and `pip install .` build the package in an isolated environment where your `torch` is not visible, so they install the extension sources and compile them at first import (JIT, a few minutes once per torch version). For a precompiled install run `pip install --no-build-isolation .` in an environment that already has `torch`, or use the release wheels. Selecting a flavor installs the matching CUDA build of `torch`.

**Option 1 — Working in the cloned repo directly (`uv sync`):**

```sh
git clone https://github.com/turboderp-org/exllamav3
cd exllamav3
# (Optional) switch to dev branch for latest in-progress features
git checkout dev

uv venv
uv sync --extra cu130
# add --extra examples and/or --extra eval for those extra dependencies
```

**Option 2 — Using `exllamav3` as a dependency from another project (`uv add`):**

```sh
# `uv add` works inside an existing project (a directory with a pyproject.toml).
# `uv init` creates one if you're starting a new project, if integrating into
# an existing project skip `uv init`.
uv init my-project
cd my-project

# local checkout
uv add 'path/to/exllamav3[cu130]'               # non-editable
uv add 'path/to/exllamav3[cu130]' --editable    # editable

# straight from GitHub
uv add 'git+https://github.com/turboderp-org/exllamav3.git[cu130]'                 # default branch
uv add 'git+https://github.com/turboderp-org/exllamav3.git[cu130]' --branch dev    # specific branch
```

**Option 3 — Bring your own `torch` and let `uv` pick the backend automatically:**

```sh
uv venv            # or: uv venv --python-preference only-managed
source .venv/bin/activate
uv pip install torch --torch-backend=auto
uv pip install .
```

`--torch-backend=auto` inspects your system and installs the matching PyTorch CUDA build; see [Automatic backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection).

**Option 4 — With `pip`:**

On Windows, you also need the `triton-windows` package (declared as a dependency in `pyproject.toml`); the attention, cache and recurrent kernels are Triton and ExLlamaV3 does not import without it.

```sh
# install a CUDA-enabled torch first so it matches your setup, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install .
```

</details>

<details>
<summary>Pinning a specific PyTorch version (optional)</summary>

#### Pinning a specific PyTorch version (optional)

The flavor extra picks the *index*, but by default torch resolves to the latest version on that
index that satisfies `>=2.6.0`. To pin a specific torch version while developing on `exllamav3`,
create a **"thin" project** that consumes your local checkout as an editable install and declares
the exact `torch` version itself. This keeps the pin out of the `exllamav3` pyproject, so
you can change the torch version freely without touching the repo.

```
my-exllamav3-dev/          # thin project (uv init)
├── pyproject.toml
└── src/                  # package sources (auto-generated)
```

In `pyproject.toml`:

```toml
[project]
name = "my-exllamav3-dev"
version = "0.1.0"
description = "Dev environment for exllamav3"
requires-python = ">=3.10.11"
dependencies = [
    "exllamav3[cu130]",   # select correct CUDA version
    "torch==2.13.0",      # pin the exact torch version you need
]

[tool.uv.sources]
exllamav3 = { path = "../exllamav3", editable = true }
```

Adjust `../exllamav3` to point at your local checkout, then a plain `uv sync` sets up an
environment with the correct PyTorch index (routed via the `cuXXX` extra),
the pinned version of `torch` from that index (as long as it exists), and an editable install of `exllamav3` so code
changes apply immediately. Switch CUDA flavors by changing the extra (`exllamav3[cu124]`,
`exllamav3[cu128]`, …) and/or the torch pin in the thin project.

Or, if you're installing torch manually with `uv pip install torch` (e.g. as in Option 3 above),
specify the version directly, e.g. `uv pip install "torch==2.11.0" --torch-backend=auto`.

</details>

After installing with one of the options above, you should be able to run the conversion, eval and 
example scripts from the main repo directory, e.g., `uv run python convert.py -i ...` or, for manual
installations once the venv is active, `python convert.py -i ...`

**Build environment variables**

- `MAX_JOBS`: by default ninja may launch too many processes and run out of system memory for 
compilation. Set this to a reasonable value like 4 in that case.
- `EXLLAMA_NOCOMPILE`: set to install the library without compiling the C++/CUDA extension. Torch
will build/load it at runtime instead.

## Examples

A number of example scripts are provided to showcase the features of the backend and generator. 
For instance, a versatile CLI chatbot:

<p align="center">
  <img src="doc/chatpy.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

```sh
python examples/chat.py -m <input_dir> -mode <prompt_mode>

# Wealth of options
python examples/chat.py -h
```

## Architecture support

| Model family                                     | HF architecture | Multimodal | Notes |
|--------------------------------------------------| --- | :---: | --- |
| **AFM**                                          | `ArceeForCausalLM` |  |  |
| **AfMoE**                                        | `AfmoeForCausalLM` |  |  |
| **Apertus**                                      | `ApertursForCausalLM` |  |  |
| **Command-R** etc.                               | `CohereForCausalLM` |  |  |
| **Command-A**, **Command-R+** etc.               | `Cohere2ForCausalLM` |  |  |
| **DeciLM**, **Nemotron**                         | `DeciLMForCausalLM` |  |  |
| **Deepseek V3**                                  | `DeepseekV3ForCausalLM` |  |  |
| **Deepseek V4**                                  | `DeepseekV4ForCausalLM` | ✓ |  |
| **dots.llm1**                                    | `Dots1ForCausalLM` |  | |
| **ERNIE 4.5**                                    | `Ernie4_5_ForCausalLM`<br>`Ernie4_5_MoeForCausalLM` |  |  |
| **EXAONE 4.0**                                   | `Exaone4ForCausalLM` |  |  |
| **Gemma 2**                                      | `Gemma2ForCausalLM` |  |  |
| **Gemma 3**                                      | `Gemma3ForCausalLM`<br>`Gemma3ForConditionalGeneration` | ✓ |  |
| **Gemma 4**                                      | `Gemma4ForConditionalGeneration`<br>`Gemma4UnifiedForConditionalGeneration` | ✓ | E2B/E4B unsupported |
| **GLM 4**, **GLM 4.6**, etc.                     | `Glm4ForCausalLM`<br>`Glm4MoeForCausalLM` |  |  |
| **GLM 4.1V**, **GLM 4.5V**                       | `Glm4vForConditionalGeneration`<br>`Glm4vMoeForConditionalGeneration` | ✓ |  |
| **GLM 4.7 Flash**                                | `Glm4MoeLiteForCausalLM` |  |  |
| **GLM 5.2**                                      | `GlmMoeDsaForCausalLM` |  |  |
| **GLM 5.3-Flash**                                | `Glm5NextForConditionalGeneration` | ✓ |  |
| **GPT-OSS**                                      | `GptOssForCausalLM` |  |  |
| **HyperCLOVAX**                                  | `HyperCLOVAXForCausalLM`<br>`HCXVisionV2ForCausalLM` | ✓ |  |
| **Hy3**                                          | `HYV3ForCausalLM` |  |  |
| **IQuest-Coder**                                 | `IQuestCoderForCausalLM` |  |  |
| **Kimi Linear**                                  | `KimiLinearForCausalLM` |  |  |
| **Laguna 2.1**                                   | `LagunaForCausalLM` |  |  |
| **LFM 2.5**                                      | `Lfm2ForCausalLM`<br>`Lfm2MoeForCausalLM` |  |  |
| **Llama 1/2/3**,**3.1-Nemotron** etc.            | `LlamaForCausalLM` |  |  |
| **MiMo-RL**                                      | `MiMoForCausalLM` |  |  |
| **MiniMax-M2**                                   | `MiniMaxM2ForCausalLM` |  |  |
| **Mistral**, **Ministral 3**, **Mistral-4** etc. | `MistralForCausalLM`<br>`Mistral3ForConditionalGeneration` | ✓ |  |
| **Mixtral**                                      | `MixtralForCausalLM` |  |  |
| **NemotronH, Nemotron-3 Nano/Super**              | `NemotronHForCausalLM` |  |  |
| **Olmo 3.1**                                     | `Olmo3ForCausalLM` |  |  |
| **Olmo-Hybrid**                                  | `OlmoHybridForCausalLM` |  |  |
| **Phi3**, **Phi4**                               | `Phi3ForCausalLM` |  |  |
| **Qwen 2**, **Qwen 2.5**, **Qwen 2.5 VL**        | `Qwen2ForCausalLM`<br>`Qwen2_5_VLForConditionalGeneration` | ✓ |  |
| **Qwen 3**                                       | `Qwen3ForCausalLM`<br>`Qwen3MoeForCausalLM` |  |  |
| **Qwen 3-Next**                                  | `Qwen3NextForCausalLM` |  |  |
| **Qwen 3-VL**                                    | `Qwen3VLForConditionalGeneration` | ✓ |  |
| **Qwen 3-VL MoE**                                | `Qwen3VLMoeForConditionalGeneration` | ✓ |  |
| **Qwen 3.5**                                     | `Qwen3_5ForConditionalGeneration` | ✓ |  |
| **Qwen 3.5 MoE**                                 | `Qwen3_5MoeForConditionalGeneration` | ✓ |  |
| **Qwen 3.8-Flash-Next**                          | `Qwen4ExpForConditionalGeneration` | ✓ |  |
| **Seed-OSS**                                     | `SeedOssForCausalLM` |  |  |
| **SmolLM**                                       | `SmolLM3ForCausalLM` |  |  |
| **SolarOpen**                                    | `SolarOpenForCausalLM` |  |  |
| **Step 3.5 Flash**                               | `Step3p5ForCausalLM` |  |  |
| **Step 3.7 Flash**                               | `Step3p7ForConditionalGeneration` | ✓ |  |

Always adding more, stay tuned.

## Conversion

To convert a model to EXL3 format, use:

```sh
# Convert model
python convert.py -i <input_dir> -o <output_dir> -w <working_dir> -b <bitrate>

# Resume an interrupted quant job
python convert.py -w <working_dir> -r

# More options
python convert.py -h
```

The working directory is temporary storage for state checkpoints and for storing quantized tensors 
until the converted model can be compiled. It should have enough free space to store an entire copy 
of the output model.

See the [conversion guide](doc/convert.md) for more information, or the 
[self-calibration guide](doc/optimize.md). 

## EXL3 quantization

EXL3 quantization is a streamlined variant of [**QTIP**](https://github.com/Cornell-RelaxML/qtip) from Cornell RelaxML. It aims to make
SOTA quantization available to users on consumer hardware. The conversion process is designed to be
simple and efficient and requires only an input model (in HF format) and a target bitrate. By
computing Hessians on the fly and thanks to a fused Viterbi kernel, the quantizer can convert a 
model in a single step, taking a couple of minutes for smaller models, up to a few hours for larger
ones (70B+) on a single high-end consumer GPU (see the [conversion guide](doc/convert.md)).

For more information, see the [**QTIP**](https://arxiv.org/abs/2406.11235) and [**QuIP#**](https://arxiv.org/abs/2402.04396) papers, as well as this 
[excellent writeup](https://www.together.ai/blog/even-better-even-faster-quantized-llms-with-qtip) on **QTIP** from together.ai.


## Community

You are always welcome to join the [ExLlama discord server](https://discord.gg/NSFwVuCjRq) ←🎮


### 🤗 Models on Hugging Face

Browse the [EXL3 model collection](https://huggingface.co/collections/turboderp/exl3-models-67f2dfe530f05cb9f596d21a) for quantized models. Also shout out to the following lovely
people:

- [ArtusDev](https://huggingface.co/ArtusDev)
- [MikeRoz](https://huggingface.co/MikeRoz)
- [MetaphoricalCode](https://huggingface.co/MetaphoricalCode)
- [Ready.Art](https://huggingface.co/ReadyArt)
- [isogen](https://huggingface.co/isogen/models)


## Acknowledgements

This project owes its existence to a wonderful community of FOSS developers and some very generous
supporters (🐈❤️!) The following projects in particular deserve a special mention:

- [TabbyAPI](https://github.com/theroyallab/tabbyAPI/)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [QTIP](https://github.com/Cornell-RelaxML/qtip)
- [Transformers](https://github.com/huggingface/transformers)
- [Marlin](https://github.com/IST-DASLab/marlin)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) (chunked linear-attention prefill kernels, vendored under `exllamav3/vendor/fla`)

<p align="center">
  <img src="doc/cat.png" width="40" alt="">
</p>


