# smelt

Fast CPU inference for ternary LLMs.

- Pack any already-ternary HF model (BitNet, Falcon-E) into TL1 format with AVX2 SIMD kernels.
- Smelt (quantize and fuse) floating point/arbitrary pretrained models using [PTQTP](https://arxiv.org/abs/2509.16989) two-plane decomposition, (provides 9x better reconstruction than absmean).
- `torch.compile` compatible. Smelt ops registered via `torch.library` for graph-level fusion to avoid all conversion head (since the model is pure int C kernels).
    
  ```python
  import smelt, torch
  from transformers import AutoModelForCausalLM

  model = AutoModelForCausalLM.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
  smelt.quantize(model)
  model.generation_config.cache_implementation = "static"
  model.forward = torch.compile(model.forward, fullgraph=True)
  model.generate(...)
  ```

## Training

smelt is inference-only. Train ternary models with [onebitllms](https://github.com/tiiuae/onebitllms), then pack with `smelt.quantize()`. See [notebook 08](notebooks/08_qat.ipynb) for a full example.

  ```python
from onebitllms import replace_linear_with_bitnet_linear, quantize_to_1bit
from transformers import AutoModelForCausalLM

# 1. insert ternary layers + train on GPU
  model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", ...)
  model = replace_linear_with_bitnet_linear(model)
  trainer = SFTTrainer(model=model, ...)
  trainer.train()

# 2. freeze ternary weights
  quantize_to_1bit("checkpoint/", "checkpoint_1bit/")

# 3. smelt for CPU inference
  model = AutoModelForCausalLM.from_pretrained("checkpoint_1bit/")
  smelt.quantize(model)
  model.forward = torch.compile(model.forward, fullgraph=True)
  ```

## Install

  ```
  uv sync
  uv pip install -e ".[train]"      # onebitllms + trl + datasets
  uv pip install -e ".[bench]"      # psutil + matplotlib
  uv pip install -e ".[notebooks]"  # matplotlib + jupyter
  ```

## Already-ternary models (BitNet, Falcon-E)

  | layer | pipeline | kernel |
  |:---|:---|:---|
  | linear (q/k/v/o/gate/up/down) | `int8 -> ternary GEMM -> int32 -> float` | `ternary_gemm.c` (AVX2 vpshufb) |
  | attention QK^T | `float Q,K -> int8 -> int8 GEMM -> int32` | `int8_gemm.c` |
  | attention softmax | `int32 -> LUT softmax -> int32 (Q16.16)` | `softmax_int.c` |
  | attention AV | `int32 attn -> int8, float V -> int8, int8 GEMM -> int32 -> float` | `int8_gemm.c` |
  | lm_head | `float -> int8 -> int8 GEMV -> int32 -> float` | `int8_gemm.c` |
  | norms | float (HF native) | — |

## Arbitrary float models (PTQTP)

  | layer | pipeline | kernel |
  |:---|:---|:---|
  | linear (q/k/v/o/gate/up/down) | `int8 -> dual ternary GEMM × 2 -> int32 -> float` | `ternary_gemm.c` × 2 |
  | activation (SiLU/GELU) | `float -> Q16.16 -> PLAC shifts+adds -> Q16.16 -> float` | `plac_eval.c` (AVX2) |
  | attention QK^T | `float Q,K -> int8 -> int8 GEMM -> int32` | `int8_gemm.c` |
  | attention softmax | `int32 -> LUT softmax -> int32 (Q16.16)` | `softmax_int.c` |
  | attention AV | `int32 attn -> int8, float V -> int8, int8 GEMM -> int32 -> float` | `int8_gemm.c` |
  | lm_head | `float -> int8 -> int8 GEMV -> int32 -> float` | `int8_gemm.c` |
  | norms | float (HF native) | — |

  Float boundaries: norms (precision-sensitive) and the final rescale after each GEMM. Everything else in int C kernels.

## Todo

  - NEON fallback (ARM / Apple Silicon)
  - model serialization (save/load without re-quantizing)
  - per-channel calibration for int8 attention on PTQTP models
