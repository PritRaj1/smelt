# smelt

Fast CPU inference for ternary LLMs. Pack any already-ternary HuggingFace model (BitNet, Falcon-E) into TL1 format with AVX2 SIMD kernels.

```python
import smelt
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
smelt.quantize(model)
model.generate(...)
```

## Kernels

| op | technique |
|:---|:---|
| ternary GEMM | TL1 LUT + vpshufb + int16 acc + OpenMP |
| int8 attention (QK^T) | tiled dot products + AVX2 madd_epi16 |
| SiLU/GELU | PLAC: piecewise linear, dense LUT |
| softmax | base-2 exp LUT + AVX2 gather + reciprocal multiply |
| RoPE | precomputed sin/cos tables |

## Training

smelt is inference-only. Train ternary models with [onebitllms](https://github.com/tiiuae/onebitllms), then pack with `smelt.quantize()`.

## Install

```
uv sync
uv pip install -e ".[train]"      # onebitllms + trl + datasets
uv pip install -e ".[bench]"      # llama-cpp-python
uv pip install -e ".[notebooks]"  # matplotlib + jupyter
```

## Todo

- PTQTP (post-training float -> ternary, code pending upstream release)
- NEON fallback (ARM / Apple Silicon)
- model serialization (save/load without re-quantizing)
