# smelt

Integer-only LLM inference. Quantize any HuggingFace model to ternary weights + int8 activations. Run on CPU with zero float ops in the forward pass.

```python
import smelt
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
smelt.quantize(model)
model.generate(...)
```

## Pipeline

```
int8 -> ternary GEMM (int32) -> activation (int32 shifts) -> rescale to int8 -> next layer
```

Unlike BitNet.cpp which converts to float for activations and norms, smelt stays integer throughout.

| op | technique |
|:---|:----------|
| linear | TL1 LUT + vpshufb + OpenMP |
| SiLU/GELU | PLAC: bit-shift piecewise linear, dense LUT |
| softmax | base-2 exp LUT + AVX2 gather + reciprocal multiply |
| RMSNorm | clz + isqrt LUT |
| LayerNorm | mean + clz + isqrt LUT |
| RoPE | precomputed sin/cos in Q16.16 |

All C kernels compiled on first use: `gcc -O3 -march=native -flto -fopenmp`.

## Install

```
uv sync
```

## Todo

- int8 attention matmul (Q@K, scores@V)
- GQA (grouped query attention)
- QAT fine-tuning (STE for ternary weights)
- Falcon-Edge / MatMul-free LM support
- NEON fallback (ARM / Apple Silicon)
- fused layer C kernels
- model serialization
