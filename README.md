# smelt

Fast CPU inference for ternary LLMs.

- Pack any already-ternary HF model (BitNet, Falcon-E) into TL1 format with AVX2 SIMD kernels.
- Smelt (quantize and fuse) float/arbitrary pretrained models using [PTQTP](https://arxiv.org/abs/2509.16989) two-plane decomposition, provides 9x better reconstruction than absmean.
  
```python
import smelt
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
smelt.quantize(model)
model.generate(...)
```

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

- NEON fallback (ARM / Apple Silicon)
- model serialization (save/load without re-quantizing)
