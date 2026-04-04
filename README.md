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
model = torch.compile(model)  # recommended, fuses surrounding ops + removes redundant conversions
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
model = torch.compile(model)
```

## Install

```
uv sync
uv pip install -e ".[train]"      # onebitllms + trl + datasets
uv pip install -e ".[bench]"      # psutil + matplotlib
uv pip install -e ".[notebooks]"  # matplotlib + jupyter
```

## Todo

- NEON fallback (ARM / Apple Silicon)
- model serialization (save/load without re-quantizing)
