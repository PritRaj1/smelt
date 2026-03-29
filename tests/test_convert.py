import pytest
import torch
from transformers import (
    BitNetConfig,
    BitNetForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
)

import smelt
from smelt.quantize import TernaryLinear

_MODELS = {
    "gpt2": lambda: GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=4, n_embd=128, vocab_size=1000)),
    "llama": lambda: LlamaForCausalLM(
        LlamaConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            vocab_size=1000,
        )
    ),
    "bitnet": lambda: BitNetForCausalLM(
        BitNetConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
        )
    ),
}


@pytest.mark.parametrize("arch", _MODELS.keys())
def test_convert(arch):
    model = _MODELS[arch]()
    smelt.quantize(model)

    n = sum(1 for m in model.modules() if isinstance(m, TernaryLinear))
    assert n > 0, f"no layers converted for {arch}"

    out = model(torch.randint(0, 1000, (1, 16)))
    assert out.logits.shape == (1, 16, 1000)


def test_skip():
    model = _MODELS["llama"]()
    smelt.quantize(model, skip=["lm_head", "model.layers.0"])

    assert isinstance(model.model.layers[0].self_attn.q_proj, torch.nn.Linear)
    assert isinstance(model.model.layers[1].self_attn.q_proj, TernaryLinear)
