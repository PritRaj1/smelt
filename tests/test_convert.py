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


def _small_gpt2():
    return GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=4, n_embd=128, vocab_size=1000))


def _small_llama():
    return LlamaForCausalLM(
        LlamaConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            vocab_size=1000,
        )
    )


def _small_bitnet():
    return BitNetForCausalLM(
        BitNetConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=1000,
        )
    )


def _check_converted(model, vocab=1000):
    """Verify layers replaced and forward pass works."""
    n = sum(1 for m in model.modules() if isinstance(m, TernaryLinear))
    assert n > 0, "no layers converted"

    out = model(torch.randint(0, vocab, (1, 16)))
    assert out.logits.shape == (1, 16, vocab)


def test_gpt2():
    model = _small_gpt2()
    smelt.quantize(model)
    _check_converted(model)


def test_llama():
    model = _small_llama()
    smelt.quantize(model)
    _check_converted(model)


def test_bitnet():
    model = _small_bitnet()
    smelt.quantize(model)
    _check_converted(model)


def test_skip():
    model = _small_llama()
    smelt.quantize(model, skip=["lm_head", "model.layers.0"])

    assert isinstance(model.model.layers[0].self_attn.q_proj, torch.nn.Linear)
    assert isinstance(model.model.layers[1].self_attn.q_proj, TernaryLinear)
