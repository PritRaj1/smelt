import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

import smelt
from smelt.quantize import TernaryLinear


def _small_gpt2():
    config = GPT2Config(n_layer=2, n_head=4, n_embd=128, vocab_size=1000)
    return GPT2LMHeadModel(config)


def _small_llama():
    config = LlamaConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        vocab_size=1000,
    )
    return LlamaForCausalLM(config)


def test_gpt2():
    model = _small_gpt2()
    smelt.quantize(model)

    n_ternary = sum(1 for m in model.modules() if isinstance(m, TernaryLinear))
    assert n_ternary > 0, "no layers converted"

    out = model(torch.randint(0, 1000, (1, 16)))
    assert out.logits.shape == (1, 16, 1000)


def test_llama():
    model = _small_llama()
    smelt.quantize(model)

    n_ternary = sum(1 for m in model.modules() if isinstance(m, TernaryLinear))
    assert n_ternary > 0, "no layers converted"

    out = model(torch.randint(0, 1000, (1, 16)))
    assert out.logits.shape == (1, 16, 1000)


def test_skip():
    model = _small_llama()
    smelt.quantize(model, skip=["lm_head", "model.layers.0"])

    # layer 0 should be untouched
    assert isinstance(model.model.layers[0].self_attn.q_proj, torch.nn.Linear)

    # layer 1 should be converted
    assert isinstance(model.model.layers[1].self_attn.q_proj, TernaryLinear)
