import torch
import torch.nn as nn

from smelt.attention import Attention, KVCache
from smelt.rope import precompute_freqs_float


def _make_attn(dim=64, n_heads=4, n_kv_heads=None):
    n_kv = n_kv_heads or n_heads
    return Attention(
        q_proj=nn.Linear(dim, dim, bias=False),
        k_proj=nn.Linear(dim, n_kv * (dim // n_heads), bias=False),
        v_proj=nn.Linear(dim, n_kv * (dim // n_heads), bias=False),
        o_proj=nn.Linear(dim, dim, bias=False),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )


def test_causal_masking():
    """Output at position i should not depend on positions > i."""
    torch.manual_seed(0)
    attn = _make_attn()
    x = torch.randn(1, 8, 64)

    y_full = attn(x)
    y_prefix = attn(x[:, :4])

    assert torch.allclose(y_full[:, :4], y_prefix, atol=1e-5)


def test_kv_cache_decode():
    """Prefill + single-token decode should match full-sequence forward."""
    torch.manual_seed(0)
    attn = _make_attn()
    cos, sin = precompute_freqs_float(16, 32)
    freqs = (cos, sin)

    x = torch.randn(1, 8, 64)

    y_ref = attn(x, freqs=freqs)

    cache = KVCache(32, 4, 16)
    y_prefill = attn(x[:, :6], freqs=freqs, cache=cache)
    y_decode_6 = attn(x[:, 6:7], freqs=freqs, cache=cache)
    y_decode_7 = attn(x[:, 7:8], freqs=freqs, cache=cache)

    assert torch.allclose(y_ref[:, :6], y_prefill, atol=1e-5)
    assert torch.allclose(y_ref[:, 6:7], y_decode_6, atol=1e-5)
    assert torch.allclose(y_ref[:, 7:8], y_decode_7, atol=1e-5)


def test_gqa_with_cache():
    """GQA + KV cache should match full forward."""
    torch.manual_seed(0)
    attn = _make_attn(dim=64, n_heads=4, n_kv_heads=2)
    cos, sin = precompute_freqs_float(16, 32)
    freqs = (cos, sin)
    x = torch.randn(1, 6, 64)

    y_ref = attn(x, freqs=freqs)

    cache = KVCache(32, 2, 16)
    y_prefill = attn(x[:, :4], freqs=freqs, cache=cache)
    y_rest = attn(x[:, 4:], freqs=freqs, cache=cache)

    assert torch.allclose(y_ref[:, :4], y_prefill, atol=1e-5)
    assert torch.allclose(y_ref[:, 4:], y_rest, atol=1e-5)
