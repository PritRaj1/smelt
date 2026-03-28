# smelt
LLMs for integer ops only. Ternary quantization + bit-shift piecewise linear.

`float in -> int8 -> [ternary GEMM (int32 acc) -> PLAC activation (int32 shifts) -> rescale to int8] x N layers -> float out`

## todo

- load BitNet weight format (autobitlinear -> TL1 packed)
- model conversion API (smelt.quantize)
- int8 attention matmul (Q@K, scores@V, not ternary)
- GQA support (grouped query attention)
- QAT fine-tuning (STE for ternary weights)
- Falcon-Edge / MatMul-free LM support
- NEON fallback (ARM / Apple Silicon)
- fused layer C kernels (one Python->C call per layer)
- model serialization (save/load quantized weights)
