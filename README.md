# smelt
LLMs for integer ops only. Ternary quantization + bit-shift piecewise linear + fine-tuning.

`float in -> int8 -> [ternary GEMM (int32 acc) -> PLAC activation (int32 shifts) -> rescale to int8] x N layers -> float out`
