import torch


def rmsnorm_int32(x, gamma):
    return torch.ops.smelt.rmsnorm(x.contiguous(), gamma.contiguous())


def layernorm_int32(x, gamma, beta):
    return torch.ops.smelt.layernorm(x.contiguous(), gamma.contiguous(), beta.contiguous())
