import torch


def softmax_int32(x):
    return torch.ops.smelt.softmax(x.contiguous())
