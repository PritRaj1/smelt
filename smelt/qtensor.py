import torch
import torch.utils._pytree as pytree

FRAC = 16
SCALE = 1 << FRAC


class QTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, int_data):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            int_data.shape,
            dtype=torch.float32,
            device=int_data.device,
        )

    def __init__(self, int_data):
        self.int_data = int_data

    def __tensor_flatten__(self):
        return ["int_data"], []

    @classmethod
    def __tensor_unflatten__(cls, tensors, _meta, _size, _stride):
        return cls(tensors["int_data"])

    def dequantize(self):
        return self.int_data.float() / SCALE

    def __repr__(self):
        return f"QTensor(shape={list(self.shape)}, device={self.device})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}

        # detach/clone preserve QTensor
        if func is torch.ops.aten.detach.default:
            return cls(args[0].int_data.detach())
        if func is torch.ops.aten.clone.default:
            return cls(args[0].int_data.clone())

        # everything else: dequant and run normally
        def unwrap(t):
            return t.dequantize() if isinstance(t, QTensor) else t

        return func(*pytree.tree_map(unwrap, args), **pytree.tree_map(unwrap, kwargs))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        def unwrap(t):
            return t.dequantize() if isinstance(t, QTensor) else t

        return func(*pytree.tree_map(unwrap, args), **pytree.tree_map(unwrap, kwargs))
