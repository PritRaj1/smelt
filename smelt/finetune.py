import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import quantize


class _STELinear(nn.Module):
    """Float shadow weights, ternary forward via straight-through estimator."""

    def __init__(self, linear):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data.float())
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x):
        scale = self.weight.abs().mean()
        w_t = (self.weight / (scale + 1e-10)).round().clamp(-1, 1)
        w_ste = self.weight + (w_t * scale - self.weight).detach()
        return F.linear(x, w_ste, self.bias)


def prepare_qat(model, skip=None):
    """Replace linears with STE-trainable ternary layers."""
    skip = skip or ["lm_head"]
    for name, mod in list(model.named_modules()):
        if any(name.startswith(s) or name == s for s in skip):
            continue

        if isinstance(mod, nn.Linear):
            model.set_submodule(name, _STELinear(mod))

    return model


def freeze(model, skip=None, target_mae=1e-2):
    """Convert STE layers to TernaryLinear, fit PLAC + int norms."""
    skip = skip or ["lm_head"]
    for name, mod in list(model.named_modules()):
        if any(name.startswith(s) or name == s for s in skip):
            continue

        if isinstance(mod, _STELinear):
            linear = nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None)
            linear.weight.data = mod.weight.data
            if mod.bias is not None:
                linear.bias.data = mod.bias.data
            model.set_submodule(name, linear)

    return quantize(model, skip=skip, target_mae=target_mae)


def finetune(model, dataset, epochs=1, lr=1e-4, skip=None, target_mae=1e-2, **trainer_kwargs):
    """QAT fine-tune then freeze for smelt inference. Requires optional deps."""
    from trl import SFTConfig, SFTTrainer

    prepare_qat(model, skip=skip)

    cfg = SFTConfig(
        output_dir=trainer_kwargs.pop("output_dir", "/tmp/smelt_qat"),
        num_train_epochs=epochs,
        learning_rate=lr,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        **trainer_kwargs,
    )

    trainer = SFTTrainer(model=model, train_dataset=dataset, args=cfg)
    trainer.train()
    return freeze(model, skip=skip, target_mae=target_mae)
