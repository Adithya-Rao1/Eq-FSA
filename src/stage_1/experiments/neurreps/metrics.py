from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

class EmbeddingCapture:
    def __init__(self, embedding_module: nn.Embedding):
        self.embedding_module = embedding_module
        self.captured: Optional[torch.Tensor] = None
        self._handle = None

    def __enter__(self) -> "EmbeddingCapture":
        def hook(module, inputs, output):
            output.retain_grad()
            self.captured = output
        self._handle = self.embedding_module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._handle is not None:
            self._handle.remove()


def jacobian_sensitivity(
    logits: torch.Tensor,
    embedding_output: torch.Tensor,
    source_idx: torch.Tensor,
) -> float:
    signal = logits.pow(2).sum()
    (grad,) = torch.autograd.grad(signal, embedding_output, retain_graph=True)
    B = grad.shape[0]
    source_grad = grad[torch.arange(B, device=grad.device), source_idx]
    return float(source_grad.norm(dim=-1).mean().item())


def mi_lower_bound(classes: int, ce_val: float) -> float:
    return math.log(classes) - ce_val
