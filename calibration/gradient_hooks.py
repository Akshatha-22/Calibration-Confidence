"""Utilities for tracking gradient norms during training.

This module provides small, framework-agnostic utilities, with a primary
focus on PyTorch models. The main goal is to record gradient magnitudes
over time (e.g. over many training steps) in order to diagnose *why*
calibration can fail, especially for sequence models such as RNNs/LSTMs.

Typical usage with PyTorch:

.. code-block:: python

    from calibration.gradient_hooks import GradientNormTracker, register_gradient_norm_hooks

    model = ...
    tracker = GradientNormTracker()

    # Attach hooks to all parameters whose name matches a predicate.
    handles = register_gradient_norm_hooks(
        model,
        tracker,
        step_getter=lambda: global_step,
        module_name_filter=lambda name, module: "rnn" in name.lower() or "lstm" in name.lower(),
    )

    for global_step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()

    # After training:
    data = tracker.as_arrays()
    # data["steps"], data["param_names"], data["grad_norms"]

You can then analyze gradient norm statistics for recurrent layers vs.
other layers, correlate with calibration metrics, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

try:  # Optional PyTorch dependency
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - allows non-PyTorch environments
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False


@dataclass
class GradientNormTracker:
    """Track per-parameter gradient norms over training steps.

    The tracker is intentionally minimal: it just stores triples
    (step, param_name, grad_norm). You can later aggregate these by
    parameter, by module type (e.g. RNN/LSTM), or by step to inspect
    exploding/vanishing gradients and relate them to calibration.
    """

    steps: List[int] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)

    def record(self, step: int, param_name: str, grad: "torch.Tensor") -> None:
        """Record the L2 norm of a gradient tensor.

        Args:
            step: Global training step index.
            param_name: String name of the parameter.
            grad: Gradient tensor for that parameter.
        """
        if grad is None:
            return
        # Use .detach() to avoid autograd tracking and .float() for stability.
        norm = float(grad.detach().float().norm().cpu().item())
        self.steps.append(step)
        self.param_names.append(param_name)
        self.grad_norms.append(norm)

    def as_arrays(self) -> Dict[str, np.ndarray]:
        """Return recorded data as NumPy arrays."""
        return {
            "steps": np.asarray(self.steps, dtype=np.int64),
            "param_names": np.asarray(self.param_names, dtype=object),
            "grad_norms": np.asarray(self.grad_norms, dtype=np.float32),
        }


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for gradient norm hooks but is not installed."
        )


def register_gradient_norm_hooks(
    model: "nn.Module",
    tracker: GradientNormTracker,
    *,
    step_getter: Callable[[], int],
    module_name_filter: Optional[Callable[[str, "nn.Module"], bool]] = None,
    param_name_filter: Optional[Callable[[str, "torch.nn.Parameter"], bool]] = None,
):
    """Register gradient hooks that record per-parameter gradient norms.

    Args:
        model:
            PyTorch ``nn.Module`` whose parameters you want to track.
        tracker:
            ``GradientNormTracker`` instance that will collect data.
        step_getter:
            Callable with no arguments returning the *current* global
            training step (e.g. a closure over an integer counter).
        module_name_filter:
            Optional predicate ``(module_name, module) -> bool``. If
            provided, only parameters belonging to modules for which this
            returns True will have hooks registered. This is useful for
            focusing on recurrent layers (e.g. RNN/LSTM/GRU) when studying
            calibration failures.
        param_name_filter:
            Optional predicate ``(full_param_name, param) -> bool``. If
            provided, only parameters for which this returns True will
            have hooks registered.

    Returns:
        List of hook handles. Keep these around if you later want to
        remove the hooks via ``handle.remove()``.
    """
    _check_torch()

    handles = []

    # Iterate modules so we can optionally filter by module name/type.
    for module_name, module in model.named_modules():
        if module_name_filter is not None and not module_name_filter(module_name, module):
            continue

        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name

            if param_name_filter is not None and not param_name_filter(full_name, param):
                continue

            def _make_hook(name: str):
                def _hook(grad):
                    step = int(step_getter())
                    tracker.record(step, name, grad)

                return _hook

            handle = param.register_hook(_make_hook(full_name))
            handles.append(handle)

    return handles
