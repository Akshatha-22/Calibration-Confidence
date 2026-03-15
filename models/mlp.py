"""Simple MLP model for sliding-window forecasting.

This module defines a small fully-connected network that consumes a fixed
window of past observations (seq_length x num_features) and predicts the next
value(s) for the same set of features.

The model flattens the input window into a single vector per batch and passes it
through 2-3 dense layers with optional dropout and activation.

Example:
    model = MLP(input_size=seq_len * num_features, hidden_sizes=[128, 64], output_size=num_features)

    # In training loop:
    preds = model(x)  # x: (batch, seq_len, num_features)

"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple feedforward network for sequence-to-one regression.

    The network expects input of shape ``(batch, seq_len, num_features)`` and
    flattens it to ``(batch, seq_len * num_features)`` before applying linear
    layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Iterable[int] = (128, 64),
        output_size: Optional[int] = None,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
    ):
        """Initialize the MLP.

        Args:
            input_size: Size of the flattened input (seq_len * num_features).
            hidden_sizes: Sizes of hidden layers (2-3 recommended).
            output_size: Size of the output vector. If ``None``, defaults to
                ``input_size`` (useful when predicting the same number of features as input).
            dropout: Dropout probability between hidden layers.
            activation: Activation module to use after each hidden layer. If ``None``, uses ``nn.ReLU``.
        """
        super().__init__()
        self.output_size = output_size if output_size is not None else input_size
        activation = activation or nn.ReLU

        layers: list[nn.Module] = []
        in_features = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden

        layers.append(nn.Linear(in_features, self.output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, num_features).

        Returns:
            Tensor of shape (batch, output_size) representing the next-step prediction.
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.model(x_flat)


def build_mlp(
    seq_len: int,
    num_features: int,
    hidden_sizes: Iterable[int] = (128, 64),
    dropout: float = 0.0,
    output_size: Optional[int] = None,
    activation: Optional[nn.Module] = None,
) -> MLP:
    """Helper to build a standard windowed MLP.

    Args:
        seq_len: Number of past time steps used as input.
        num_features: Number of features per time step.
        hidden_sizes: Hidden layer sizes.
        dropout: Dropout probability.
        output_size: Output vector length (defaults to num_features).
        activation: Activation module (defaults to `nn.ReLU`).

    Returns:
        An ``MLP`` instance.
    """

    input_size = seq_len * num_features
    if output_size is None:
        output_size = num_features
    return MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout=dropout,
        activation=activation,
    )
