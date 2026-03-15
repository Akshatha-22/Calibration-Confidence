"""Residual MLP model for sliding-window forecasting.

This module defines a residual MLP with skip connections that add the input
to the output of each block, helping with gradient flow in deep networks.

The model flattens the input window and passes it through multiple residual
blocks, each with skip connections, before a final linear layer.

Example:
    model = ResidualMLP(input_size=seq_len * num_features, hidden_size=128, num_blocks=3, output_size=num_features)

    # In training loop:
    preds = model(x)  # x: (batch, seq_len, num_features)

"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A residual block with two linear layers and a skip connection."""

    def __init__(self, size: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out += residual  # Skip connection
        out = self.activation(out)
        return out


class ResidualMLP(nn.Module):
    """Residual MLP with skip connections for better gradient flow.

    Stacks multiple residual blocks, each with skip connections, followed by
    a final linear layer for output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_blocks: int = 3,
        output_size: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Initialize the ResidualMLP.

        Args:
            input_size: Size of the flattened input (seq_len * num_features).
            hidden_size: Size of hidden layers in each residual block.
            num_blocks: Number of residual blocks to stack.
            output_size: Size of the output vector. If ``None``, defaults to ``input_size``.
            dropout: Dropout probability in each block.
        """
        super().__init__()
        self.output_size = output_size if output_size is not None else input_size

        # Initial projection to hidden_size if different
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

        # Stack residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)
        ])

        # Final output layer
        self.output_layer = nn.Linear(hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, num_features).

        Returns:
            Tensor of shape (batch, output_size) representing the next-step prediction.
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        x_proj = self.input_proj(x_flat)

        for block in self.blocks:
            x_proj = block(x_proj)

        return self.output_layer(x_proj)


def build_residual_mlp(
    seq_len: int,
    num_features: int,
    hidden_size: int = 128,
    num_blocks: int = 3,
    output_size: Optional[int] = None,
    dropout: float = 0.0,
) -> ResidualMLP:
    """Helper to build a standard ResidualMLP.

    Args:
        seq_len: Number of past time steps used as input.
        num_features: Number of features per time step.
        hidden_size: Size of hidden layers in blocks.
        num_blocks: Number of residual blocks.
        output_size: Output vector length (defaults to num_features).
        dropout: Dropout probability.

    Returns:
        A ``ResidualMLP`` instance.
    """
    input_size = seq_len * num_features
    if output_size is None:
        output_size = num_features
    return ResidualMLP(
        input_size=input_size,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        output_size=output_size,
        dropout=dropout,
    )
