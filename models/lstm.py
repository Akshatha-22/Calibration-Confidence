"""LSTM model for sliding-window forecasting.

This module defines a Long Short-Term Memory (LSTM) network, an advanced RNN
with gating mechanisms (forget, input, output gates) that better handle long-term
dependencies and longer sequences compared to simple RNNs.

The LSTM expects input of shape ``(batch, seq_len, num_features)`` and outputs
``(batch, output_size)`` by using the final hidden state.

Example:
    model = LSTM(input_size=num_features, hidden_size=64, output_size=num_features)

    # In training loop:
    preds = model(x)  # x: (batch, seq_len, num_features)

"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM for sequence-to-one regression.

    Uses an LSTM layer to process the sequence with gating for better long-term
    memory, then a linear layer on the final hidden state.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: Optional[int] = None,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """Initialize the LSTM.

        Args:
            input_size: Number of features per time step.
            hidden_size: Size of the LSTM hidden state.
            num_layers: Number of LSTM layers (stacked).
            output_size: Size of the output vector. If ``None``, defaults to ``input_size``.
            dropout: Dropout probability between LSTM layers (only if num_layers > 1).
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()
        self.output_size = output_size if output_size is not None else input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # If bidirectional, hidden_size * 2
        fc_input_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_input_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, num_features).

        Returns:
            Tensor of shape (batch, output_size) representing the next-step prediction.
        """
        # LSTM output: (batch, seq_len, hidden_size * directions)
        # h_n: (num_layers * directions, batch, hidden_size)
        # c_n: (num_layers * directions, batch, hidden_size)  [cell state, not used]
        out, (h_n, c_n) = self.lstm(x)

        # Use the last layer's final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch, hidden_size * 2)
        else:
            h_last = h_n[-1]  # (batch, hidden_size)

        return self.fc(h_last)


def build_lstm(
    seq_len: int,
    num_features: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    output_size: Optional[int] = None,
    dropout: float = 0.0,
    bidirectional: bool = False,
) -> LSTM:
    """Helper to build a standard LSTM.

    Args:
        seq_len: Number of past time steps used as input (not used in model, for consistency).
        num_features: Number of features per time step.
        hidden_size: Size of the LSTM hidden state.
        num_layers: Number of LSTM layers.
        output_size: Output vector length (defaults to num_features).
        dropout: Dropout probability.
        bidirectional: Whether to use bidirectional LSTM.

    Returns:
        An ``LSTM`` instance.
    """
    if output_size is None:
        output_size = num_features
    return LSTM(
        input_size=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
        bidirectional=bidirectional,
    )
