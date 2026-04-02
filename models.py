"""
Pure Feedforward MLP Model - No Hidden State
"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Pure feedforward MLP with no hidden state persistence.

    This is a standard feedforward neural network that processes
    each input independently with no recurrence or memory.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        """
        Initialize MLP.

        Args:
            input_size: Size of input features (flattened window)
            hidden_size: Size of hidden layer
            output_size: Size of output
        """
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),  # Flatten input to 1D
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
                or (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
                or (batch_size, output_size)
        """
        # Flatten the input if it's 3D (batch, seq, features)
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, seq_len, feature_size = original_shape
            # Flatten sequence dimension into batch dimension
            x = x.view(batch_size * seq_len, feature_size)

        # Forward through layers
        output = self.layers(x)

        # Reshape back if input was 3D
        if len(original_shape) == 3:
            output = output.view(batch_size, seq_len, -1)

        return output

    def reset_hidden(self):
        """
        No-op for MLP since there are no hidden states to reset.
        This method exists for compatibility with RNN interfaces.
        """
        pass

class LSTM(nn.Module):
    """
    LSTM with correct forget gate initialization.

    The forget gate bias is initialized to 1.0 to prevent the model
    from forgetting information too aggressively at the start of training.
    """

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, output_size: int = 1):
        """
        Initialize LSTM with correct forget gate bias.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            output_size: Size of output
        """
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.0 if num_layers == 1 else 0.1)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize forget gate bias to 1.0
        # In PyTorch LSTM, biases are ordered: [input_gate, forget_gate, cell_gate, output_gate]
        # So forget gate bias is at index hidden_size:2*hidden_size
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                # bias shape: (4*hidden_size,) for each layer
                bias = param
                # Forget gate bias (second quarter of the bias vector)
                forget_start = hidden_size
                forget_end = 2 * hidden_size
                bias.data[forget_start:forget_end] = 1.0

                print(f"Initialized forget gate bias to 1.0 for {name}")

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Initial hidden state (optional)

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)

        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)

        # Apply output layer to all time steps
        output = self.fc(lstm_out)

        return output, hidden

    def reset_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        """
        Reset hidden state.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple of (h0, c0)
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)</content>
</xai:function_call name="replace_string_in_file">
<parameter name="filePath">d:\Calibration-Confidence\models.py</content>
<parameter name="filePath">d:\Calibration-Confidence\models.py