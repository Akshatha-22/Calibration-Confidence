import torch

from models.mlp import build_mlp
from models.deep_mlp import build_deep_mlp
from models.residual_mlp import build_residual_mlp
from models.vanilla_rnn import build_vanilla_rnn
from models.lstm import build_lstm


def _run_forward(model_factory, seq_len=10, num_features=2, batch_size=4):
    model = model_factory(seq_len, num_features)
    x = torch.randn(batch_size, seq_len, num_features)
    out = model(x)
    assert out.shape == (batch_size, num_features)


def test_mlp_forward():
    _run_forward(build_mlp)


def test_deep_mlp_forward():
    _run_forward(build_deep_mlp)


def test_residual_mlp_forward():
    _run_forward(build_residual_mlp)


def test_vanilla_rnn_forward():
    _run_forward(build_vanilla_rnn)


def test_lstm_forward():
    _run_forward(build_lstm)
