import os

import pandas as pd
import torch

from data.preprocessing.finsen_loader import FinSenDataset


def _make_csv(path, rows=10):
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=rows, freq="D"),
        "value": range(rows),
    })
    df.to_csv(path, index=False)


def test_finsen_dataset_basic(tmp_path):
    # Create a minimal dataset folder with a single CSV file
    data_dir = tmp_path / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    _make_csv(data_dir / "sample.csv", rows=10)

    ds = FinSenDataset(data_path=str(data_dir), seq_length=3)

    assert len(ds) == 7  # rows - seq_length
    seq, target = ds[0]
    assert isinstance(seq, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert seq.shape == (3, 1)
    assert target.shape == (1,)


def test_finsen_dataset_numeric_only(tmp_path):
    # Ensure non-numeric columns are dropped yet numeric columns remain
    data_dir = tmp_path / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "text": ["a", "b", "c", "d", "e"],
        "value": [1, 2, 3, 4, 5],
    })
    df.to_csv(data_dir / "sample.csv", index=False)

    ds = FinSenDataset(data_path=str(data_dir), seq_length=2)
    assert ds.values.shape[1] == 1  # only numeric column kept
