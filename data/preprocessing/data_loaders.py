import os
from typing import Tuple, Optional

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset



class FinSenDataset(Dataset):
    """Torch Dataset for FinSen text classification."""

    def __init__(
        self,
        texts,
        labels,
        vectorizer: Optional[TfidfVectorizer] = None,
        max_features: int = 2000,
    ):
        # If a vectorizer is not provided, we fit a TF-IDF vectorizer on the texts.
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            self.features = self.vectorizer.fit_transform(texts).toarray()
        else:
            self.vectorizer = vectorizer
            self.features = self.vectorizer.transform(texts).toarray()

        # Map labels to integers if they are not already numeric
        if labels.dtype == object or labels.dtype == str:
            self.label_map = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
            self.labels = labels.map(self.label_map).values
        else:
            self.label_map = None
            self.labels = labels.values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class SequenceFinSenDataset(Dataset):
    """Torch Dataset for sequence-based models (RNN/LSTM)."""

    def __init__(
        self,
        texts,
        labels,
        vocab=None,
        max_seq_len=128,
        min_freq=2,
        unk_token='<UNK>',
        pad_token='<PAD>',
    ):
        self.max_seq_len = max_seq_len
        self.unk_token = unk_token
        self.pad_token = pad_token

        # Build vocab from training texts if not supplied
        if vocab is None:
            self.vocab = self._build_vocab(texts, min_freq=min_freq)
        else:
            self.vocab = vocab

        self.sequences = [
            self._text_to_sequence(text) for text in texts
        ]
        self.sequences = self._pad_sequences(self.sequences, max_len=self.max_seq_len)

        # Map labels to integers if they are not already numeric
        if labels.dtype == object or labels.dtype == str:
            self.label_map = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
            self.labels = labels.map(self.label_map).values
        else:
            self.label_map = None
            self.labels = labels.values

    def _build_vocab(self, texts, min_freq=2):
        token_counts = {}
        for text in texts:
            for token in self._tokenize(text):
                token_counts[token] = token_counts.get(token, 0) + 1

        vocab = {self.pad_token: 0, self.unk_token: 1}
        idx = 2
        for token, freq in sorted(token_counts.items(), key=lambda x: (-x[1], x[0])):
            if freq < min_freq:
                continue
            vocab[token] = idx
            idx += 1

        return vocab

    def _tokenize(self, text: str):
        # Simple whitespace tokenizer; replace with a better tokenizer as needed
        return str(text).lower().split()

    def _text_to_sequence(self, text: str):
        tokens = self._tokenize(text)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def _pad_sequences(self, sequences, max_len):
        padded = []
        pad_id = self.vocab[self.pad_token]
        for seq in sequences:
            if len(seq) >= max_len:
                padded.append(seq[:max_len])
            else:
                padded.append(seq + [pad_id] * (max_len - len(seq)))
        return padded

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def _load_split(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_csv(path)


def _build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 64,
    max_features: int = 2000,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build torch DataLoaders using a shared TF-IDF vectorizer."""
    # Use title + content as the combined text input
    def _make_text(df: pd.DataFrame) -> pd.Series:
        return (df['Title'].fillna('') + ' ' + df['Content'].fillna('')).astype(str)

    train_texts = _make_text(train_df)
    val_texts = _make_text(val_df)
    test_texts = _make_text(test_df)

    train_labels = train_df['Category']
    val_labels = val_df['Category']
    test_labels = test_df['Category']

    # Fit vectorizer on training data only
    train_dataset = FinSenDataset(train_texts, train_labels, vectorizer=None, max_features=max_features)
    vectorizer = train_dataset.vectorizer

    val_dataset = FinSenDataset(val_texts, val_labels, vectorizer=vectorizer)
    test_dataset = FinSenDataset(test_texts, test_labels, vectorizer=vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_mlp_loaders(
    train_path: str = 'data/finsen/processed/train.csv',
    val_path: str = 'data/finsen/processed/val.csv',
    test_path: str = 'data/finsen/processed/test.csv',
    batch_size: int = 64,
    max_features: int = 2000,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get DataLoaders for MLP models (uses TF-IDF features)."""
    train_df = _load_split(train_path)
    val_df = _load_split(val_path)
    test_df = _load_split(test_path)
    return _build_dataloaders(train_df, val_df, test_df, batch_size=batch_size, max_features=max_features)


def get_rnn_loaders(
    train_path: str = 'data/finsen/processed/train.csv',
    val_path: str = 'data/finsen/processed/val.csv',
    test_path: str = 'data/finsen/processed/test.csv',
    batch_size: int = 64,
    max_seq_len: int = 128,
    min_freq: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get DataLoaders for RNN/LSTM models using token sequences.

    The loader builds a vocabulary on the training set and pads/truncates
    sequences to `max_seq_len`. This provides real sequential inputs instead of
    TF-IDF vectors.
    """

    def _make_text(df: pd.DataFrame) -> pd.Series:
        return (df['Title'].fillna('') + ' ' + df['Content'].fillna('')).astype(str)

    train_df = _load_split(train_path)
    val_df = _load_split(val_path)
    test_df = _load_split(test_path)

    train_texts = _make_text(train_df)
    val_texts = _make_text(val_df)
    test_texts = _make_text(test_df)

    train_labels = train_df['Category']
    val_labels = val_df['Category']
    test_labels = test_df['Category']

    train_dataset = SequenceFinSenDataset(
        train_texts,
        train_labels,
        vocab=None,
        max_seq_len=max_seq_len,
        min_freq=min_freq,
    )
    vocab = train_dataset.vocab

    val_dataset = SequenceFinSenDataset(
        val_texts,
        val_labels,
        vocab=vocab,
        max_seq_len=max_seq_len,
    )

    test_dataset = SequenceFinSenDataset(
        test_texts,
        test_labels,
        vocab=vocab,
        max_seq_len=max_seq_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



def get_all_model_loaders(
    train_path: str = 'data/finsen/processed/train.csv',
    val_path: str = 'data/finsen/processed/val.csv',
    test_path: str = 'data/finsen/processed/test.csv',
    batch_size: int = 64,
    max_features: int = 2000,
    max_seq_len: int = 128,
    min_freq: int = 2,
) -> dict:
    """Return a dict with DataLoader tuples for each model type."""
    loaders = {
        'mlp': get_mlp_loaders(train_path, val_path, test_path, batch_size, max_features),
        'deep_mlp': get_mlp_loaders(train_path, val_path, test_path, batch_size, max_features),
        'residual_mlp': get_mlp_loaders(train_path, val_path, test_path, batch_size, max_features),
        'vanilla_rnn': get_rnn_loaders(train_path, val_path, test_path, batch_size, max_seq_len, min_freq),
        'lstm': get_rnn_loaders(train_path, val_path, test_path, batch_size, max_seq_len, min_freq),
    }
    return loaders


if __name__ == '__main__':
    loaders = get_all_model_loaders()
    print('Loaded data loaders for:', list(loaders.keys()))
    for name, (tr, val, te) in loaders.items():
        print(f"{name} -> train: {len(tr.dataset)}, val: {len(val.dataset)}, test: {len(te.dataset)}")
