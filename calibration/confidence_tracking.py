"""Utilities for tracking confidence trajectories over time.

This module focuses on *token-level / timestep-level* confidence tracking.
It is designed for sequence models (e.g. language models) where we want to
analyze how confidence and calibration evolve across timesteps in a sequence.

Key ideas:
- For every example in a batch and every timestep in its sequence, we record:
  - model confidence (max probability over classes)
  - predicted class (argmax)
  - optional ground-truth label at that timestep
  - optional sequence id and timestep index
- We support incremental logging over many batches via ``ConfidenceTracker``.

The recorded trajectories can then be used to compute time-resolved ECE,
reliability diagrams per timestep, or other calibration diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray]


def _to_probs(
    logits_or_probs: np.ndarray,
    is_logits: bool = True,
) -> np.ndarray:
    """Convert logits to probabilities if needed.

    Args:
        logits_or_probs: Array of shape (batch, seq_len, n_classes).
        is_logits: If True, apply softmax along the last axis.

    Returns:
        Probabilities of the same shape as input.
    """
    if logits_or_probs.ndim != 3:
        raise ValueError(
            f"logits_or_probs must have shape (batch, seq_len, n_classes), "
            f"got {logits_or_probs.shape}"
        )

    if not is_logits:
        return logits_or_probs

    # Numerically stable softmax along the class dimension
    x = logits_or_probs
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return probs


def extract_confidence_trajectories(
    logits_or_probs: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    is_logits: bool = True,
    sequence_ids: Optional[np.ndarray] = None,
    pad_token_id: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Extract timestep-level confidence trajectories from a batch.

    This is the core utility that turns a batch of model outputs into
    flattened per-timestep records.

    Args:
        logits_or_probs:
            Array of shape (batch, seq_len, n_classes) containing either
            logits or probabilities.
        labels:
            Optional integer labels of shape (batch, seq_len). If provided,
            they are flattened alongside the confidences.
        is_logits:
            If True, ``logits_or_probs`` are interpreted as logits and
            converted to probabilities using softmax.
        sequence_ids:
            Optional array of shape (batch,) providing an id for each
            sequence in the batch. These will be broadcast to shape
            (batch, seq_len) and flattened.
        pad_token_id:
            Optional integer id used for padding positions in ``labels``.
            If provided, any positions where ``labels == pad_token_id``
            are removed from the returned arrays, so that only real
            (non-padding) timesteps are tracked.

    Returns:
        A dictionary with flattened arrays of equal length ``N``:

        - ``'confidences'``: shape (N,)
        - ``'predictions'``: shape (N,)
        - ``'labels'`` (optional): shape (N,)
        - ``'sequence_ids'`` (optional): shape (N,)
        - ``'timesteps'``: shape (N,) with integer timestep indices
    """
    probs = _to_probs(np.asarray(logits_or_probs), is_logits=is_logits)
    batch_size, seq_len, _ = probs.shape

    confidences = np.max(probs, axis=-1)  # (batch, seq_len)
    predictions = np.argmax(probs, axis=-1)  # (batch, seq_len)

    # Timesteps: 0..seq_len-1, broadcast to (batch, seq_len)
    timesteps = np.broadcast_to(
        np.arange(seq_len, dtype=np.int32)[None, :],
        (batch_size, seq_len),
    )

    mask = np.ones((batch_size, seq_len), dtype=bool)
    out: Dict[str, np.ndarray] = {}

    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (batch_size, seq_len):
            raise ValueError(
                f"labels must have shape (batch, seq_len) = "
                f"({batch_size}, {seq_len}), got {labels.shape}"
            )
        if pad_token_id is not None:
            # Exclude padding positions from all outputs
            mask &= labels != pad_token_id
        out["labels"] = labels[mask]

    if sequence_ids is not None:
        sequence_ids = np.asarray(sequence_ids)
        if sequence_ids.shape != (batch_size,):
            raise ValueError(
                f"sequence_ids must have shape (batch,), got {sequence_ids.shape}"
            )
        seq_ids_2d = np.broadcast_to(
            sequence_ids[:, None],
            (batch_size, seq_len),
        )
        out["sequence_ids"] = seq_ids_2d[mask]

    # Apply mask and flatten
    out["confidences"] = confidences[mask]
    out["predictions"] = predictions[mask]
    out["timesteps"] = timesteps[mask]

    return out


@dataclass
class ConfidenceTracker:
    """Incrementally track confidence trajectories over many batches.

    Typical usage:

    .. code-block:: python

        tracker = ConfidenceTracker()
        for batch in dataloader:
            logits = model(batch["input_ids"])          # (B, T, C)
            labels = batch["labels"]                    # (B, T)
            seq_ids = batch.get("sequence_ids", None)   # (B,)

            tracker.update(
                logits,
                labels=labels,
                sequence_ids=seq_ids,
                is_logits=True,
                pad_token_id=pad_token_id,
            )

        tracker.save("results/confidence_trajectories.npz")

    After saving, you can load the arrays via ``np.load`` and compute
    calibration metrics per-timestep (e.g. ECE over tokens at timestep ``t``).
    """

    confidences: list[np.ndarray] = field(default_factory=list)
    predictions: list[np.ndarray] = field(default_factory=list)
    timesteps: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    sequence_ids: list[np.ndarray] = field(default_factory=list)

    def update(
        self,
        logits_or_probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
        *,
        is_logits: bool = True,
        sequence_ids: Optional[np.ndarray] = None,
        pad_token_id: Optional[int] = None,
    ) -> None:
        """Update tracker with a new batch.

        Args:
            logits_or_probs:
                Array of shape (batch, seq_len, n_classes) containing either
                logits or probabilities.
            labels:
                Optional integer labels of shape (batch, seq_len).
            is_logits:
                If True, interpret ``logits_or_probs`` as logits.
            sequence_ids:
                Optional array of shape (batch,) providing an id for each
                sequence in the batch.
            pad_token_id:
                Optional integer id used for padding positions in ``labels``.
        """
        batch_data = extract_confidence_trajectories(
            logits_or_probs=logits_or_probs,
            labels=labels,
            is_logits=is_logits,
            sequence_ids=sequence_ids,
            pad_token_id=pad_token_id,
        )

        self.confidences.append(batch_data["confidences"])
        self.predictions.append(batch_data["predictions"])
        self.timesteps.append(batch_data["timesteps"])

        if "labels" in batch_data:
            self.labels.append(batch_data["labels"])
        if "sequence_ids" in batch_data:
            self.sequence_ids.append(batch_data["sequence_ids"])

    def as_arrays(self) -> Dict[str, np.ndarray]:
        """Return all tracked data concatenated into single arrays."""
        if not self.confidences:
            raise ValueError("No data has been tracked yet.")

        def _concat(parts: Iterable[np.ndarray]) -> np.ndarray:
            return np.concatenate(list(parts), axis=0)

        result: Dict[str, np.ndarray] = {
            "confidences": _concat(self.confidences),
            "predictions": _concat(self.predictions),
            "timesteps": _concat(self.timesteps),
        }

        if self.labels:
            result["labels"] = _concat(self.labels)
        if self.sequence_ids:
            result["sequence_ids"] = _concat(self.sequence_ids)

        return result

    def save(
        self,
        path: Union[str, Path],
        *,
        compressed: bool = True,
        metadata: Optional[Dict[str, ArrayLike]] = None,
    ) -> Path:
        """Save tracked confidence trajectories to disk.

        The data is stored as a NumPy ``.npz`` archive with fields:
        ``confidences``, ``predictions``, ``timesteps``, and optionally
        ``labels`` and ``sequence_ids``. Any additional ``metadata`` entries
        are added as extra arrays.

        Args:
            path:
                Output file path. ``.npz`` will be appended if missing.
            compressed:
                If True (default) use ``np.savez_compressed``, otherwise
                use ``np.savez``.
            metadata:
                Optional mapping of additional arrays to include in the file,
                for example dataset/model identifiers.

        Returns:
            The resolved output path.
        """
        arrays = self.as_arrays()

        if metadata:
            # Shallow copy to avoid mutating user dict
            arrays = {**arrays, **metadata}

        out_path = Path(path)
        if out_path.suffix != ".npz":
            out_path = out_path.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if compressed:
            np.savez_compressed(out_path, **arrays)
        else:
            np.savez(out_path, **arrays)

        return out_path

# Confidence tracking over time