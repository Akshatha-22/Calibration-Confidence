"""Expected Calibration Error (ECE) for model calibration assessment.

ECE measures the difference between a model's predicted confidence and its actual
accuracy. Lower ECE indicates better calibrated predictions.

For multi-class classification, ECE bins predictions by their maximum probability
(confidence) and computes the weighted average of |accuracy - confidence| per bin.
"""

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities of shape (n_samples, n_classes).
        labels: True class labels of shape (n_samples,).
        n_bins: Number of bins to divide confidence into.

    Returns:
        ECE value (float between 0 and 1).
    """
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probs and labels must have the same number of samples")

    # Get confidence (max probability) and predicted class
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for bin_idx in range(n_bins):
        bin_start = bin_boundaries[bin_idx]
        bin_end = bin_boundaries[bin_idx + 1]

        # Find samples in this bin
        bin_mask = (confidences >= bin_start) & (confidences < bin_end)
        if not np.any(bin_mask):
            continue

        bin_size = np.sum(bin_mask)
        bin_conf = np.mean(confidences[bin_mask])
        bin_acc = np.mean(predictions[bin_mask] == labels[bin_mask])

        ece += (bin_size / len(labels)) * abs(bin_acc - bin_conf)

    return ece


def regression_calibration_error(
    preds: np.ndarray, targets: np.ndarray, n_bins: int = 10
) -> float:
    """Regression calibration: bin by predicted value, then |mean_pred - mean_target| per bin.

    Lower is better (predictions match targets within bins).
    """
    preds = np.asarray(preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if preds.shape[0] != targets.shape[0]:
        raise ValueError("preds and targets must have the same number of samples")
    if preds.shape[0] == 0:
        return float("nan")
    lo, hi = float(np.min(preds)), float(np.max(preds))
    if hi <= lo:
        return 0.0
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (preds >= bin_edges[i]) & (preds <= bin_edges[i + 1])
        if not np.any(mask):
            continue
        n = np.sum(mask)
        mean_pred = np.mean(preds[mask])
        mean_tgt = np.mean(targets[mask])
        err += (n / len(preds)) * abs(mean_pred - mean_tgt)
    return float(err)


# Alias for convenience
ece = expected_calibration_error
