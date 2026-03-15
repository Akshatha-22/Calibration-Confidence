"""Reliability diagrams for visualizing model calibration.

A reliability diagram plots confidence (x-axis) against accuracy (y-axis) across
bins of predictions. A perfectly calibrated model follows the diagonal line.
"""

import matplotlib.pyplot as plt
import numpy as np


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: str = None,
    show: bool = True,
):
    """Plot a reliability diagram for model calibration assessment.

    Args:
        probs: Predicted probabilities of shape (n_samples, n_classes).
        labels: True class labels of shape (n_samples,).
        n_bins: Number of bins for confidence levels.
        title: Plot title.
        save_path: Path to save the plot (optional).
        show: Whether to display the plot.

    Returns:
        Dictionary with bin data: {'confidences': [...], 'accuracies': [...], 'counts': [...]}
    """
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probs and labels must have the same number of samples")

    # Get confidence and predictions
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for bin_idx in range(n_bins):
        bin_start = bin_boundaries[bin_idx]
        bin_end = bin_boundaries[bin_idx + 1]

        if bin_idx == n_bins - 1:
            # Last bin includes the upper boundary
            bin_mask = (confidences >= bin_start) & (confidences <= bin_end)
        else:
            bin_mask = (confidences >= bin_start) & (confidences < bin_end)
        if not np.any(bin_mask):
            bin_confidences.append((bin_start + bin_end) / 2)
            bin_accuracies.append(0.0)
            bin_counts.append(0)
            continue

        bin_conf = np.mean(confidences[bin_mask])
        bin_acc = np.mean(predictions[bin_mask] == labels[bin_mask])
        bin_count = np.sum(bin_mask)

        bin_confidences.append(bin_conf)
        bin_accuracies.append(bin_acc)
        bin_counts.append(bin_count)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Reliability curve
    ax.plot(bin_confidences, bin_accuracies, marker='o', linestyle='-', color='blue', label='Model')

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')

    # Bar plot for sample counts (normalized)
    bin_centers = [(bin_boundaries[i] + bin_boundaries[i+1]) / 2 for i in range(n_bins)]
    max_count = max(bin_counts) if bin_counts else 1
    if max_count == 0:
        bar_heights = [0.0] * n_bins
    else:
        bar_heights = [count / max_count * 0.1 for count in bin_counts]  # Scale for visibility
    ax.bar(bin_centers, bar_heights, width=0.05, alpha=0.3, color='red', label='Sample count')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        'confidences': bin_confidences,
        'accuracies': bin_accuracies,
        'counts': bin_counts,
    }
