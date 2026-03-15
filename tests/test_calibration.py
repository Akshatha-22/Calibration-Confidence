import os
import sys

# Ensure repo root is on sys.path when running this script directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from calibration.ece import ece
from calibration.reliability import reliability_diagram


def test_ece_perfect_calibration():
    # Perfect calibration: probs match labels exactly
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1, 0, 1])
    assert np.isclose(ece(probs, labels), 0.0)


def test_ece_miscalibration():
    # Miscalibration: high confidence but wrong predictions
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    labels = np.array([1, 0, 1, 0])  # All wrong
    ece_val = ece(probs, labels)
    assert ece_val > 0.0  # Should be positive for miscalibration


def test_ece_shape_mismatch():
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    labels = np.array([0])  # Wrong length
    try:
        ece(probs, labels)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_reliability_diagram_perfect():
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive

    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    result = reliability_diagram(probs, labels, show=False)
    assert len(result['confidences']) == 10
    assert result['accuracies'][-1] == 1.0  # Last bin should be accurate
    assert result['counts'][-1] == 2


def test_reliability_diagram_shape_mismatch():
    probs = np.array([[0.5, 0.5]])
    labels = np.array([0, 1])  # Wrong length
    try:
        reliability_diagram(probs, labels, show=False)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
