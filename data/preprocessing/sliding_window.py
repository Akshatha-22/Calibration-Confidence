import numpy as np


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    horizon: int = 1,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window samples for sequence data.

    This is useful when applying non-sequential models (e.g., MLP) to time-series
    or sequential data by converting it into a supervised learning problem.

    Args:
        data: Array of shape (N, ...) where N is sequence length.
        window_size: Number of past points to use as input.
        horizon: How many steps ahead to predict (default 1).
        step: Window stride (default 1).

    Returns:
        X: np.ndarray of shape (num_windows, window_size, ...)
        y: np.ndarray of shape (num_windows, ...)
    """

    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")

    data = np.asarray(data)
    n = len(data)
    if n < window_size + horizon:
        return np.empty((0, window_size) + data.shape[1:]), np.empty((0,) + data.shape[1:])

    X_windows = []
    y_targets = []

    for start in range(0, n - window_size - horizon + 1, step):
        end = start + window_size
        target_idx = end + horizon - 1
        X_windows.append(data[start:end])
        y_targets.append(data[target_idx])

    X = np.stack(X_windows)
    y = np.stack(y_targets)
    return X, y


if __name__ == "__main__":
    # Demo: simulate a simple sequence
    seq = np.arange(1, 21)  # 1..20
    X, y = create_sliding_windows(seq, window_size=5, horizon=1)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First X:\n", X[0])
    print("First y:\n", y[0])
