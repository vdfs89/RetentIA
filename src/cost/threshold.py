import numpy as np


def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 100.0,
) -> float:
    """Finds threshold minimizing: (FN * cost_fn) + (FP * cost_fp)."""
    thresholds = np.linspace(0.0, 1.0, 101)
    best_threshold = 0.5
    min_cost = float("inf")

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        cost = (fn * cost_fn) + (fp * cost_fp)

        if cost < min_cost:
            min_cost = cost
            best_threshold = t

    return float(best_threshold)
