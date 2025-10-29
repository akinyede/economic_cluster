from typing import Sequence


def compute_mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Absolute Percentage Error in percent (0-100). Ignores zero true values."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be same-length non-empty sequences")
    total = 0.0
    count = 0
    for t, p in zip(y_true, y_pred):
        t = float(t)
        p = float(p)
        if t == 0:
            continue
        total += abs((p - t) / t)
        count += 1
    if count == 0:
        return 0.0
    return (total / count) * 100.0

