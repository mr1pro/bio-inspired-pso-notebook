"""Classification metrics for a binary problem, written with NumPy only.

Class 1 is treated as the positive class.
"""

from typing import Dict

import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return the four cells of the confusion matrix as tp, tn, fp, fn."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that are correct."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Accuracy, precision, recall and F1 together with the confusion counts.

    precision = tp / (tp + fp)   of the notes flagged as class 1, how many were
    recall    = tp / (tp + fn)   of the true class-1 notes, how many were found
    F1        = harmonic mean of precision and recall

    A ratio with a zero denominator is reported as 0.
    """
    counts = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **counts,
    }
