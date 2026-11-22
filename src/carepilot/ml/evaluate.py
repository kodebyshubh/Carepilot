"""Metrics with bootstrap confidence intervals, threshold selection and diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: CI, Docker, Streamlit worker threads
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "brier")
N_BOOTSTRAP = 1000
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def point_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def bootstrap_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Point estimate plus percentile 95% CI from resampling the test set with replacement.

    With a test set of ~60-150 rows the CI is wide; reporting it is the honest view of
    how much a single split can be trusted.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:  # AUC undefined on single-class resample
            continue
        for k, v in point_metrics(y_true[idx], proba[idx], threshold).items():
            samples[k].append(v)
    point = point_metrics(y_true, proba, threshold)
    return {
        m: {
            "value": float(point[m]),
            "ci_low": float(np.percentile(samples[m], 2.5)),
            "ci_high": float(np.percentile(samples[m], 97.5)),
        }
        for m in METRIC_NAMES
    }


def select_threshold(y_true: np.ndarray, proba: np.ndarray, beta: float = 2.0) -> float:
    """Threshold maximizing F-beta on out-of-fold predictions.

    beta=2 weights recall twice as much as precision: in screening, a missed case
    (false negative) costs more than an unnecessary follow-up test (false positive).
    """
    scores = [fbeta_score(y_true, (proba >= t).astype(int), beta=beta) for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmax(scores))])


def plot_roc(y_true: np.ndarray, probas: dict[str, np.ndarray], title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_calibration(
    y_true: np.ndarray, probas: dict[str, np.ndarray], title: str, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in probas.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=8, strategy="quantile")
        brier = brier_score_loss(y_true, p)
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{name} (Brier {brier:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfectly calibrated")
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency", title=title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_confusion(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, title: str, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, (proba >= threshold).astype(int), display_labels=["No", "Yes"],
        ax=ax, colorbar=False, cmap="Blues",
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
