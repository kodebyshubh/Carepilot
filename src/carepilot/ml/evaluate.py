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


def subgroup_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, groups: np.ndarray
) -> list[dict[str, float | int | str | None]]:
    """Recall, precision, false positive rate and AUC per subgroup on the test set.

    Groups are small (tens of patients), so these numbers flag where to look, they do not
    prove a disparity. n and positives are reported so the reader can judge.
    """
    y_true, proba, groups = np.asarray(y_true), np.asarray(proba), np.asarray(groups)
    rows = []
    for g in sorted(set(groups.tolist()), key=str):
        mask = groups == g
        y, p = y_true[mask], proba[mask]
        pred = (p >= threshold).astype(int)
        negatives = (y == 0).sum()
        rows.append({
            "group": str(g),
            "n": int(mask.sum()),
            "positives": int(y.sum()),
            "recall": float(recall_score(y, pred, zero_division=0)) if y.sum() else None,
            "precision": float(precision_score(y, pred, zero_division=0)) if pred.sum() else None,
            "false_positive_rate": float(((pred == 1) & (y == 0)).sum() / negatives) if negatives else None,
            "roc_auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None,
        })  # fmt: skip
    return rows


TARGET_RECALL = 0.85


def select_threshold(
    y_true: np.ndarray, proba: np.ndarray, target_recall: float = TARGET_RECALL
) -> float:
    """Highest threshold whose recall on out-of-fold predictions is at least target_recall.

    Screening logic: fix the sensitivity we refuse to go below (a missed case costs more
    than an extra follow-up test), then take the highest threshold that still meets it,
    which gives the best precision under that constraint. Falls back to max F2 if the
    target is unreachable.
    """
    feasible = [
        t
        for t in THRESHOLD_GRID
        if recall_score(y_true, (proba >= t).astype(int), zero_division=0) >= target_recall
    ]
    if feasible:
        return float(max(feasible))
    scores = [fbeta_score(y_true, (proba >= t).astype(int), beta=2) for t in THRESHOLD_GRID]
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
        y_true,
        (proba >= threshold).astype(int),
        display_labels=["No", "Yes"],
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
