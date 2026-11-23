import numpy as np

from carepilot.ml.evaluate import bootstrap_metrics, point_metrics, select_threshold


def test_point_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = point_metrics(y, p, 0.5)
    assert m["accuracy"] == m["recall"] == m["precision"] == m["roc_auc"] == 1.0


def test_threshold_meets_recall_target_and_is_highest_such():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    t = select_threshold(y, p, target_recall=0.85)
    recall_at = lambda thr: ((p >= thr) & (y == 1)).sum() / (y == 1).sum()  # noqa: E731
    assert recall_at(t) >= 0.85
    assert recall_at(round(t + 0.01, 2)) < 0.85


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.4 + rng.random(200) * 0.6, 0, 1)
    out = bootstrap_metrics(y, p, 0.5)
    for m in out.values():
        assert m["ci_low"] <= m["value"] <= m["ci_high"]


def test_subgroup_metrics_per_group():
    from carepilot.ml.evaluate import subgroup_metrics

    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.1])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    rows = {r["group"]: r for r in subgroup_metrics(y, p, 0.5, groups)}
    assert rows["a"]["n"] == 3 and rows["a"]["recall"] == 1.0
    assert rows["a"]["false_positive_rate"] == 1.0
    assert rows["b"]["recall"] == 0.0 and rows["b"]["precision"] is None
