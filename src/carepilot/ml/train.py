"""Train, compare and save risk models for each disease.

Protocol (per disease):
  1. Stratified 80/20 train/test split. The test set is touched exactly once, at the end.
  2. Each candidate (LogReg, RandomForest, XGBoost) is tuned with a small grid using
     5-fold stratified CV on the training set, scored by ROC-AUC. All preprocessing lives
     inside the Pipeline, so imputation/scaling statistics never see validation folds.
  3. Selection: highest mean CV ROC-AUC, except that Logistic Regression wins if it is
     within one standard error of the best (parsimony: simpler, calibrated, interpretable).
  4. Decision threshold: maximize F2 on out-of-fold training predictions (recall-weighted).
  5. Report every candidate on the untouched test set with bootstrap 95% CIs.

Usage: python -m carepilot.ml.train --disease all
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from carepilot.config import MODELS_DIR, SEED
from carepilot.ml import evaluate
from carepilot.ml.datasets import DiseaseSpec, get_spec, load

log = logging.getLogger(__name__)

TEST_SIZE = 0.2
CV_FOLDS = 5
MODEL_LABELS = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}


def build_preprocessor(spec: DiseaseSpec) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = [("num", numeric, list(spec.numeric))]
    if spec.categorical:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical, list(spec.categorical)))
    return ColumnTransformer(transformers, verbose_feature_names_out=True)


def candidates() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Estimator plus a deliberately small grid (small data: big grids overfit the CV)."""
    return {
        "logreg": (
            LogisticRegression(max_iter=2000),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "rf": (
            RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=1),
            {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [1, 5]},
        ),
        "xgb": (
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=1,
            ),
            {"clf__max_depth": [2, 3, 4], "clf__min_child_weight": [1, 5]},
        ),
    }


@dataclass(frozen=True)
class CvResult:
    name: str
    best_params: dict[str, Any]
    cv_auc_mean: float
    cv_auc_std: float
    estimator: Pipeline

    @property
    def cv_auc_se(self) -> float:
        return self.cv_auc_std / np.sqrt(CV_FOLDS)


def tune(spec: DiseaseSpec, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> list[CvResult]:
    results = []
    for name, (est, grid) in candidates().items():
        pipe = Pipeline([("prep", build_preprocessor(spec)), ("clf", est)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        search.fit(X, y)
        i = search.best_index_
        results.append(
            CvResult(
                name=name,
                best_params={k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
                cv_auc_mean=float(search.cv_results_["mean_test_score"][i]),
                cv_auc_std=float(search.cv_results_["std_test_score"][i]),
                estimator=search.best_estimator_,
            )
        )
        log.info(
            "  %-7s CV AUC %.3f +/- %.3f  %s",
            name,
            results[-1].cv_auc_mean,
            results[-1].cv_auc_std,
            results[-1].best_params,
        )
    return results


def select_model(results: list[CvResult]) -> tuple[CvResult, str]:
    """Best mean CV AUC, but prefer Logistic Regression within one SE of the best."""
    best = max(results, key=lambda r: r.cv_auc_mean)
    logreg = next(r for r in results if r.name == "logreg")
    if best.name != "logreg" and logreg.cv_auc_mean >= best.cv_auc_mean - best.cv_auc_se:
        return logreg, (
            f"{MODEL_LABELS[best.name]} had the top CV ROC-AUC ({best.cv_auc_mean:.3f}), but "
            f"Logistic Regression ({logreg.cv_auc_mean:.3f}) is within one standard error "
            f"({best.cv_auc_se:.3f}), so the simpler, better-calibrated, directly interpretable "
            "model is preferred (one-standard-error rule)."
        )
    return best, (
        f"{MODEL_LABELS[best.name]} had the highest mean CV ROC-AUC ({best.cv_auc_mean:.3f}) "
        "and no simpler model was within one standard error."
    )


def feature_groups(pipe: Pipeline, spec: DiseaseSpec) -> dict[str, list[int]]:
    """Map each original feature to the transformed column indices derived from it.

    Lets per-column SHAP values (one-hot levels, missing indicators) be summed back to
    the feature a patient actually entered.
    """
    names = pipe.named_steps["prep"].get_feature_names_out()
    groups: dict[str, list[int]] = {f: [] for f in spec.feature_order}
    for idx, full in enumerate(names):
        raw = full.split("__", 1)[1].removeprefix("missingindicator_")
        owner = next(
            f
            for f in sorted(spec.feature_order, key=len, reverse=True)
            if raw == f or raw.startswith(f + "_")
        )
        groups[owner].append(idx)
    return groups


def subgroups(X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Slices for the fairness check. Pima is all women, so only age applies there."""
    age_col = "Age" if "Age" in X else "age"
    out = {
        "age band": pd.cut(X[age_col], [0, 39, 54, 200], labels=["<40", "40-54", "55+"])
        .astype(str)
        .to_numpy()
    }
    if "sex" in X:
        out["sex"] = X["sex"].map({0: "Female", 1: "Male"}).to_numpy()
    return out


def train_disease(disease: str) -> dict[str, Any]:
    spec = get_spec(disease)
    X, y = load(disease)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    log.info("[%s] train %d / test %d, positive rate %.3f", disease, len(X_tr), len(X_te), y.mean())

    results = tune(spec, X_tr, y_tr, cv)
    chosen, why = select_model(results)

    oof = cross_val_predict(clone(chosen.estimator), X_tr, y_tr, cv=cv, method="predict_proba")[
        :, 1
    ]
    threshold = evaluate.select_threshold(y_tr.to_numpy(), oof)

    out_dir = MODELS_DIR / disease
    out_dir.mkdir(parents=True, exist_ok=True)
    test_probas = {MODEL_LABELS[r.name]: r.estimator.predict_proba(X_te)[:, 1] for r in results}
    y_te_np = y_te.to_numpy()
    comparison = {
        MODEL_LABELS[r.name]: {
            "cv_auc_mean": r.cv_auc_mean,
            "cv_auc_std": r.cv_auc_std,
            "best_params": r.best_params,
            "test_at_0.5": evaluate.bootstrap_metrics(
                y_te_np, test_probas[MODEL_LABELS[r.name]], 0.5
            ),
            "test_at_tuned": evaluate.bootstrap_metrics(
                y_te_np, test_probas[MODEL_LABELS[r.name]], threshold
            ),
        }
        for r in results
    }

    chosen_label = MODEL_LABELS[chosen.name]
    evaluate.plot_roc(y_te_np, test_probas, f"ROC, {disease} (test set)", out_dir / "roc.png")
    evaluate.plot_calibration(
        y_te_np, test_probas, f"Calibration, {disease} (test set)", out_dir / "calibration.png"
    )
    evaluate.plot_confusion(
        y_te_np,
        test_probas[chosen_label],
        threshold,
        f"{chosen_label} @ {threshold:.2f}",
        out_dir / "confusion.png",
    )

    background = chosen.estimator.named_steps["prep"].transform(
        X_tr.sample(n=min(100, len(X_tr)), random_state=SEED)
    )
    joblib.dump(
        {"pipeline": chosen.estimator, "shap_background": background}, out_dir / "model.joblib"
    )

    metadata = {
        "disease": disease,
        "display_name": spec.display_name,
        "selected_model": chosen_label,
        "selected_key": chosen.name,
        "selection_reason": why,
        "threshold": threshold,
        "threshold_rule": (
            f"highest threshold with recall >= {evaluate.TARGET_RECALL:.0%} on 5-fold "
            "out-of-fold training predictions (screening sensitivity target)"
        ),
        "feature_order": spec.feature_order,
        "feature_groups": feature_groups(chosen.estimator, spec),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "positive_rate": float(y.mean()),
        "fairness": {
            name: evaluate.subgroup_metrics(y_te_np, test_probas[chosen_label], threshold, groups)
            for name, groups in subgroups(X_te).items()
        },
        "trained_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    log.info("[%s] selected %s, threshold %.2f -> %s", disease, chosen_label, threshold, out_dir)
    return {"metadata": metadata, "comparison": comparison}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="all", choices=["all", "diabetes", "heart"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    diseases = ["diabetes", "heart"] if args.disease == "all" else [args.disease]
    for d in diseases:
        train_disease(d)
    from carepilot.ml.report import write_report  # local import: report reads saved artifacts

    write_report()


if __name__ == "__main__":
    main()
