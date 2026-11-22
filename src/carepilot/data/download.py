"""Download the two public tabular datasets, add headers, write CSVs and a sha256 manifest.

Usage: python -m carepilot.data.download
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import pandas as pd
import requests

from carepilot.config import RAW_DIR

log = logging.getLogger(__name__)
TIMEOUT_S = 30


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str
    columns: tuple[str, ...]
    na_values: tuple[str, ...]
    license: str
    citation: str


SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        name="diabetes",
        url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        columns=(
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Outcome",
        ),
        na_values=(),
        license="Public domain origin (NIDDK); redistributed as CC0 on Kaggle",
        citation=(
            "Smith et al. (1988). Using the ADAP learning algorithm to forecast the onset "
            "of diabetes mellitus. Proc. Symp. Computer Applications in Medical Care."
        ),
    ),
    DatasetSource(
        name="heart",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        columns=(
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "num",
        ),
        na_values=("?",),
        license="CC BY 4.0 (UCI Machine Learning Repository)",
        citation=(
            "Janosi, Steinbrunn, Pfisterer, Detrano (1989). Heart Disease [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X"
        ),
    ),
)


def fetch_dataset(src: DatasetSource) -> tuple[pd.DataFrame, str]:
    """Fetch raw bytes, return parsed frame and sha256 of the raw download."""
    resp = requests.get(src.url, timeout=TIMEOUT_S)
    resp.raise_for_status()
    digest = hashlib.sha256(resp.content).hexdigest()
    df = pd.read_csv(
        io.BytesIO(resp.content),
        header=None,
        names=list(src.columns),
        na_values=list(src.na_values),
    )
    return df, digest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for src in SOURCES:
        df, digest = fetch_dataset(src)
        out = RAW_DIR / f"{src.name}.csv"
        df.to_csv(out, index=False)
        manifest.append(
            {
                **asdict(src),
                "file": out.name,
                "rows": len(df),
                "cols": df.shape[1],
                "sha256_raw": digest,
                "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        log.info("%s: %d rows x %d cols -> %s", src.name, len(df), df.shape[1], out)
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
