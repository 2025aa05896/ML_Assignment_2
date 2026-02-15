from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXPECTED_KAGGLE_SLUG = "uciml/faulty-steel-plates"


def main() -> None:
    with open(ROOT / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("dataset_slug") != EXPECTED_KAGGLE_SLUG:
        raise RuntimeError(
            "Artifacts are not from the configured Kaggle dataset. "
            "Run `python model/train_models.py` after placing `data/faults.csv`."
        )

    test_df = pd.read_csv(ROOT / metadata["test_data_file"])
    feature_columns = metadata["feature_columns"]
    X = test_df[feature_columns]

    for model_name, info in metadata["model_registry"].items():
        model = joblib.load(ROOT / info["path"])
        preds = model.predict(X.head(5))
        print(f"{model_name}: OK, sample predictions -> {preds.tolist()}")

    metrics_df = pd.read_csv(ROOT / metadata["model_metrics_file"])
    expected_cols = ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    missing_cols = [c for c in expected_cols if c not in metrics_df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing metric columns: {missing_cols}")

    print("Metrics file schema: OK")


if __name__ == "__main__":
    main()
