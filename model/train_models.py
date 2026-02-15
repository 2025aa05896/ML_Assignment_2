from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ARTIFACTS_DIR = ROOT
DATA_DIR = PROJECT_ROOT / "data"

KAGGLE_DATASET_SLUG = "uciml/faulty-steel-plates"
KAGGLE_EXPECTED_FILE = DATA_DIR / "faults.csv"
TARGET_CLASS_COLUMNS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]


def _resolve_dataset_path() -> Path:
    candidates = [
        KAGGLE_EXPECTED_FILE,
        DATA_DIR / "Faults.csv",
        DATA_DIR / "faulty_steel_plates.csv",
        PROJECT_ROOT / "faults.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Kaggle dataset file not found. Download dataset "
        f"'{KAGGLE_DATASET_SLUG}' and place CSV at '{KAGGLE_EXPECTED_FILE}'."
    )


def _load_faulty_steel_plates() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    csv_path = _resolve_dataset_path()
    df = pd.read_csv(csv_path)

    class_cols_present = [c for c in TARGET_CLASS_COLUMNS if c in df.columns]
    if len(class_cols_present) == len(TARGET_CLASS_COLUMNS):
        target_series = df[class_cols_present].idxmax(axis=1)
        # Expected format is one-hot targets; this validates integrity.
        if not (df[class_cols_present].sum(axis=1) == 1).all():
            raise ValueError("Fault class columns are not one-hot encoded per row.")
        feature_df = df.drop(columns=class_cols_present)
    elif "target" in df.columns:
        target_series = df["target"]
        feature_df = df.drop(columns=["target"])
    elif "class" in df.columns:
        target_series = df["class"]
        feature_df = df.drop(columns=["class"])
    else:
        raise ValueError(
            "Could not infer target column(s). Expected one-hot fault columns or a target/class column."
        )

    # Keep numeric feature columns only for model compatibility.
    feature_df = feature_df.select_dtypes(include=[np.number]).copy()

    if feature_df.shape[0] < 500:
        raise ValueError("Dataset has fewer than 500 instances.")
    if feature_df.shape[1] < 12:
        raise ValueError("Dataset has fewer than 12 numeric features.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(target_series.astype(str).to_numpy())
    class_names = label_encoder.classes_

    dataset_info: Dict[str, Any] = {
        "dataset_name": "Kaggle - Faulty Steel Plates",
        "dataset_slug": KAGGLE_DATASET_SLUG,
        "dataset_file": str(csv_path.relative_to(PROJECT_ROOT)),
    }
    return feature_df, y_encoded, class_names, dataset_info


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, float]:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    try:
        if len(classes) == 2:
            score_col = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics["AUC"] = roc_auc_score(y_true, score_col)
        else:
            y_true_bin = label_binarize(y_true, classes=classes)
            metrics["AUC"] = roc_auc_score(
                y_true_bin,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
    except ValueError:
        metrics["AUC"] = float("nan")

    return metrics


def build_models(num_classes: int) -> Dict[str, Tuple[str, Any]]:
    models: Dict[str, Tuple[str, Any]] = {
        "logistic_regression": (
            "Logistic Regression",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=5000, random_state=42)),
                ]
            ),
        ),
        "decision_tree": (
            "Decision Tree",
            DecisionTreeClassifier(random_state=42),
        ),
        "knn": (
            "kNN",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=7)),
                ]
            ),
        ),
        "naive_bayes": (
            "Naive Bayes",
            GaussianNB(),
        ),
        "random_forest": (
            "Random Forest (Ensemble)",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=1,
            ),
        ),
    }

    if HAS_XGBOOST:
        xgb_params: Dict[str, Any] = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": 1,
        }
        if num_classes > 2:
            xgb_params.update({"objective": "multi:softprob", "num_class": num_classes})
        else:
            xgb_params.update({"objective": "binary:logistic"})

        xgb_model = XGBClassifier(**xgb_params)
    else:
        # Fallback keeps the project executable in restricted environments.
        xgb_model = GradientBoostingClassifier(random_state=42)

    models["xgboost"] = ("XGBoost (Ensemble)", xgb_model)
    return models


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, class_names, dataset_info = _load_faulty_steel_plates()
    class_labels = np.array(sorted(np.unique(y)))
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv(ARTIFACTS_DIR / "faults_test_data.csv", index=False)

    models = build_models(num_classes=len(class_labels))
    summary_rows: List[Dict[str, Any]] = []
    model_registry: Dict[str, Dict[str, str]] = {}
    detailed_results: Dict[str, Any] = {}

    for model_key, (model_name, model) in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = label_binarize(y_pred, classes=class_labels)

        metrics = compute_metrics(y_test, y_pred, y_proba, class_labels)
        summary_rows.append({"ML Model Name": model_name, **metrics})

        model_file = ARTIFACTS_DIR / f"{model_key}.joblib"
        joblib.dump(model, model_file)
        model_registry[model_name] = {
            "key": model_key,
            "path": model_file.name,
        }

        detailed_results[model_name] = {
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
        }

    metrics_df = pd.DataFrame(summary_rows).sort_values(by="Accuracy", ascending=False)
    metrics_df = metrics_df[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    metrics_df.to_csv(ARTIFACTS_DIR / "model_comparison_metrics.csv", index=False)

    label_mapping = {
        str(int(label)): str(class_names[int(label)]) for label in class_labels
    }

    metadata = {
        "dataset_name": dataset_info["dataset_name"],
        "dataset_slug": dataset_info["dataset_slug"],
        "dataset_file": dataset_info["dataset_file"],
        "num_instances": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "target_name": "target",
        "feature_columns": feature_columns,
        "class_labels": class_labels.tolist(),
        "class_name_mapping": label_mapping,
        "xgboost_backend": "xgboost" if HAS_XGBOOST else "gradient_boosting_fallback",
        "test_data_file": "faults_test_data.csv",
        "model_metrics_file": "model_comparison_metrics.csv",
        "model_registry": model_registry,
        "detailed_results": detailed_results,
    }

    with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete. Artifacts saved to:", ARTIFACTS_DIR)
    print(
        "Dataset:",
        metadata["dataset_name"],
        "| instances:",
        metadata["num_instances"],
        "| features:",
        metadata["num_features"],
    )
    print("Kaggle dataset slug:", metadata["dataset_slug"])
    print("XGBoost backend:", metadata["xgboost_backend"])


if __name__ == "__main__":
    main()
