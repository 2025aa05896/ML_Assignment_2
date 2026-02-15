from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from sklearn.preprocessing import label_binarize

try:
    import streamlit as st
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Streamlit is not installed. Install dependencies and run: streamlit run app.py"
    ) from exc


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
EXPECTED_KAGGLE_SLUG = "uciml/faulty-steel-plates"


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, float]:
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    try:
        if len(classes) == 2:
            score_col = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            result["AUC"] = roc_auc_score(y_true, score_col)
        else:
            y_true_bin = label_binarize(y_true, classes=classes)
            result["AUC"] = roc_auc_score(
                y_true_bin,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
    except ValueError:
        result["AUC"] = float("nan")
    return result


def load_artifacts() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metrics_df = pd.read_csv(MODEL_DIR / metadata["model_metrics_file"])
    test_df = pd.read_csv(MODEL_DIR / metadata["test_data_file"])
    return metadata, metrics_df, test_df


def main() -> None:
    st.set_page_config(page_title="ML Assignment 2 - Classifier Demo", layout="wide")
    st.title("Machine Learning Assignment 2 - Classification Model Demo")

    metadata, metrics_df, default_test_df = load_artifacts()
    model_registry = metadata["model_registry"]
    feature_columns = metadata["feature_columns"]
    target_col = metadata["target_name"]
    class_labels = np.array(metadata["class_labels"])
    class_name_mapping = metadata.get("class_name_mapping", {})
    id_to_name = {int(k): v for k, v in class_name_mapping.items()}
    name_to_id = {v: k for k, v in id_to_name.items()}

    st.write(
        f"Dataset: `{metadata['dataset_name']}` | "
        f"Kaggle: `{metadata.get('dataset_slug', 'N/A')}` | "
        f"Instances: `{metadata['num_instances']}` | "
        f"Features: `{metadata['num_features']}` | "
        f"Classes: `{len(class_labels)}`"
    )

    if metadata.get("dataset_slug") != EXPECTED_KAGGLE_SLUG:
        st.error(
            "Artifacts appear to be from an older/non-Kaggle dataset. "
            "Place `data/faults.csv` and rerun `python model/train_models.py`."
        )
        st.stop()

    if metadata["xgboost_backend"] != "xgboost":
        st.warning(
            "XGBoost package is unavailable in this environment. "
            "A GradientBoosting fallback was used for the XGBoost slot."
        )

    st.subheader("Model Comparison (Default Holdout Test Set)")
    display_df = metrics_df.copy()
    for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        display_df[col] = display_df[col].round(4)
    st.dataframe(display_df, width="stretch")

    st.subheader("Inference")
    uploaded_file = st.file_uploader(
        "Upload test CSV (must include all feature columns; `target` column optional)",
        type=["csv"],
    )

    model_name = st.selectbox(
        "Select model",
        options=metrics_df["ML Model Name"].tolist(),
    )

    model_info = model_registry[model_name]
    model = joblib.load(MODEL_DIR / model_info["path"])

    if uploaded_file is not None:
        eval_df = pd.read_csv(uploaded_file)
        source_name = "Uploaded CSV"
    else:
        eval_df = default_test_df.copy()
        source_name = "Default holdout test set"

    st.caption(f"Evaluation source: {source_name}")

    missing_features = [c for c in feature_columns if c not in eval_df.columns]
    if missing_features:
        st.error(
            "Uploaded dataset is missing required feature columns. "
            f"Missing count: {len(missing_features)}"
        )
        st.stop()

    if st.button("Run Model", type="primary"):
        X_eval = eval_df[feature_columns]
        y_pred = model.predict(X_eval)

        result_df = eval_df.copy()
        pred_series = pd.Series(y_pred, index=result_df.index)
        result_df["prediction"] = pred_series
        if id_to_name:
            mapped_labels = pred_series.map(id_to_name)
            # Avoid array-like fillna values, which can fail on newer pandas versions.
            result_df["prediction_label"] = mapped_labels.combine_first(pred_series.astype(str))
        st.write("Sample predictions")
        st.dataframe(result_df.head(20), width="stretch")

        if target_col in eval_df.columns:
            target_series = eval_df[target_col]
            if target_series.dtype == object and name_to_id:
                mapped = target_series.map(name_to_id)
                if mapped.isna().any():
                    st.error(
                        "Target column contains labels not found in training metadata. "
                        "Use the provided holdout CSV format or encoded target values."
                    )
                    st.stop()
                y_true = mapped.to_numpy()
            else:
                y_true = target_series.to_numpy()

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_eval)
            else:
                y_proba = label_binarize(y_pred, classes=class_labels)

            metric_values = compute_metrics(y_true, y_pred, y_proba, class_labels)

            st.subheader("Evaluation Metrics")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{metric_values['Accuracy']:.4f}")
            c2.metric("AUC", f"{metric_values['AUC']:.4f}")
            c3.metric("Precision", f"{metric_values['Precision']:.4f}")
            c4.metric("Recall", f"{metric_values['Recall']:.4f}")
            c5.metric("F1", f"{metric_values['F1']:.4f}")
            c6.metric("MCC", f"{metric_values['MCC']:.4f}")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=class_labels)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Classification Report")
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), width="stretch")
        else:
            st.info(
                "No `target` column in the uploaded CSV, so only predictions are shown. "
                "Upload labeled test data to compute metrics and confusion matrix."
            )


if __name__ == "__main__":
    main()
