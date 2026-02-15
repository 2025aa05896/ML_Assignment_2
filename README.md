# Machine Learning Assignment 2

## a. Problem statement
Build an end-to-end multi-class classification workflow on a Kaggle dataset using six required classifiers. Compare all models with Accuracy, AUC, Precision, Recall, F1, and MCC. Expose the pipeline using a Streamlit app with CSV upload, model selection, metrics view, and confusion matrix/classification report.

## b. Dataset description
- Dataset: **Faulty Steel Plates**
- Kaggle slug: `uciml/faulty-steel-plates`
- Type: Multi-class classification (fault type prediction)
- Size: 1941 rows, 27 numeric features (meets assignment constraints: >500 rows, >12 features)
- Why this dataset: less common than typical student choices while still structured and deployment-friendly.

## c. Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN) Classifier
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison table (required metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7275 | 0.9066 | 0.7329 | 0.7275 | 0.7277 | 0.6487 |
| Decision Tree | 0.7455 | 0.8313 | 0.7492 | 0.7455 | 0.7452 | 0.6733 |
| kNN | 0.7429 | 0.8986 | 0.7414 | 0.7429 | 0.7388 | 0.6732 |
| Naive Bayes | 0.4524 | 0.7955 | 0.5741 | 0.4524 | 0.4003 | 0.3789 |
| Random Forest (Ensemble) | 0.8021 | 0.9468 | 0.8087 | 0.8021 | 0.8016 | 0.7434 |
| XGBoost (Ensemble) | 0.8046 | 0.9517 | 0.8129 | 0.8046 | 0.8068 | 0.7474 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good baseline with balanced weighted scores, but underperforms tree ensembles on non-linear fault boundaries. |
| Decision Tree | Interpretable and reasonably strong recall, but single-tree variance limits overall generalization. |
| kNN | Comparable to Decision Tree, but performance is sensitive to local geometry and class imbalance. |
| Naive Bayes | Fastest model but weakest quality due to strong independence assumptions not fitting this dataset. |
| Random Forest (Ensemble) | Strong robust performance with high AUC and stable class-wise behavior across fault types. |
| XGBoost (Ensemble) | Best overall performer (highest Accuracy, AUC, F1, MCC), capturing complex feature interactions effectively. |

## Repository structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   `-- faults.csv  # Kaggle dataset CSV (download manually)
`-- model/
    |-- train_models.py
    |-- smoke_test.py
    |-- metadata.json
    |-- model_comparison_metrics.csv
    |-- faults_test_data.csv
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- random_forest.joblib
    `-- xgboost.joblib
```

## How to run

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Download Kaggle dataset `uciml/faulty-steel-plates` and place CSV at:

```text
data/faults.csv
```

3. Train models and generate artifacts:

```bash
python model/train_models.py
```

4. Optional smoke test:

```bash
python model/smoke_test.py
```

5. Run Streamlit app:

```bash
streamlit run app.py
```

## Streamlit features implemented
- CSV dataset upload option (test data)
- Model selection dropdown
- Evaluation metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix
- Classification report

## Required links (fill before final submission PDF)
- GitHub Repository: `https://github.com/2025aa05896/ML_Assignment_2`
- Streamlit App: `https://mlassignment2-hu8nyurwpukpaygtruuqna.streamlit.app`
- BITS Lab screenshot: `<attach in submission PDF>`
