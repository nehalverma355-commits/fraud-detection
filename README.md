# Fraud Detection App

A machine learning web app that predicts whether a financial transaction is fraudulent, built on the PaySim mobile money simulation dataset.

**Live demo:** https://fraud-detection-app-100.streamlit.app/

## Overview

This project trains and compares multiple classification models to detect fraudulent transactions in a highly imbalanced dataset (~0.13% fraud rate), then deploys the best-performing model as an interactive Streamlit app.

## Dataset

- **Source:** (https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset) (Kaggle)
- 6.3M+ transactions, 11 original columns
- Target: `isFraud` (binary)

## Approach

1. **EDA** — explored transaction types, balance distributions, and class imbalance.
2. **Feature engineering:**
   - `balanceDiffOrig` / `balanceDiffDest` — raw balance changes
   - `errorBalanceOrig` / `errorBalanceDest` — balance inconsistency signals (amount vs. actual balance change)
   - `hourOfDay` — extracted from the `step` time index
3. **Imbalance handling:** compared `class_weight="balanced"` vs. SMOTE oversampling using stratified cross-validation.
4. **Model comparison:** Logistic Regression, Random Forest, and XGBoost, evaluated on PR-AUC (more meaningful than accuracy on imbalanced data).
5. **Deployment:** best model saved with `joblib` and served through a Streamlit interface.

## Results

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.577 | 0.991 |
| Random Forest | 0.997 | 0.999 |
| XGBoost | 0.997 | 0.999 |

XGBoost was selected as the final model.

## Running locally

```bash
git clone https://github.com/sameekshasingh-exe/fraud-detection-app
cd fraud-detection-app
pip install -r requirements.txt
streamlit run fraud_detection.py
```

## Files

- `anlaysis_model.ipynb` — full EDA, feature engineering, and model training notebook
- `fraud_detection.py` — Streamlit app
- `fraud_detection_pipeline.pkl` — trained model pipeline
- `requirements.txt` — dependencies

## Tech stack

Python, pandas, scikit-learn, XGBoost, imbalanced-learn, Streamlit
