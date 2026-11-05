# CardioPredict AI — Heart Disease Prediction

CardioPredict AI is a responsive Streamlit dashboard for exploring heart disease risk estimates from the project's **existing trained Logistic Regression model**. It uses the saved `heart_disease_model.pkl`, `feature_columns.pkl`, and `scaler.pkl` without retraining or modifying them.

> **For educational and informational purposes only. This application does not provide a medical diagnosis or replace advice from a qualified healthcare professional.**

## Features

- Dashboard with model and dataset overview.
- Patient form for all 11 original input variables.
- Prediction probabilities from the saved classifier.
- Patient input summary and downloadable PDF report.
- Model analytics, confusion matrix, dataset breakdown, and feature descriptions.
- Streamlit caching for model artifacts and data.
- Graceful messages for missing or unreadable model files.

## Model and preprocessing

The original notebook (`heart_disease_pred_model.ipynb`) trained a `LogisticRegression` model using a 33% test split with `random_state=42`. It reports accuracy **0.8679868 (86.8%)** and confusion matrix `[[111, 12], [28, 152]]`, with labels ordered `0` (no heart disease) and `1` (heart disease).

Input preprocessing follows the notebook:

1. Categorical fields are encoded with pandas `get_dummies(drop_first=True)`.
2. Encoded columns are aligned to the exact order stored in `feature_columns.pkl`.
3. The persisted `scaler.pkl` transforms only `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, and `Oldpeak`.
4. The existing classifier returns the prediction and class probabilities.

Precision, recall, F1, and specificity shown in Model Analytics are calculated from the notebook's reported confusion matrix. No new model results are claimed.

## Project structure

```text
heart_disease_prediction_ml-model/
├── app.py
├── requirements.txt
├── README.md
├── heart_disease_model.pkl       # original saved model, unchanged
├── feature_columns.pkl          # original feature order, unchanged
├── scaler.pkl                   # original fitted scaler, unchanged
├── heart.csv
├── heart_disease_pred_model.ipynb
└── utils/
    ├── __init__.py
    ├── prediction.py
    └── report.py
```

## Run locally

Python 3.10–3.12 is recommended for broad compatibility with the pinned dependency ranges.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Keep the three `.pkl` files beside `app.py`. `heart.csv` is needed for the dataset overview and charts; the prediction form can still work if the CSV is unavailable.

## Deploy to Streamlit Community Cloud

1. Push this project and its three original `.pkl` artifacts to a GitHub repository. Confirm that the artifacts are available to the deployment (repository size limits apply).
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository and select `app.py` as the entry point.
3. Select a supported Python version (3.11 recommended) and deploy. Community Cloud installs `requirements.txt` automatically.
4. Open the deployed app and confirm that the dashboard loads and a sample form submission produces a result.

If your repository uses Git LFS for model artifacts, ensure the deployment environment can retrieve the actual files rather than LFS pointer text.

## Data fields

The 11 raw inputs are `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, and `ST_Slope`. Category values match those present in the supplied `heart.csv`.

