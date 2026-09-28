# ❤️ Heart Disease Prediction (ML Model)

A machine learning project that predicts whether a patient is likely to have heart disease, based on 11 common clinical measurements such as age, blood pressure, cholesterol and ECG results.

The model is a **Logistic Regression** classifier trained on 918 patient records. It reaches **~86.8% accuracy** on unseen test data.

> ⚠️ **Disclaimer:** This is an educational project. It is **not** a medical device and must not be used for real diagnosis or treatment decisions. Always consult a qualified doctor.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Project Structure](#-project-structure)
3. [Dataset](#-dataset)
4. [How It Works](#-how-it-works)
5. [Results](#-results)
6. [Installation](#-installation)
7. [How to Use](#-how-to-use)
8. [Important Notes](#-important-notes)
9. [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

| | |
|---|---|
| **Goal** | Predict `HeartDisease` (1 = disease, 0 = no disease) |
| **Problem type** | Binary classification |
| **Algorithm** | Logistic Regression (scikit-learn) |
| **Input features** | 11 raw features, which become 15 after encoding |
| **Test accuracy** | 86.80% |
| **Language** | Python 3 (Jupyter Notebook) |

---

## 📁 Project Structure

```
heart_disease_prediction_ml-model/
│
├── heart.csv                       # Dataset (918 rows, 12 columns)
├── heart_disease_pred_model.ipynb  # Full notebook: EDA → cleaning → training → prediction
├── heart_disease_model.pkl         # Saved, trained Logistic Regression model
├── feature_columns.pkl             # Ordered list of the 15 features the model expects
├── scaler.pkl                      # Fitted StandardScaler for the numeric features
└── README.md
```

---

## 📊 Dataset

**File:** `heart.csv` has 918 patients, 11 input features and 1 target. There are no missing values and no duplicate rows.

The classes are fairly balanced: **508** patients with heart disease and **410** without.

| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | Age of the patient (years) |
| `Sex` | Categorical | `M` = male, `F` = female |
| `ChestPainType` | Categorical | `TA` typical angina, `ATA` atypical angina, `NAP` non-anginal pain, `ASY` asymptomatic |
| `RestingBP` | Numeric | Resting blood pressure (mm Hg) |
| `Cholesterol` | Numeric | Serum cholesterol (mg/dl) |
| `FastingBS` | Binary | Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) |
| `RestingECG` | Categorical | `Normal`, `ST` (ST-T wave abnormality), `LVH` (left ventricular hypertrophy) |
| `MaxHR` | Numeric | Maximum heart rate achieved |
| `ExerciseAngina` | Categorical | Exercise-induced angina (`Y` / `N`) |
| `Oldpeak` | Numeric | ST depression induced by exercise relative to rest |
| `ST_Slope` | Categorical | Slope of the peak-exercise ST segment: `Up`, `Flat`, `Down` |
| **`HeartDisease`** | **Target** | **1 = heart disease, 0 = normal** |

---

## ⚙️ How It Works

### 1. Exploratory Data Analysis (EDA)
- Checked shape, data types, summary statistics, duplicates and null values.
- Plotted the distributions of `Age`, `RestingBP`, `Cholesterol` and `MaxHR`.
- Explored relationships with count plots, a box plot, a violin plot and a correlation heatmap.

### 2. Data Cleaning
Some columns contained impossible values of `0`, which really mean missing data:
- **`Cholesterol`**: 172 rows had `0`. They were replaced with the mean of the non-zero values.
- **`RestingBP`**: 1 row had `0`. It was replaced with the mean of the non-zero values.

### 3. Preprocessing
- **One-hot encoding** of the categorical columns with `pd.get_dummies(drop_first=True)`, giving **15 model features**.
- **Feature scaling** of the numeric columns (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) with `StandardScaler`.

### 4. Model Training
- **Train/test split:** 67% training and 33% testing (`random_state=42`).
- **Model:** `LogisticRegression()` with default settings.

### 5. Saving the Model
Three objects are saved with `joblib`, so the model can be reused without retraining:
- `heart_disease_model.pkl`, the trained model
- `feature_columns.pkl`, the feature names in the exact training order
- `scaler.pkl`, the **fitted** scaler used on the numeric columns

### 6. Interactive Prediction
The last notebook cell asks for a patient's details one by one. It scales the numeric values with the saved scaler, then prints the prediction and the probability of each class.

### The 15 model features
```
Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
Sex_M,
ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA,
RestingECG_Normal, RestingECG_ST,
ExerciseAngina_Y,
ST_Slope_Flat, ST_Slope_Up
```

> Because of `drop_first=True`, one category in each group is the baseline. Setting all the dummy columns of a group to 0 means that baseline: `Sex = F`, `ChestPainType = ASY`, `RestingECG = LVH`, `ExerciseAngina = N`, `ST_Slope = Down`.

---

## 📈 Results

Evaluated on the 303 test patients:

| Metric | Value |
|---|---|
| **Accuracy** | **86.80%** |
| Precision (disease) | 92.7% |
| Recall / Sensitivity (disease) | 84.4% |
| Specificity (healthy) | 90.2% |
| F1-score (disease) | 88.4% |

**Confusion matrix**

|  | Predicted: No Disease | Predicted: Disease |
|---|:---:|:---:|
| **Actual: No Disease** | 111 (TN) | 12 (FP) |
| **Actual: Disease** | 28 (FN) | 152 (TP) |

The model catches most disease cases, but 28 patients with the disease were missed (false negatives). In a medical setting a missed case is the more serious kind of error, so improving recall would be a priority.

---

## 🛠 Installation

**Requirements:** Python 3.8+ and these libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

The notebook also installs an optional EDA helper (`sheryanalysis==0.1.0`) in one cell. It is only used for a quick data summary.

> Load the `.pkl` files with the same scikit-learn version you trained with. A different version can cause warnings or, in rare cases, wrong results.

**Clone and run:**

```bash
git clone https://github.com/Arnav-Rastogi/heart_disease_prediction_ml-model.git
cd heart_disease_prediction_ml-model
jupyter notebook heart_disease_pred_model.ipynb
```

---

## 🚀 How to Use

### Option A: Run the notebook
Open `heart_disease_pred_model.ipynb` and run all cells from top to bottom. The final cell prompts you for patient details and prints the result:

```
===================================
     HEART DISEASE PREDICTION
===================================
Enter the patient's information:

Age: ...
Resting Blood Pressure: ...
...
Prediction: 1
Result: Heart Disease
Probability of Heart Disease: xx.xx%
```

### Option B: Load the saved model in your own code

```python
import joblib
import pandas as pd

model    = joblib.load("heart_disease_model.pkl")
features = joblib.load("feature_columns.pkl")
scaler   = joblib.load("scaler.pkl")

patient = pd.DataFrame([{
    "Age": 54, "RestingBP": 130, "Cholesterol": 240, "FastingBS": 0,
    "MaxHR": 150, "Oldpeak": 1.0,
    "Sex_M": 1,
    "ChestPainType_ATA": 0, "ChestPainType_NAP": 0, "ChestPainType_TA": 0,
    "RestingECG_Normal": 1, "RestingECG_ST": 0,
    "ExerciseAngina_Y": 0,
    "ST_Slope_Flat": 1, "ST_Slope_Up": 0,
}])[features]          # keep the exact training column order

# Scale the numeric columns, exactly as during training
numerical_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
patient[numerical_cols] = scaler.transform(patient[numerical_cols])

prediction  = model.predict(patient)[0]
probability = model.predict_proba(patient)[0]

print("Prediction:", "Heart Disease" if prediction == 1 else "No Heart Disease")
print(f"Probability of Heart Disease: {probability[1] * 100:.2f}%")
```

---

## 📌 Important Notes

- **Always scale the numeric inputs before predicting.** The model was trained on standardized values for `Age`, `RestingBP`, `Cholesterol`, `MaxHR` and `Oldpeak`. Passing raw values gives unreliable and overconfident results.
- **The scaler must be fitted before it is saved.** `scaler.pkl` has to be the scaler that was fitted on the training data. A newly created `StandardScaler()` has no mean or standard deviation stored, so it cannot transform anything. You can check with `hasattr(joblib.load("scaler.pkl"), "mean_")`, which should be `True`.
- **Small data-leakage caveat.** Mean imputation and scaling were done on the full dataset before the train/test split, so a little test information leaks into training. The reported accuracy may be slightly optimistic. Splitting first and then computing means and scaling from the training set only would be cleaner.

---

## 🔮 Future Improvements

- Split the data before imputing and scaling, to remove the small leakage.
- Compare other models (Random Forest, XGBoost, SVM, KNN) and tune hyperparameters.
- Use cross-validation for more reliable performance estimates.
- Report ROC-AUC and focus on **recall**, since missing a sick patient is costly.
- Add a `predict.py` script that handles scaling automatically.
- Build a simple web app (Streamlit or Flask) so anyone can enter patient data through a form.
- Try median or model-based imputation for the missing cholesterol values.

---

## 📜 Author & License

- **Author:** [Arnav Rastogi](https://github.com/Arnav-Rastogi)
- **License:** _Add a license (e.g. MIT)_

If you found this project useful, please ⭐ the repository!
