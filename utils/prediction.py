"""Prediction helpers that reproduce the notebook's saved preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd


NUMERICAL_COLUMNS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL_COLUMNS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
VALID_CATEGORIES = {
    "Sex": {"F", "M"},
    "ChestPainType": {"ASY", "ATA", "NAP", "TA"},
    "RestingECG": {"LVH", "Normal", "ST"},
    "ExerciseAngina": {"N", "Y"},
    "ST_Slope": {"Down", "Flat", "Up"},
}


class ModelFilesError(RuntimeError):
    """Raised when saved model artifacts are missing, corrupt, or incompatible."""


def load_artifacts(model_path: str | Path, feature_path: str | Path, scaler_path: str | Path) -> tuple[Any, list[str], Any]:
    """Load the original model, feature ordering, and fitted scaler without fitting."""
    paths = [Path(model_path), Path(feature_path), Path(scaler_path)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ModelFilesError("Required model file(s) are missing: " + ", ".join(missing))
    try:
        model, features, scaler = (joblib.load(path) for path in paths)
    except Exception as exc:
        raise ModelFilesError(f"Could not load a saved model artifact: {exc}") from exc
    if not isinstance(features, list) or not features:
        raise ModelFilesError("feature_columns.pkl must contain a non-empty feature list.")
    if getattr(model, "n_features_in_", len(features)) != len(features):
        raise ModelFilesError("The model's feature count does not match feature_columns.pkl.")
    if getattr(scaler, "n_features_in_", len(NUMERICAL_COLUMNS)) != len(NUMERICAL_COLUMNS):
        raise ModelFilesError("The saved scaler does not match the five numeric input columns.")
    return model, features, scaler


def encode_patient(patient: Mapping[str, Any], feature_columns: list[str], scaler: Any) -> pd.DataFrame:
    """One-hot encode with drop_first=True and transform the exact five numeric columns."""
    missing = [column for column in NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS + ["FastingBS"] if column not in patient]
    if missing:
        raise ValueError("Missing patient field(s): " + ", ".join(missing))

    row: dict[str, Any] = {}
    for column in NUMERICAL_COLUMNS:
        value = float(patient[column])
        if not pd.notna(value):
            raise ValueError(f"{column} must be a finite number.")
        row[column] = value
    fasting = int(patient["FastingBS"])
    if fasting not in (0, 1):
        raise ValueError("FastingBS must be 0 (No) or 1 (Yes).")
    row["FastingBS"] = fasting
    for column in CATEGORICAL_COLUMNS:
        value = str(patient[column])
        if value not in VALID_CATEGORIES[column]:
            raise ValueError(f"Invalid {column} category: {value}")
        row[column] = value

    encoded = pd.get_dummies(pd.DataFrame([row]), columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=int)
    # The training matrix was made from the full dataset, so absent dummy levels
    # need explicit zero columns before selection in the persisted order.
    for column in feature_columns:
        if column not in encoded:
            encoded[column] = 0
    unexpected = [column for column in encoded.columns if column not in feature_columns]
    if unexpected:
        raise ModelFilesError("Unexpected encoded feature(s): " + ", ".join(unexpected))
    encoded = encoded.loc[:, feature_columns].copy()
    encoded.loc[:, NUMERICAL_COLUMNS] = scaler.transform(encoded.loc[:, NUMERICAL_COLUMNS])
    return encoded


def predict_patient(patient: Mapping[str, Any], model: Any, feature_columns: list[str], scaler: Any) -> dict[str, Any]:
    """Return prediction and class probabilities from the original fitted model."""
    features = encode_patient(patient, feature_columns, scaler)
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    return {
        "prediction": prediction,
        "probability_no_disease": float(probabilities[classes.index(0)]),
        "probability_disease": float(probabilities[classes.index(1)]),
        "features": features,
    }

