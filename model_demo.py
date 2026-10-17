"""CSV-based Random Forest workflow for a non-clinical teaching exercise."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def train_kyphosis_model(data: pd.DataFrame, target_column="Kyphosis", seed=7):
    if target_column not in data.columns:
        raise ValueError(f"CSV must include target column {target_column!r}")
    feature_columns = [
        column for column in data.columns
        if column != target_column and pd.api.types.is_numeric_dtype(data[column])
    ]
    if not feature_columns:
        raise ValueError("CSV needs numeric predictor columns such as Age, Number, and Start")
    cleaned = data.dropna(subset=[target_column]).copy()
    if len(cleaned) < 10:
        raise ValueError("At least 10 labeled rows are needed for the teaching split")
    counts = cleaned[target_column].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Target must have at least two classes with two rows each")

    features = cleaned[feature_columns]
    target = cleaned[target_column].astype(str)
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.25, random_state=seed, stratify=target
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=seed,
            min_samples_leaf=2,
        ),
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    labels = model.named_steps["randomforestclassifier"].classes_.tolist()
    return {
        "model": model,
        "feature_columns": feature_columns,
        "labels": labels,
        "accuracy": float(accuracy_score(test_y, predictions)),
        "confusion_matrix": confusion_matrix(test_y, predictions, labels=labels).tolist(),
        "row_count": len(cleaned),
        "test_count": len(test_y),
        "importance": model.named_steps["randomforestclassifier"].feature_importances_.tolist(),
    }


def predict_case(model, feature_columns, values):
    if set(feature_columns) - set(values):
        raise ValueError("Provide a value for every model feature")
    row = pd.DataFrame([[values[name] for name in feature_columns]], columns=feature_columns)
    return str(model.predict(row)[0]), {
        str(label): float(probability)
        for label, probability in zip(model.classes_, model.predict_proba(row)[0])
    }
