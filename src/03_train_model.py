"""
train_model.py
--------------
Trains a Random Forest Classifier to predict student dropout risk.
Evaluates model performance using Recall, Precision, and ROC-AUC metrics,
and saves the trained model artifact to models/retention_model.joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def load_data(filepath: str):
    """Loads feature-engineered dataset."""
    df = pd.read_csv(filepath)

    # Exclude non-feature columns
    X = df.drop(columns=["student_id", "at_risk"])
    y = df["at_risk"]

    return df, X, y


def train_and_evaluate_model(X, y):
    """Splits data, trains Random Forest model, and reports performance metrics."""
    # 80/20 Train-Test Split with Stratification to maintain target risk proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(
        f"Training Set: {X_train.shape[0]:,} records | Test Set: {X_test.shape[0]:,} records"
    )

    # Instantiate Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict probabilities and binary outcomes
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Model Evaluation Metrics
    auc_score = roc_auc_score(y_test, y_probs)

    print("\n" + "=" * 50)
    print(" MODEL EVALUATION REPORT ")
    print("=" * 50)
    print(f"ROC-AUC Score: {auc_score:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Feature Importance Analysis
    feature_importances = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    print("\nTop 5 Predictive Risk Features:")
    print(feature_importances.head(5).to_string(index=False))

    return model, feature_importances


def save_artifact(model, output_path: str):
    """Saves serialized model object to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\nModel artifact successfully saved to: {output_path}")


def main():
    processed_path = os.path.join(
        "data", "processed", "processed_student_features.csv"
    )
    model_output_path = os.path.join("models", "retention_model.joblib")

    print("Starting Model Training Pipeline...")
    _, X, y = load_data(processed_path)
    model, _ = train_and_evaluate_model(X, y)
    save_artifact(model, model_output_path)


if __name__ == "__main__":
    main()