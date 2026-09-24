# -*- coding: utf-8 -*-
"""Train a house price regression model and save the artifacts."""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "house_price.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

FEATURE_COLUMNS = ["area", "rooms"]
TARGET_COLUMN = "price"
RANDOM_STATE = 42


def train_test_split_numpy(X, y, test_size=0.2, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    test_count = max(1, min(len(X) - 1, int(round(len(X) * test_size))))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def fit_scaler(X):
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=0)
    scale = np.where(scale == 0, 1.0, scale)
    return {"mean": mean, "scale": scale}


def transform(X, scaler):
    return (X - scaler["mean"]) / scaler["scale"]


def fit_linear_regression(X, y):
    design = np.column_stack([np.ones(len(X)), X])
    params, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(params[0])
    coefficients = np.asarray(params[1:], dtype=float)
    return intercept, coefficients


def predict(intercept, coefficients, X):
    return intercept + X @ coefficients


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if total == 0:
        return 1.0
    residual = np.sum((y_true - y_pred) ** 2)
    return float(1 - (residual / total))


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(df.describe())

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split_numpy(X, y)
    scaler = fit_scaler(X_train)
    X_train_scaled = transform(X_train, scaler)
    X_test_scaled = transform(X_test, scaler)

    intercept, coefficients = fit_linear_regression(X_train_scaled, y_train)
    y_pred = predict(intercept, coefficients, X_test_scaled)

    metrics = {
        "mae": mae(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "features": FEATURE_COLUMNS,
    }

    print("\n-- Model Evaluation --")
    print(f"  MAE  : ${metrics['mae']:,.2f}")
    print(f"  RMSE : ${metrics['rmse']:,.2f}")
    print(f"  R2   : {metrics['r2']:.4f}")

    model_payload = {
        "model_type": "numpy_linear_regression",
        "feature_names": FEATURE_COLUMNS,
        "intercept": intercept,
        "coefficients": coefficients,
    }
    scaler_payload = {
        "feature_names": FEATURE_COLUMNS,
        "mean": scaler["mean"],
        "scale": scaler["scale"],
    }

    joblib.dump(model_payload, MODEL_PATH)
    joblib.dump(scaler_payload, SCALER_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("\nModel, scaler, and metrics saved to /model.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("House Price Prediction - Model Diagnostics", fontsize=14)

    axes[0].scatter(y_test, y_pred, color="#3b82d4", edgecolors="white", s=80, alpha=0.85)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 "r--", lw=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Price ($)")
    axes[0].set_ylabel("Predicted Price ($)")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()

    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=10, color="#7c5cd8", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "diagnostics.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Diagnostics plot saved: {plot_path}")


if __name__ == "__main__":
    main()
