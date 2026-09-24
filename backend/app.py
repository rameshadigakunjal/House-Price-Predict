# -*- coding: utf-8 -*-
"""Flask API for house price prediction."""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "house_price.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEATURE_COLUMNS = ["area", "rooms"]
TARGET_COLUMN = "price"

app = Flask(__name__)


def split_numpy(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    test_count = max(1, min(len(X) - 1, int(round(len(X) * test_size))))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize(X, scaler):
    return (X - scaler["mean"]) / scaler["scale"]


def predict_price(model, scaler, area, rooms):
    features = np.array([[area, rooms]], dtype=float)
    scaled = standardize(features, scaler)
    prediction = model["intercept"] + scaled @ np.asarray(model["coefficients"], dtype=float)
    return float(prediction[0])


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


def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model/model.pkl is missing. Run train_model.py first.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("model/scaler.pkl is missing. Run train_model.py first.")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("data/house_price.csv is missing.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    dataset = pd.read_csv(DATA_PATH)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    else:
        X = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = dataset[TARGET_COLUMN].to_numpy(dtype=float)
        _, X_test, _, y_test = split_numpy(X, y)
        y_pred = predict_batch(model, scaler, X_test)
        metrics = {
            "mae": mae(y_test, y_pred),
            "rmse": rmse(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "test_size": int(len(y_test)),
        }

    return model, scaler, dataset, metrics


def predict_batch(model, scaler, X):
    scaled = standardize(X, scaler)
    return model["intercept"] + scaled @ np.asarray(model["coefficients"], dtype=float)


def get_model_info(model, metrics):
    coefficients = {
        feature: float(coef)
        for feature, coef in zip(FEATURE_COLUMNS, model["coefficients"])
    }
    return {
        "model_type": model.get("model_type", "numpy_linear_regression"),
        "features": FEATURE_COLUMNS,
        "coefficients": coefficients,
        "intercept": float(model["intercept"]),
        "metrics": metrics,
    }


try:
    MODEL, SCALER, DATASET, METRICS = load_bundle()
    BUNDLE_ERROR = None
except Exception as exc:  # pragma: no cover - startup failure is handled in endpoints
    MODEL = None
    SCALER = None
    DATASET = None
    METRICS = None
    BUNDLE_ERROR = str(exc)


@app.get("/health")
def health():
    if BUNDLE_ERROR:
        return jsonify({
            "status": "unhealthy",
            "model_ready": False,
            "error": BUNDLE_ERROR,
        }), 503

    return jsonify({
        "status": "healthy",
        "model_ready": True,
        "dataset_rows": int(len(DATASET)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })


@app.post("/predict")
def predict():
    if BUNDLE_ERROR:
        return jsonify({"error": BUNDLE_ERROR}), 503

    payload = request.get_json(silent=True) or {}
    try:
        area = float(payload["area"])
        rooms = float(payload["rooms"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide numeric 'area' and 'rooms' fields."}), 400

    if area <= 0 or rooms <= 0:
        return jsonify({"error": "'area' and 'rooms' must be greater than zero."}), 400

    predicted_price = predict_price(MODEL, SCALER, area, rooms)
    return jsonify({
        "area": area,
        "rooms": rooms,
        "predicted_price": round(predicted_price, 2),
        "currency": "USD",
    })


@app.get("/dataset")
def dataset():
    if BUNDLE_ERROR:
        return jsonify({"error": BUNDLE_ERROR}), 503
    return jsonify({
        "rows": int(len(DATASET)),
        "columns": DATASET.columns.tolist(),
        "data": DATASET.to_dict(orient="records"),
    })


@app.get("/model_info")
def model_info():
    if BUNDLE_ERROR:
        return jsonify({"error": BUNDLE_ERROR}), 503
    return jsonify(get_model_info(MODEL, METRICS))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
