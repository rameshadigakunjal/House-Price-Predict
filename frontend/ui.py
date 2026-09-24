# -*- coding: utf-8 -*-
"""Streamlit UI for the house price prediction project."""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "house_price.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
DIAGNOSTICS_PATH = os.path.join(MODEL_DIR, "diagnostics.png")
FEATURE_COLUMNS = ["area", "rooms"]
TARGET_COLUMN = "price"

st.set_page_config(
    page_title="House Price Studio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(250, 200, 166, 0.22), transparent 30%),
            radial-gradient(circle at top right, rgba(104, 154, 255, 0.14), transparent 26%),
            linear-gradient(180deg, #fbf7f2 0%, #f4efe7 48%, #eef3f8 100%);
        color: #1d2430;
        font-family: "Palatino Linotype", Palatino, Georgia, serif;
    }
    .hero {
        padding: 2rem 2rem 1.3rem 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(18, 28, 46, 0.94), rgba(32, 52, 82, 0.86));
        color: white;
        box-shadow: 0 22px 45px rgba(13, 23, 38, 0.18);
        margin-bottom: 1.2rem;
    }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.22em;
        font-size: 0.76rem;
        opacity: 0.78;
        margin-bottom: 0.3rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 3rem;
        line-height: 1.02;
    }
    .hero p {
        margin-top: 0.8rem;
        max-width: 720px;
        font-size: 1.02rem;
        opacity: 0.92;
    }
    .section-card {
        background: rgba(255, 255, 255, 0.84);
        border-radius: 20px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 14px 28px rgba(29, 36, 48, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.68);
    }
    .big-number {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.15rem 0 0;
    }
    .small-label {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        opacity: 0.7;
    }
    .predict-form {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(45, 55, 72, 0.08);
        border-radius: 20px;
        padding: 1rem 1.2rem;
        box-shadow: 0 12px 26px rgba(29, 36, 48, 0.08);
    }
    .predict-panel {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(45, 55, 72, 0.08);
        border-radius: 20px;
        padding: 1rem 1.2rem;
        box-shadow: 0 12px 26px rgba(29, 36, 48, 0.08);
    }
    div.stButton > button {
        width: 100%;
        min-height: 2.8rem;
        border-radius: 12px;
        border: 0;
        color: #ffffff;
        background: linear-gradient(135deg, #23344d 0%, #1a2538 100%);
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    div.stButton > button:hover {
        color: #ffffff;
        background: linear-gradient(135deg, #2f4565 0%, #23344d 100%);
    }
    div.stButton > button:focus,
    div.stButton > button:focus:not(:active) {
        color: #ffffff;
        border: 2px solid #8fb3d9;
        box-shadow: 0 0 0 0.2rem rgba(143, 179, 217, 0.24);
    }
    div[data-baseweb="input"] input {
        background: rgba(255, 255, 255, 0.95);
        color: #182233;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH)


def split_numpy(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    test_count = max(1, min(len(X) - 1, int(round(len(X) * test_size))))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize(X, scaler):
    return (X - scaler["mean"]) / scaler["scale"]


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


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    dataset = load_dataset()
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
        }
    return model, scaler, dataset, metrics


def predict_batch(model, scaler, X):
    scaled = standardize(X, scaler)
    return model["intercept"] + scaled @ np.asarray(model["coefficients"], dtype=float)


@st.cache_data(show_spinner=False)
def get_backend_health(base_url):
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=4)
        return response.status_code, response.json()
    except Exception as exc:
        return None, {"error": str(exc)}


def format_currency(value):
    return f"${value:,.0f}"


def predict_local(model, scaler, area, rooms):
    features = np.array([[area, rooms]], dtype=float)
    return float(predict_batch(model, scaler, features)[0])


def predict_api(base_url, area, rooms):
    response = requests.post(
        f"{base_url.rstrip('/')}/predict",
        json={"area": area, "rooms": rooms},
        timeout=8,
    )
    response.raise_for_status()
    return response.json()


def build_gauge(value, minimum, maximum):
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"prefix": "$", "valueformat": ",.0f"},
            gauge={
                "axis": {"range": [minimum, maximum]},
                "bar": {"color": "#3f6f8f"},
                "steps": [
                    {"range": [minimum, minimum + (maximum - minimum) * 0.35], "color": "#ecf2f6"},
                    {"range": [minimum + (maximum - minimum) * 0.35, minimum + (maximum - minimum) * 0.7], "color": "#d9e4ef"},
                    {"range": [minimum + (maximum - minimum) * 0.7, maximum], "color": "#c7d9ea"},
                ],
            },
        )
    )


try:
    MODEL, SCALER, DATASET, METRICS = load_model_bundle()
    MODEL_READY = True
except Exception as exc:
    MODEL = None
    SCALER = None
    DATASET = None
    METRICS = None
    MODEL_READY = False
    MODEL_ERROR = str(exc)
else:
    MODEL_ERROR = None


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">House Price Studio</div>
        <h1>Predict property value with a clean, focused workflow.</h1>
        <p>Train once, serve with Flask, and explore the same model through a Streamlit interface built around area and room count.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)
col_a.markdown(
    f'<div class="section-card"><div class="small-label">Training samples</div><div class="big-number">{len(DATASET) if DATASET is not None else "--"}</div></div>',
    unsafe_allow_html=True,
)
col_b.markdown(
    '<div class="section-card"><div class="small-label">Model</div><div class="big-number">NumPy Linear Regression</div></div>',
    unsafe_allow_html=True,
)
col_c.markdown(
    f'<div class="section-card"><div class="small-label">Model ready</div><div class="big-number">{"Yes" if MODEL_READY else "No"}</div></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    backend_url = st.text_input("Backend URL", "http://localhost:5000")
    inference_mode = st.radio("Inference mode", ["Auto", "Local", "API"], index=1)
    st.caption("Auto uses API when available and switches to local prediction otherwise.")

    status_code, health_payload = get_backend_health(backend_url)
    backend_ready = status_code == 200

    if backend_ready:
        st.success("Backend connected")
        st.caption("API endpoints are available.")
    else:
        if inference_mode == "API":
            st.error("API mode is selected, but backend is unavailable.")
            with st.expander("Connection details"):
                st.text(health_payload.get("error", "Start backend/app.py to enable API inference."))
        else:
            st.info("Backend is offline. The app will use local inference.")
            st.caption("Start backend/app.py only when you want API mode.")

predict_tab, explore_tab, insights_tab = st.tabs(["Predict Price", "Dataset Explorer", "Model Insights"])

with predict_tab:
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="predict-form">', unsafe_allow_html=True)
        area = st.slider("Area (sq ft)", min_value=500, max_value=3500, value=1800, step=50)
        rooms = st.slider("Rooms", min_value=1, max_value=6, value=3, step=1)
        location_hint = st.text_input("Optional note", placeholder="Detached, suburban, renovated...")
        predict_clicked = st.button("Predict Price", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="predict-panel">', unsafe_allow_html=True)
        if predict_clicked:
            source = "Local"
            prediction = None
            if inference_mode in ("API", "Auto") and backend_url:
                try:
                    api_result = predict_api(backend_url, area, rooms)
                    prediction = float(api_result["predicted_price"])
                    source = "API"
                except Exception:
                    if inference_mode == "API":
                        st.error("API prediction failed. Check that the Flask backend is running.")
            if prediction is None and MODEL_READY:
                prediction = predict_local(MODEL, SCALER, area, rooms)
            if prediction is not None:
                st.metric("Predicted price", format_currency(prediction), delta=f"via {source}")
                gauge = build_gauge(
                    prediction,
                    float(DATASET[TARGET_COLUMN].min()) if DATASET is not None else 0,
                    float(DATASET[TARGET_COLUMN].max()) if DATASET is not None else max(prediction * 1.2, prediction + 1),
                )
                gauge.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(gauge, use_container_width=True)
                if location_hint:
                    st.info(f"Context note: {location_hint}")
            elif not MODEL_READY:
                st.error(MODEL_ERROR)
            else:
                st.info("Choose a prediction mode and try again.")
        else:
            st.info("Set the property details on the left, then run a prediction.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### API Example")
    example_left, example_right = st.columns(2)
    example_left.markdown(
        """
        **Request**
        ```json
        { "area": 2000, "rooms": 3 }
        ```
        """
    )
    example_right.markdown(
        """
        **Response**
        ```json
        {
          "area": 2000.0,
          "rooms": 3.0,
          "predicted_price": 347820.50,
          "currency": "USD"
        }
        ```
        """
    )

with explore_tab:
    if DATASET is None:
        st.error(MODEL_ERROR or "Dataset unavailable.")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        summary_left, summary_right, summary_third = st.columns(3)
        summary_left.metric("Average price", format_currency(DATASET[TARGET_COLUMN].mean()))
        summary_right.metric("Median price", format_currency(DATASET[TARGET_COLUMN].median()))
        summary_third.metric("Average area", f"{DATASET['area'].mean():.0f} sq ft")
        st.dataframe(DATASET, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        chart_left, chart_right = st.columns(2)
        scatter = px.scatter(
            DATASET,
            x="area",
            y="price",
            color="rooms",
            title="Area vs Price",
            template="plotly_white",
        )
        scatter.update_traces(marker=dict(size=11, line=dict(width=0.6, color="white")))
        chart_left.plotly_chart(scatter, use_container_width=True)

        hist = px.histogram(
            DATASET,
            x="price",
            nbins=12,
            title="Price distribution",
            template="plotly_white",
        )
        chart_right.plotly_chart(hist, use_container_width=True)

        corr = DATASET.corr(numeric_only=True)
        heatmap = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", title="Feature correlation")
        st.plotly_chart(heatmap, use_container_width=True)

with insights_tab:
    if not MODEL_READY:
        st.error(MODEL_ERROR)
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        metric_left, metric_mid, metric_right = st.columns(3)
        metric_left.metric("R²", f"{METRICS.get('r2', 0):.4f}")
        metric_mid.metric("MAE", format_currency(METRICS.get('mae', 0)))
        metric_right.metric("RMSE", format_currency(METRICS.get('rmse', 0)))
        st.markdown("</div>", unsafe_allow_html=True)

        coeff_df = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "coefficient": [float(value) for value in MODEL["coefficients"]],
            }
        )
        st.subheader("Model coefficients")
        st.dataframe(coeff_df, use_container_width=True, hide_index=True)

        if os.path.exists(DIAGNOSTICS_PATH):
            st.subheader("Training diagnostics")
            st.image(DIAGNOSTICS_PATH, use_container_width=True)
        else:
            st.info("Run train_model.py to generate diagnostics.png in the model folder.")

        with st.expander("Raw metrics payload"):
            st.json({
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "metrics": METRICS,
            })
