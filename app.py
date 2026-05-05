"""Flask API for serving the trained Bank Marketing classifier."""

from __future__ import annotations

import os

import pandas as pd
from flask import Flask, jsonify, request

from src.config import MODEL_PATH, PREPROCESSOR_PATH, TRAIN_DATA_PATH
from src.preprocessing import create_features, preprocess_dates, create_preprocessor
from src.train import load_model


def load_artifacts():
    """Load the trained model and preprocessor saved during training."""
    model = load_model(MODEL_PATH)
    preprocessor = None

    try:
        if PREPROCESSOR_PATH.exists():
            import joblib

            preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception:
        preprocessor = None

    if preprocessor is None and TRAIN_DATA_PATH.exists():
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        train_df = preprocess_dates(train_df)
        train_df = create_features(train_df)
        X_train = train_df.drop(columns=["target", "year"], errors="ignore")
        preprocessor = create_preprocessor()
        preprocessor.fit(X_train)

    return model, preprocessor


app = Flask(__name__)
model = None
preprocessor = None

try:
    model, preprocessor = load_artifacts()
except Exception as exc:
    app.logger.exception("Failed to load API artifacts: %s", exc)


@app.get("/")
def home():
    return jsonify(
        {
            "service": "bank-marketing-ml-api",
            "status": "running",
            "endpoints": ["GET /health", "POST /predict"],
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok" if model is not None and preprocessor is not None else "degraded",
            "model_loaded": model is not None,
            "preprocessor_loaded": preprocessor is not None,
        }
    )


@app.post("/predict")
def predict():
    if model is None or preprocessor is None:
        return jsonify({"error": "Model artifacts not loaded"}), 503

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid or empty JSON payload"}), 400

    if isinstance(payload, dict):
        rows = [payload]
    else:
        rows = payload

    frame = pd.DataFrame(rows)
    frame = create_features(preprocess_dates(frame))
    features = frame.drop(columns=["target", "year"], errors="ignore")
    transformed = preprocessor.transform(features)

    probabilities = model.predict_proba(transformed)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return jsonify(
        [
            {
                "prediction": "yes" if pred == 1 else "no",
                "probability": float(prob),
            }
            for pred, prob in zip(predictions, probabilities)
        ]
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
