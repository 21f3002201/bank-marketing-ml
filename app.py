"""
Bank Marketing ML - Flask API

This Flask application serves the trained model for making predictions
on whether a client will subscribe to a term deposit.

Usage:
    python app.py

Then make a POST request to http://localhost:5000/predict with JSON data.
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Global variables for the model and preprocessor
model = None
scaler = None

def load_model():
    """Load the trained model from disk."""
    global model
    if os.path.exists('best_model.pkl'):
        model = joblib.load('best_model.pkl')
        print("Model loaded successfully!")
    else:
        print("ERROR: best_model.pkl not found. Run train.py first.")
        return False
    return True

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation."""
    return jsonify({
        'message': 'Bank Marketing ML API',
        'version': '1.0',
        'endpoints': {
            'POST /predict': 'Make a prediction',
            'GET /health': 'Check API health'
        },
        'example': {
            'url': 'http://localhost:5000/predict',
            'method': 'POST',
            'data': {
                'age': 30,
                'job': 'technician',
                'marital': 'married',
                'education': 'secondary',
                'default': 'no',
                'balance': 1000,
                'housing': 'yes',
                'loan': 'no',
                'contact': 'cellular',
                'day': 15,
                'month': 'may',
                'duration': 300,
                'campaign': 2,
                'pdays': -1,
                'previous': 0,
                'poutcome': 'unknown'
            }
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    status = 'healthy' if model is not None else 'model not loaded'
    return jsonify({'status': status})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction using the trained model.
    
    Expected JSON format:
    {
        'age': int,
        'job': str,
        'marital': str,
        ...
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'prediction_label': 'Will subscribe' if prediction == 1 else 'Will not subscribe',
            'probability': {
                'no': float(probability[0]),
                'yes': float(probability[1])
            },
            'confidence': float(max(probability))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Make multiple predictions at once.
    
    Expected JSON format:
    {
        'data': [
            {...},
            {...}
        ]
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided or missing "data" key'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'index': i,
                'prediction': int(pred),
                'prediction_label': 'Will subscribe' if pred == 1 else 'Will not subscribe',
                'probability': {
                    'no': float(prob[0]),
                    'yes': float(prob[1])
                },
                'confidence': float(max(prob))
            })
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Loading model...")
    if load_model():
        print("Starting Flask API...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to start API - model not found")
