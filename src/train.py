"""
Model training and evaluation for Bank Marketing prediction.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import warnings

from src.config import (
    RANDOM_STATE, TEST_SIZE, XGBOOST_PARAMS, LIGHTGBM_PARAMS,
    RANDOM_FOREST_PARAMS, MODEL_PATH
)
from src.preprocessing import create_preprocessor

warnings.filterwarnings('ignore')


def train_xgboost(X_train, X_test, y_train, y_test):
    """
    Train XGBoost model.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test sets
        
    Returns:
        tuple: (model, predictions on test set)
    """
    print("\n" + "="*50)
    print("Training XGBoost...")
    print("="*50)
    
    model = XGBClassifier(**XGBOOST_PARAMS)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_pred = model.predict(X_test)
    
    print("\nXGBoost Results:")
    print(classification_report(y_test, y_pred))
    
    return model, y_pred


def train_lightgbm(X_train, X_test, y_train, y_test):
    """
    Train LightGBM model.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test sets
        
    Returns:
        tuple: (model, predictions on test set)
    """
    print("\n" + "="*50)
    print("Training LightGBM...")
    print("="*50)
    
    model = LGBMClassifier(**LIGHTGBM_PARAMS)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\nLightGBM Results:")
    print(classification_report(y_test, y_pred))
    
    return model, y_pred


def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train Random Forest model.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test sets
        
    Returns:
        tuple: (model, predictions on test set)
    """
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)
    
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\nRandom Forest Results:")
    print(classification_report(y_test, y_pred))
    
    return model, y_pred


def train_all_models(X_train_transformed, X_test_transformed, y_train, y_test):
    """
    Train all three models and return results.
    
    Args:
        X_train_transformed, X_test_transformed: Transformed training and test data
        y_train, y_test: Target variables
        
    Returns:
        dict: Dictionary with models and predictions
    """
    results = {}
    
    # Train models
    xgb_model, xgb_pred = train_xgboost(
        X_train_transformed, X_test_transformed, y_train, y_test
    )
    results['XGBoost'] = {'model': xgb_model, 'predictions': xgb_pred}
    
    lgbm_model, lgbm_pred = train_lightgbm(
        X_train_transformed, X_test_transformed, y_train, y_test
    )
    results['LightGBM'] = {'model': lgbm_model, 'predictions': lgbm_pred}
    
    rf_model, rf_pred = train_random_forest(
        X_train_transformed, X_test_transformed, y_train, y_test
    )
    results['RandomForest'] = {'model': rf_model, 'predictions': rf_pred}
    
    return results


def compare_models(results, y_test):
    """
    Compare all trained models.
    
    Args:
        results (dict): Dictionary with model results
        y_test: Test target variable
        
    Returns:
        pd.DataFrame: Model comparison table
    """
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    comparison = []
    
    for model_name, model_data in results.items():
        y_pred = model_data['predictions']
        
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['1']['precision']
        recall = report['1']['recall']
        
        comparison.append({
            'Model': model_name,
            'F1 Score': f1,
            'ROC-AUC': roc_auc,
            'Precision': precision,
            'Recall': recall
        })
    
    comparison_df = pd.DataFrame(comparison)
    print("\n", comparison_df)
    
    return comparison_df


def select_best_model(results, y_test):
    """
    Select the best model based on F1 score.
    
    Args:
        results (dict): Dictionary with model results
        y_test: Test target variable
        
    Returns:
        tuple: (best_model, best_model_name)
    """
    best_model_name = None
    best_f1 = -1
    
    for model_name, model_data in results.items():
        y_pred = model_data['predictions']
        f1 = f1_score(y_test, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name
    
    best_model = results[best_model_name]['model']
    
    print(f"\n✅ Best Model: {best_model_name} (F1 Score: {best_f1:.4f})")
    
    return best_model, best_model_name


def save_model(model, filepath):
    """
    Save model to disk.
    
    Args:
        model: Trained model
        filepath (str): Path to save model
    """
    joblib.dump(model, filepath)
    print(f"\n✅ Model saved to {filepath}")


def load_model(filepath):
    """
    Load model from disk.
    
    Args:
        filepath (str): Path to saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"✅ Model loaded from {filepath}")
    return model
