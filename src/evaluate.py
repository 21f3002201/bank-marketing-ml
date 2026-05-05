"""
Model evaluation and visualization utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, roc_auc_score, precision_recall_curve
)


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model for title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name (str): Name of the model for title
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name (str): Name of the model for title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance.
    
    Args:
        model: Trained model (must have feature_importances_ attribute)
        feature_names (list): Names of features
        top_n (int): Number of top features to display
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.show()


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba (array, optional): Predicted probabilities
        model_name (str): Name of the model
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n{'='*50}")
    print(f"EVALUATION - {model_name}")
    print(f"{'='*50}")
    
    print(f"\nClassification Report:\n")
    report = classification_report(y_true, y_pred)
    print(report)
    
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score (macro): {f1:.4f}")
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    else:
        roc_auc = None
    
    return {
        'model_name': model_name,
        'f1_score': f1,
        'roc_auc_score': roc_auc
    }


def save_evaluation_report(results_dict, filepath):
    """
    Save evaluation results to CSV.
    
    Args:
        results_dict (dict): Dictionary with evaluation results
        filepath (str): Path to save results
    """
    results_df = pd.DataFrame([results_dict])
    results_df.to_csv(filepath, index=False)
    print(f"\n✅ Evaluation report saved to {filepath}")
