"""
Bank Marketing ML - Model Training Script

This script loads the Bank Marketing dataset, preprocesses it, and trains
three models: Logistic Regression, Random Forest, and LightGBM.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

def load_data(filepath):
    """Load the Bank Marketing dataset."""
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess and feature engineer the data."""
    print("Preprocessing data...")
    
    # Separate features and target
    X = df.drop(['y', 'duration'], axis=1)  # 'y' is target, 'duration' not available at prediction time
    y = df['y'].map({'yes': 1, 'no': 0})
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Features shape: {X_scaled.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X_scaled, y, scaler, le_dict

def train_models(X_train, X_test, y_train, y_test):
    """Train three models and evaluate them."""
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results[model_name] = {
            'model': model,
            'f1': f1,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'y_pred': y_pred
        }
        
        print(f"{model_name} Results:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
    
    return results

def save_results(results, output_file='results.csv'):
    """Save model comparison results to CSV."""
    print(f"\nSaving results to {output_file}...")
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'F1 Score': [results[m]['f1'] for m in results.keys()],
        'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()],
        'Precision': [results[m]['precision'] for m in results.keys()],
        'Recall': [results[m]['recall'] for m in results.keys()]
    })
    results_df.to_csv(output_file, index=False)
    print(results_df)

def main():
    """Main training pipeline."""
    print("Bank Marketing ML - Training Pipeline")
    print("=" * 50)
    
    # TODO: Update this path to your Kaggle dataset
    dataset_path = 'data/bank-additional-full.csv'
    
    # Load and preprocess
    df = load_data(dataset_path)
    X, y, scaler, le_dict = preprocess_data(df)
    
    # Apply SMOTE to handle class imbalance
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_smote, y_smote = smote.fit_resample(X, y)
    print(f"After SMOTE: {y_smote.value_counts().to_dict()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_smote, y_smote, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_smote
    )
    print(f"\nTrain set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Save results
    save_results(results)
    
    # Save best model
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    joblib.dump(best_model, 'best_model.pkl')
    print(f"\nBest model ({best_model_name}) saved as 'best_model.pkl'")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)

if __name__ == '__main__':
    main()
