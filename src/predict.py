"""
Make predictions on test data.
"""

import pandas as pd
from src.config import SUBMISSION_PATH


def make_predictions(model, X_test_transformed, test_data):
    """
    Make predictions on test data.
    
    Args:
        model: Trained model
        X_test_transformed: Preprocessed test features
        test_data (pd.DataFrame): Original test data (for ID column)
        
    Returns:
        pd.DataFrame: Predictions dataframe
    """
    print("\nMaking predictions on test data...")
    
    predictions = model.predict(X_test_transformed)
    probabilities = model.predict_proba(X_test_transformed)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_data.index,
        'target': ['yes' if pred == 1 else 'no' for pred in predictions],
        'probability': probabilities[:, 1]  # Probability of positive class
    })
    
    print(f"\n✅ Predictions created: {len(submission)} records")
    print("\nFirst few predictions:")
    print(submission.head())
    
    return submission


def save_predictions(predictions, filepath):
    """
    Save predictions to CSV.
    
    Args:
        predictions (pd.DataFrame): Predictions dataframe
        filepath (str): Path to save CSV
    """
    # Remove probability column for submission
    submission = predictions[['id', 'target']].copy()
    submission.to_csv(filepath, index=False)
    print(f"\n✅ Predictions saved to {filepath}")


def load_predictions(filepath):
    """
    Load predictions from CSV.
    
    Args:
        filepath (str): Path to predictions CSV
        
    Returns:
        pd.DataFrame: Predictions dataframe
    """
    predictions = pd.read_csv(filepath)
    return predictions
