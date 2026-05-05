"""
Data loading and preprocessing for Bank Marketing dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.config import NUMERICAL_COLS, ORDINAL_COLS, NOMINAL_COLS


def load_data(train_path, test_path=None):
    """
    Load training and optionally test data.
    
    Args:
        train_path (str): Path to training CSV file
        test_path (str, optional): Path to test CSV file
        
    Returns:
        pd.DataFrame or tuple: Training data, or (training data, test data) if test_path provided
    """
    print(f"Loading training data from {train_path}...")
    train = pd.read_csv(train_path)
    
    if test_path:
        print(f"Loading test data from {test_path}...")
        test = pd.read_csv(test_path)
        return train, test
    
    return train


def preprocess_dates(df):
    """
    Process datetime columns and extract temporal features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with temporal features
    """
    df = df.copy()
    
    if 'last contact date' in df.columns:
        df['last contact date'] = pd.to_datetime(df['last contact date'])
        df['year'] = df['last contact date'].dt.year
        df['month'] = df['last contact date'].dt.month
        df['weekday'] = df['last contact date'].dt.weekday
        df.drop(columns=['last contact date'], inplace=True)
    
    return df


def create_features(df):
    """
    Create engineered features from raw data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    df = df.copy()
    
    # Create age groups
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 20, 30, 40, 50, 60, 100],
        labels=['0-20', '21-30', '31-40', '41-50', '51-60', '60+']
    )
    
    # Create balance groups; fall back to a stable bucket for very small inputs
    if df['balance'].nunique(dropna=True) >= 5:
        df['balance_group'] = pd.qcut(
            df['balance'],
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        )
    else:
        df['balance_group'] = 'medium'
    
    # Create campaign intensity feature
    df['campaign_intensity'] = df['campaign'] / (df['pdays'].replace(-1, 999) + 1)
    df['campaign_intensity'] = df['campaign_intensity'].clip(
        upper=df['campaign_intensity'].quantile(0.99)
    )
    
    # Create contact rate
    df['contact_rate'] = df['previous'] / (df['pdays'].replace(-1, 999) + 1)
    df['contact_rate'] = df['contact_rate'].clip(
        upper=df['contact_rate'].quantile(0.99)
    )
    
    # Create interaction features
    df['age_balance'] = df['age'] * df['balance']
    df['age_balance'] = df['age_balance'].clip(
        upper=df['age_balance'].quantile(0.99)
    )
    
    df['duration_campaign'] = df['duration'] * df['campaign']
    df['duration_campaign'] = df['duration_campaign'].clip(
        upper=df['duration_campaign'].quantile(0.99)
    )
    
    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df


def create_preprocessor():
    """
    Create a preprocessing pipeline for numerical, ordinal, and nominal columns.
    
    Returns:
        ColumnTransformer: Fitted preprocessor pipeline
    """
    # Numerical pipeline
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Ordinal pipeline
    ordinal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal_encoder', OrdinalEncoder(
            categories=[['primary', 'secondary', 'tertiary']]
        ))
    ])
    
    # Nominal pipeline
    nominal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', num_pipe, NUMERICAL_COLS),
            ('ordinal', ordinal_pipe, ORDINAL_COLS),
            ('nominal', nominal_pipe, NOMINAL_COLS)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def prepare_data(train_path, test_path=None):
    """
    Load and preprocess data in one step.
    
    Args:
        train_path (str): Path to training data
        test_path (str, optional): Path to test data
        
    Returns:
        tuple: Processed data (X_train, y_train) or (X_train, y_train, X_test)
    """
    # Load data
    if test_path:
        train, test = load_data(train_path, test_path)
    else:
        train = load_data(train_path)
        test = None
    
    # Preprocess dates
    train = preprocess_dates(train)
    
    # Create features
    train = create_features(train)
    
    # Separate features and target
    X = train.drop(columns=['target', 'year'], errors='ignore')
    y = (train['target'] == 'yes').astype(int) if 'target' in train.columns else None
    
    if test is not None:
        test = preprocess_dates(test)
        test = create_features(test)
        X_test = test.drop(columns=['year'], errors='ignore')
        return X, y, X_test
    
    return X, y
