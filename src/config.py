"""
Configuration and constants for the Bank Marketing ML project.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_PROCESSED = DATA_ROOT / "processed"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
MODELS_ROOT = PROJECT_ROOT / "models"

# Create directories if they don't exist
for directory in [DATA_RAW, DATA_PROCESSED, OUTPUTS_ROOT, MODELS_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

# Data paths
TRAIN_DATA_PATH = DATA_RAW / "train.csv"
TEST_DATA_PATH = DATA_RAW / "test.csv"
SUBMISSION_PATH = OUTPUTS_ROOT / "submission.csv"
RESULTS_PATH = OUTPUTS_ROOT / "results.csv"
MODEL_PATH = MODELS_ROOT / "best_model.pkl"
PREPROCESSOR_PATH = MODELS_ROOT / "preprocessor.pkl"

# Column definitions
NUMERICAL_COLS = [
    'age', 'previous', 'pdays', 'duration', 'balance',
    'campaign_intensity', 'contact_rate', 'age_balance', 'duration_campaign'
]

ORDINAL_COLS = ['education']

NOMINAL_COLS = [
    'marital', 'housing', 'loan', 'default', 'weekday', 'contact',
    'job', 'poutcome', 'age_group', 'balance_group'
]

# Model hyperparameters
XGBOOST_PARAMS = {
    'random_state': 42,
    'scale_pos_weight': 3,
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 300,
    'subsample': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'early_stopping_rounds': 20
}

LIGHTGBM_PARAMS = {
    'random_state': 42,
    'scale_pos_weight': 2,
    'neg_bagging_fraction': 0.25,
    'verbose': -1
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 500,
    'min_samples_split': 10,
    'class_weight': 'balanced_subsample',
    'random_state': 42,
    'n_jobs': -1
}

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
RANDOM_STATE_SPLIT = 42
