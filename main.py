"""
Bank Marketing ML - Main Entry Point

This script orchestrates the complete ML pipeline:
1. Load and preprocess data
2. Create preprocessing pipeline
3. Split data
4. Train multiple models
5. Evaluate and compare models
6. Make predictions on test data
7. Save results
"""

import sys
from pathlib import Path
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, NUMERICAL_COLS, ORDINAL_COLS, NOMINAL_COLS,
    RANDOM_STATE, TEST_SIZE, MODEL_PATH, PREPROCESSOR_PATH, SUBMISSION_PATH, RESULTS_PATH
)
from src.preprocessing import (
    prepare_data, create_preprocessor
)
from src.train import (
    train_all_models, compare_models, select_best_model, save_model
)
from src.predict import make_predictions, save_predictions
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    """
    Main pipeline orchestrator.
    """
    print("\n" + "="*60)
    print("BANK MARKETING ML - COMPLETE PIPELINE")
    print("="*60)
    
    # =====================================================
    # 1. LOAD AND PREPARE DATA
    # =====================================================
    print("\n[STEP 1] Loading and preprocessing data...")
    print("-" * 60)
    
    try:
        X_train, y_train, X_test = prepare_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
        print(f"✅ Training features shape: {X_train.shape}")
        print(f"✅ Training target distribution:")
        print(f"   - Class 0 (No): {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
        print(f"   - Class 1 (Yes): {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
        print(f"✅ Test features shape: {X_test.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print(f"   Make sure train.csv and test.csv are in {TRAIN_DATA_PATH.parent}/")
        return
    
    # =====================================================
    # 2. SPLIT DATA
    # =====================================================
    print("\n[STEP 2] Splitting data into train/test sets...")
    print("-" * 60)
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, stratify=y_train
    )
    print(f"✅ Training set: {X_train_split.shape}")
    print(f"✅ Validation set: {X_val.shape}")
    
    # =====================================================
    # 3. CREATE PREPROCESSING PIPELINE
    # =====================================================
    print("\n[STEP 3] Creating preprocessing pipeline...")
    print("-" * 60)
    
    preprocessor = create_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train_split)
    X_val_transformed = preprocessor.transform(X_val)
    
    print(f"✅ Preprocessing pipeline created and fitted")
    print(f"✅ Transformed training features shape: {X_train_transformed.shape}")
    print(f"✅ Transformed validation features shape: {X_val_transformed.shape}")
    
    # =====================================================
    # 4. TRAIN MODELS
    # =====================================================
    print("\n[STEP 4] Training models...")
    print("-" * 60)
    
    results = train_all_models(
        X_train_transformed, X_val_transformed, y_train_split, y_val
    )
    
    # =====================================================
    # 5. COMPARE AND SELECT BEST MODEL
    # =====================================================
    print("\n[STEP 5] Comparing models...")
    print("-" * 60)
    
    comparison_df = compare_models(results, y_val)
    best_model, best_model_name = select_best_model(results, y_val)
    
    # Save comparison results
    comparison_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n✅ Model comparison saved to {RESULTS_PATH}")
    
    # =====================================================
    # 6. SAVE BEST MODEL
    # =====================================================
    print("\n[STEP 6] Saving best model...")
    print("-" * 60)
    
    save_model(best_model, MODEL_PATH)

    # Save the fitted preprocessing pipeline so the API can load it directly
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"✅ Preprocessor saved to {PREPROCESSOR_PATH}")
    
    # =====================================================
    # 7. MAKE PREDICTIONS ON TEST DATA
    # =====================================================
    print("\n[STEP 7] Making predictions on test data...")
    print("-" * 60)
    
    # Transform test data
    X_test_transformed = preprocessor.transform(X_test)
    
    # Make predictions
    predictions = make_predictions(best_model, X_test_transformed, X_test)
    
    # Save predictions
    save_predictions(predictions, SUBMISSION_PATH)
    
    # =====================================================
    # FINAL SUMMARY
    # =====================================================
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print(f"\n📊 Final Results:")
    print(f"   - Best Model: {best_model_name}")
    print(f"   - Model saved: {MODEL_PATH}")
    print(f"   - Results saved: {RESULTS_PATH}")
    print(f"   - Predictions saved: {SUBMISSION_PATH}")
    
    print(f"\n📈 Model Metrics (on validation set):")
    print(comparison_df.to_string(index=False))
    
    print(f"\n🎯 Test Set Predictions:")
    print(f"   - Total predictions: {len(predictions)}")
    print(f"   - Positive predictions (Yes): {(predictions['target'] == 'yes').sum()}")
    print(f"   - Negative predictions (No): {(predictions['target'] == 'no').sum()}")
    
    print("\n" + "="*60)
    print("Ready to submit to Kaggle! 🚀")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
