"""
SHAP Interpretability Demo

This script demonstrates how to generate SHAP explanations for the trained model.
It shows which features drive predictions and creates visualization outputs.

Usage:
    python shap_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_PATH, OUTPUTS_ROOT
)
from src.preprocessing import (
    prepare_data, create_preprocessor
)
from src.train import load_model
from src.interpret import (
    generate_shap_report, get_top_features
)
from sklearn.model_selection import train_test_split


def main():
    """
    Generate SHAP explanations for model predictions.
    """
    print("\n" + "="*60)
    print("SHAP MODEL INTERPRETABILITY DEMO")
    print("="*60)
    
    # =====================================================
    # 1. LOAD DATA AND MODEL
    # =====================================================
    print("\n[STEP 1] Loading data and model...")
    print("-" * 60)
    
    try:
        # Load and preprocess data
        X_train, y_train, X_test = prepare_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
        print(f"✅ Data loaded successfully")
        
        # Load model
        model = load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # =====================================================
    # 2. CREATE PREPROCESSING PIPELINE
    # =====================================================
    print("\n[STEP 2] Creating preprocessing pipeline...")
    print("-" * 60)
    
    # Split training data
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train_split)
    X_test_transformed = preprocessor.transform(X_test)
    
    print(f"✅ Preprocessing pipeline created")
    print(f"   - Training data shape: {X_train_transformed.shape}")
    print(f"   - Test data shape: {X_test_transformed.shape}")
    
    # =====================================================
    # 3. GENERATE SHAP REPORT
    # =====================================================
    print("\n[STEP 3] Generating SHAP explanations...")
    print("-" * 60)
    
    # Get feature names (after preprocessing)
    # Note: After preprocessing, feature names will be transformed
    # We'll use generic names for now
    n_features = X_test_transformed.shape[1]
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    try:
        explainer, shap_values = generate_shap_report(
            model, 
            X_train_transformed, 
            X_test_transformed,
            feature_names=feature_names,
            num_examples=3  # Generate 3 examples
        )
    except Exception as e:
        print(f"❌ Error generating SHAP report: {e}")
        return
    
    # =====================================================
    # 4. GET TOP FEATURES
    # =====================================================
    print("\n[STEP 4] Top Features by SHAP Importance...")
    print("-" * 60)
    
    top_features = get_top_features(
        shap_values, X_test_transformed, 
        feature_names=feature_names, 
        top_n=15
    )
    
    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n" + "="*60)
    print("✅ SHAP ANALYSIS COMPLETE")
    print("="*60)
    
    print(f"\n📊 Outputs saved to: {OUTPUTS_ROOT / 'shap_explanations'}")
    print(f"\n📈 Key Insights:")
    print(f"   ✓ Summary plots show overall feature importance")
    print(f"   ✓ Waterfall plots explain individual predictions")
    print(f"   ✓ Force plots show prediction drivers")
    
    print(f"\n💡 Interpretation:")
    print(f"   - Red features push prediction UP (towards 'YES')")
    print(f"   - Blue features push prediction DOWN (towards 'NO')")
    print(f"   - Larger bars = stronger impact on prediction")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
