"""
SHAP Model Interpretability - Explain model predictions

This module provides SHAP (SHapley Additive exPlanations) visualizations
to understand which features drive the model's predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings

from src.config import OUTPUTS_ROOT

warnings.filterwarnings('ignore')


def create_shap_explainer(model, X_train_transformed):
    """
    Create SHAP explainer for the model.
    
    Args:
        model: Trained model (XGBoost, LightGBM, or RandomForest)
        X_train_transformed: Transformed training data for background
        
    Returns:
        shap.Explainer: SHAP explainer object
    """
    print("\n📊 Creating SHAP explainer...")
    
    # Use a sample of training data as background
    # (using full data can be slow for large datasets)
    background_sample = X_train_transformed[:min(100, len(X_train_transformed))]
    
    explainer = shap.TreeExplainer(model, background_sample)
    
    print("✅ SHAP explainer created successfully")
    return explainer


def compute_shap_values(explainer, X_test_transformed):
    """
    Compute SHAP values for test data.
    
    Args:
        explainer: SHAP explainer object
        X_test_transformed: Transformed test data
        
    Returns:
        np.ndarray: SHAP values
    """
    print("📊 Computing SHAP values (this may take a moment)...")
    
    shap_values = explainer.shap_values(X_test_transformed)
    
    # For binary classification, take positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    print(f"✅ SHAP values computed for {len(shap_values)} samples")
    return shap_values


def plot_shap_summary(explainer, X_test_transformed, shap_values, 
                      feature_names=None, output_path=None):
    """
    Create SHAP summary plot (feature importance).
    
    Args:
        explainer: SHAP explainer object
        X_test_transformed: Transformed test data
        shap_values: Computed SHAP values
        feature_names: List of feature names (optional)
        output_path: Path to save figure (optional)
    """
    print("\n📊 Creating SHAP summary plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Create summary plot
    shap.summary_plot(shap_values, X_test_transformed, 
                     feature_names=feature_names, show=False)
    
    plt.title("SHAP Feature Importance - Summary Plot", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Summary plot saved to {output_path}")
    
    plt.show()


def plot_shap_bar(explainer, X_test_transformed, shap_values, 
                  feature_names=None, output_path=None):
    """
    Create SHAP bar plot (mean absolute SHAP values).
    
    Args:
        explainer: SHAP explainer object
        X_test_transformed: Transformed test data
        shap_values: Computed SHAP values
        feature_names: List of feature names (optional)
        output_path: Path to save figure (optional)
    """
    print("\n📊 Creating SHAP bar plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    shap.summary_plot(shap_values, X_test_transformed, 
                     plot_type="bar", feature_names=feature_names, show=False)
    
    plt.title("SHAP Feature Importance - Mean |SHAP| Values", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Bar plot saved to {output_path}")
    
    plt.show()


def plot_shap_waterfall(explainer, X_test_transformed, shap_values, 
                        instance_idx=0, feature_names=None, output_path=None):
    """
    Create SHAP waterfall plot for a single prediction.
    
    Shows how features contribute to pushing prediction up or down.
    
    Args:
        explainer: SHAP explainer object
        X_test_transformed: Transformed test data
        shap_values: Computed SHAP values
        instance_idx: Index of instance to explain (default: 0)
        feature_names: List of feature names (optional)
        output_path: Path to save figure (optional)
    """
    print(f"\n📊 Creating SHAP waterfall plot for instance {instance_idx}...")
    
    plt.figure(figsize=(12, 8))
    
    # Create Explanation object
    explanation = shap.Explanation(
        values=shap_values[instance_idx],
        base_values=explainer.expected_value,
        data=X_test_transformed[instance_idx],
        feature_names=feature_names
    )
    
    # Create waterfall plot
    shap.plots.waterfall(explanation, show=False)
    
    plt.title(f"SHAP Waterfall Plot - Instance {instance_idx}", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Waterfall plot saved to {output_path}")
    
    plt.show()


def plot_shap_force(explainer, X_test_transformed, shap_values, 
                    instance_idx=0, feature_names=None, output_path=None):
    """
    Create SHAP force plot for a single prediction.
    
    Args:
        explainer: SHAP explainer object
        X_test_transformed: Transformed test data
        shap_values: Computed SHAP values
        instance_idx: Index of instance to explain (default: 0)
        feature_names: List of feature names (optional)
        output_path: Path to save figure (optional)
    """
    print(f"\n📊 Creating SHAP force plot for instance {instance_idx}...")
    
    # Create Explanation object
    explanation = shap.Explanation(
        values=shap_values[instance_idx],
        base_values=explainer.expected_value,
        data=X_test_transformed[instance_idx],
        feature_names=feature_names
    )
    
    # Create force plot
    shap.plots.force(explanation, matplotlib=True, show=False)
    
    plt.title(f"SHAP Force Plot - Instance {instance_idx}", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Force plot saved to {output_path}")
    
    plt.show()


def generate_shap_report(model, X_train_transformed, X_test_transformed, 
                        feature_names=None, num_examples=5):
    """
    Generate complete SHAP report with multiple visualizations.
    
    Args:
        model: Trained model
        X_train_transformed: Transformed training data
        X_test_transformed: Transformed test data
        feature_names: List of feature names (optional)
        num_examples: Number of individual predictions to explain
    """
    print("\n" + "="*60)
    print("GENERATING SHAP INTERPRETABILITY REPORT")
    print("="*60)
    
    # Create explainer
    explainer = create_shap_explainer(model, X_train_transformed)
    
    # Compute SHAP values
    shap_values = compute_shap_values(explainer, X_test_transformed)
    
    # Create output directory
    shap_dir = OUTPUTS_ROOT / "shap_explanations"
    shap_dir.mkdir(exist_ok=True)
    
    # Generate summary plots
    print("\n" + "-"*60)
    print("SHAP Summary Plots")
    print("-"*60)
    
    plot_shap_summary(
        explainer, X_test_transformed, shap_values,
        feature_names=feature_names,
        output_path=shap_dir / "01_summary_plot.png"
    )
    
    plot_shap_bar(
        explainer, X_test_transformed, shap_values,
        feature_names=feature_names,
        output_path=shap_dir / "02_bar_plot.png"
    )
    
    # Generate individual prediction explanations
    print("\n" + "-"*60)
    print("Individual Prediction Explanations")
    print("-"*60)
    
    for i in range(min(num_examples, len(X_test_transformed))):
        print(f"\n📍 Example {i+1}/{num_examples}")
        
        plot_shap_waterfall(
            explainer, X_test_transformed, shap_values,
            instance_idx=i, feature_names=feature_names,
            output_path=shap_dir / f"03_waterfall_example_{i+1}.png"
        )
    
    print("\n" + "="*60)
    print("✅ SHAP REPORT GENERATED SUCCESSFULLY")
    print("="*60)
    print(f"\n📁 All visualizations saved to: {shap_dir}")
    print(f"   - Summary plots show overall feature importance")
    print(f"   - Waterfall plots show individual prediction drivers")
    
    return explainer, shap_values


def get_top_features(shap_values, X_test_transformed, feature_names=None, top_n=10):
    """
    Get top N features by mean absolute SHAP value.
    
    Args:
        shap_values: Computed SHAP values
        X_test_transformed: Transformed test data
        feature_names: List of feature names (optional)
        top_n: Number of top features to return
        
    Returns:
        pd.DataFrame: Top features with their mean absolute SHAP values
    """
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create dataframe
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names if feature_names else [f'Feature_{i}' for i in range(len(mean_abs_shap))],
        'Mean |SHAP|': mean_abs_shap
    }).sort_values('Mean |SHAP|', ascending=False)
    
    print("\n📊 Top Features by Mean Absolute SHAP Value:")
    print(feature_importance_df.head(top_n).to_string(index=False))
    
    return feature_importance_df.head(top_n)
