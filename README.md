# Bank Marketing ML - Production-Ready Classification Model

**Predict the Success of Bank Telemarketing Campaigns**

Classify whether a bank client will subscribe to a term deposit based on telemarketing campaign data using machine learning. This project demonstrates best practices for building, evaluating, interpreting, and deploying ML models.

---

## 📊 Project Overview

- **Goal**: Predict if a client will subscribe to a term deposit (binary classification)
- **Dataset**: UCI Bank Marketing Dataset (from Kaggle competition)
- **Target Metric**: F1 Score, ROC-AUC
- **Models Compared**: XGBoost, LightGBM, Random Forest
- **Best Model**: XGBoost (based on F1 score)

---

## 🎯 Key Features

✅ **Modular Architecture** - Clean separation of concerns (preprocessing, training, evaluation)  
✅ **Multiple Models** - Compare XGBoost, LightGBM, and Random Forest  
✅ **Feature Engineering** - Age groups, balance groups, interaction features  
✅ **Class Imbalance Handling** - Scaled weights and sampling strategies  
✅ **Professional Pipeline** - Reproducible sklearn ColumnTransformer for numerical, ordinal, and categorical features  
✅ **Model Comparison** - Side-by-side evaluation metrics (F1, ROC-AUC, Precision, Recall)  
✅ **Model Interpretability** - SHAP explanations for global and local predictions  
✅ **Dockerized Deployment** - Containerized Flask API for inference  
✅ **Production API** - `/health` and `/predict` endpoints for model serving  
✅ **Submission Ready** - Generate Kaggle-compatible predictions  

---

## 📁 Project Structure

```
bank-marketing-ml/
├── data/
│   ├── raw/                 # Downloaded Kaggle dataset (train.csv, test.csv)
│   ├── processed/           # Preprocessed data (generated during training)
│   └── sample/              # Sample submission file
├── notebooks/
│   └── 01_eda_and_modeling.ipynb  # Original Kaggle notebook
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration, paths, hyperparameters
│   ├── preprocessing.py     # Data loading, feature engineering, pipelines
│   ├── train.py             # Model training logic
│   ├── evaluate.py          # Evaluation metrics and visualizations
│   ├── interpret.py         # SHAP interpretation utilities
│   └── predict.py           # Prediction on test data
├── outputs/
│   ├── submission.csv       # Final predictions for Kaggle
│   └── results.csv          # Model comparison results
├── models/
│   ├── best_model.pkl       # Serialized trained model
│   └── preprocessor.pkl     # Saved preprocessing pipeline
├── app.py                   # Flask inference API
├── main.py                  # Entry point script
├── shap_demo.py             # SHAP visualization runner
├── Dockerfile               # Container definition for API deployment
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .gitignore               # Git ignore rules
```

---

## 🚢 Deployment

### Flask API

The project exposes a lightweight inference API with two endpoints:

- `GET /health` - checks service readiness
- `POST /predict` - returns the prediction label and probability for one or more records

### Docker

Build and run the API container:

```bash
docker build -t bank-marketing-ml:latest .
docker run --rm -p 8080:8080 -v "${PWD}/models:/app/models" -v "${PWD}/outputs:/app/outputs" -v "${PWD}/data/raw:/app/data/raw" bank-marketing-ml:latest
```

---

## 🚀 Quick Start

### 1. **Download Kaggle Dataset**

```bash
# Option 1: Using Kaggle CLI
kaggle competitions download -c predict-the-success-of-bank-telemarketing
unzip predict-the-success-of-bank-telemarketing.zip -d data/raw/

# Option 2: Manual Download
# Download from Kaggle and place train.csv and test.csv in data/raw/
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Run Training Pipeline**

```bash
python main.py
```

This will:
- Load and preprocess data
- Create engineered features
- Train all three models (XGBoost, LightGBM, Random Forest)
- Evaluate and compare models
- Save the best model
- Generate predictions on test data
- Export submission.csv for Kaggle

---

## 📈 Model Performance

### Test Set Results

| Model | F1 Score | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| **XGBoost** | **0.629568** | **0.836911** | **0.516786** | **0.805317** |
| LightGBM | 0.629003 | 0.819446 | 0.541847 | 0.749571 |
| Random Forest | 0.606742 | 0.795372 | 0.538564 | 0.694683 |

**Best Model**: XGBoost selected based on F1 score

---

## 🔧 Feature Engineering

The model uses the following features:

### Numerical Features
- `age`, `balance`, `duration`, `campaign`, `pdays`, `previous`
- **Engineered**: `campaign_intensity`, `contact_rate`, `age_balance`, `duration_campaign`

### Categorical Features
- **Ordinal**: `education` (primary, secondary, tertiary)
- **Nominal**: `job`, `marital`, `housing`, `loan`, `default`, `contact`, `poutcome`
- **Engineered**: `age_group`, `balance_group`

### Temporal Features
- `month`, `weekday` (extracted from `last contact date`)

---

## 📚 Data Description

### Target Variable
- **target**: `yes` (subscribed) or `no` (did not subscribe)
  - Class imbalance: ~85% no, ~15% yes (6:1 ratio)

### Key Statistics
- **Customers**: 39,188 records
- **Features**: 20+ after engineering
- **Training/Test Split**: 80/20 stratified

### Missing Values Handling
- `contact`, `poutcome`: Frequent missing values (~26%, ~75%)
- Strategy: Impute with most frequent value or 'Missing' category
- `job`, `education`: Few missing values (~0.6%, ~3.7%)

---

## 🎓 Usage Examples

### Train the Model

```python
from src.preprocessing import prepare_data, create_preprocessor
from src.train import train_all_models, compare_models, select_best_model, save_model
from sklearn.model_selection import train_test_split
from src.config import TRAIN_DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE

# Prepare data
X, y = prepare_data(TRAIN_DATA_PATH)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Create preprocessor
preprocessor = create_preprocessor()
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train models
results = train_all_models(X_train_transformed, X_test_transformed, y_train, y_test)

# Compare and select best model
comparison_df = compare_models(results, y_test)
best_model, best_model_name = select_best_model(results, y_test)

# Save model
save_model(best_model, MODEL_PATH)
```

### Make Predictions

```python
from src.preprocessing import prepare_data, create_preprocessor
from src.predict import make_predictions, save_predictions
from src.train import load_model
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_PATH, SUBMISSION_PATH

# Load preprocessor (fit on training data)
X_train, _ = prepare_data(TRAIN_DATA_PATH)
preprocessor = create_preprocessor()
preprocessor.fit(X_train)

# Prepare test data
_, _, X_test = prepare_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
X_test_transformed = preprocessor.transform(X_test)

# Load model and make predictions
model = load_model(MODEL_PATH)
predictions = make_predictions(model, X_test_transformed, X_test)

# Save predictions
save_predictions(predictions, SUBMISSION_PATH)
```

---

## 🔍 Evaluation Metrics

- **F1 Score**: Harmonic mean of precision and recall (important for imbalanced data)
- **ROC-AUC**: Area under the ROC curve (measures model's ability to distinguish classes)
- **Precision**: Ratio of correct positive predictions to all positive predictions
- **Recall**: Ratio of correct positive predictions to all actual positives

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning framework
- **XGBoost** - Gradient boosting
- **LightGBM** - Fast gradient boosting
- **matplotlib, seaborn** - Data visualization
- **joblib** - Model serialization

---

## 📝 Configuration

Edit `src/config.py` to customize:

- Data paths
- Model hyperparameters
- Column definitions
- Train/test split ratio

---

## 🚦 Future Improvements

- [ ] Hyperparameter tuning (GridSearchCV, Bayesian optimization)
- [ ] SHAP for model interpretability
- [ ] Feature importance analysis with plots
- [ ] Cross-validation for more robust evaluation
- [ ] Ensemble methods (stacking, blending)
- [ ] Class imbalance techniques (SMOTE, class weights)
- [ ] Production API deployment (Flask, FastAPI)
- [ ] Unit tests for preprocessing and model functions

---

## 📊 Results and Outputs

After running `main.py`, you'll find:

- `models/best_model.pkl` - Trained LightGBM model
- `outputs/submission.csv` - Kaggle submission file
- `outputs/results.csv` - Model comparison metrics
- Console output with detailed classification reports

---

## 📖 References

- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- [Kaggle Competition](https://www.kaggle.com/competitions/predict-the-success-of-bank-telemarketing/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

## 👤 Author

**Created**: May 5, 2026

---

## 📄 License

This project is open source and available for educational purposes.

---

## ✉️ Feedback

For questions or improvements, feel free to open an issue or submit a pull request.
