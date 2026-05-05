# Bank Marketing ML Project

Predict the success of bank telemarketing campaigns using machine learning models.

## Project Overview

- **Goal**: Predict if a client will subscribe to a term deposit
- **Dataset**: UCI Bank Marketing Dataset (Kaggle)
- **Target Metric**: F1 Score & ROC-AUC
- **Models**: Logistic Regression, Random Forest, LightGBM

## Approach

1. **Data Exploration (EDA)**
   - Analyze feature distributions
   - Check class imbalance
   - Identify missing values

2. **Feature Engineering**
   - Handle categorical variables (one-hot encoding)
   - Scale numerical features
   - Address class imbalance (SMOTE / class weights)

3. **Model Building**
   - Logistic Regression (baseline)
   - Random Forest
   - LightGBM (final model)
   - Cross-validation for evaluation

4. **Model Evaluation**
   - F1 Score
   - ROC-AUC
   - Confusion Matrix
   - Precision & Recall

## Model Comparison

| Model | F1 Score | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| **LightGBM** | **TBD** | **TBD** | **TBD** | **TBD** |

## Setup & Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the Model

```bash
python train.py
```

This will:
- Load and preprocess the data
- Train all three models
- Save the best model as `best_model.pkl`
- Generate `results.csv`

### Run the Flask API

```bash
python app.py
```

Then make predictions via HTTP:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "balance": 1000, ...}'
```

## Project Structure

```
bank-marketing-ml/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── notebook.ipynb         # Exploratory analysis (coming soon)
├── train.py              # Training script
├── app.py                # Flask API (upgrade)
├── best_model.pkl        # Trained model (generated)
├── results.csv           # Model results (generated)
└── submission.csv        # Kaggle submission (generated)
```

## Key Features (Upgrades)

✅ **Flask API** - Deploy the model as a web service for predictions
✅ **Model Comparison Table** - Side-by-side metric comparison
✅ **Cross-validation** - Robust evaluation strategy
✅ **Class Imbalance Handling** - SMOTE & class weights

## Results

*To be updated after model training.*

## Technologies

- Python 3.8+
- scikit-learn
- LightGBM
- Pandas & NumPy
- Flask (API)
- Jupyter (notebook)

## Future Improvements

- Add hyperparameter tuning (GridSearch)
- Implement SHAP for model interpretability
- Add feature importance plots
- Deploy to cloud (Heroku/AWS)

## Author

- Created: 2026-05-05

## References

- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- [Kaggle Competition](https://www.kaggle.com)
