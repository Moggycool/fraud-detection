# Fraud Detection System – Adey Innovations Inc.
## Data Preprocessing & Feature Engineering (Task-1)
Overview

- This project applies a structured preprocessing pipeline to prepare fraud transaction data for machine learning modeling. The pipeline ensures reproducibility, prevents data leakage, and addresses severe class imbalance.

## Key Steps

- Missing values checked and handled appropriately
- IP addresses converted to integer format and mapped to countries
- Time-based features created (hour, weekday, time since signup)
- Transaction velocity features engineered
- Numerical features standardized using StandardScaler
- Categorical variables encoded using OneHotEncoder
- Train-test split performed prior to resampling
- Class imbalance handled using SMOTE applied only to training data
## Class Imbalance
Both datasets exhibit extreme class imbalance (<2% fraud). SMOTE was selected to synthetically oversample the minority class while preserving majority class information.

Reproducibility
- All preprocessing steps are implemented using Scikit-learn pipelines and ColumnTransformer to ensure consistency across training and inference stages.

# Fraud Detection Project – Task 2

## Objective
Build, train, and evaluate classification models to detect fraudulent transactions
using behavioral, temporal, and geolocation features from Task 1.  
Focus on handling class imbalance and comparing baseline and ensemble models.

---

## Repository Structure


```
fraud-detection
├─ notebooks
│  ├─ eda-creditcard.ipynb
│  ├─ eda-fraud-data.ipynb
│  ├─ feature-engineering.ipynb
│  ├─ modeling.ipynb
│  ├─ README.md
│  ├─ shap-explainability.ipynb
│  └─ __init__.py
├─ README.md
├─ requirements.txt
├─ scripts
│  ├─ README.md
│  └─ __init__.py
├─ src
│  ├─ cv.py
│  ├─ data_loader.py
│  ├─ data_split.py
│  ├─ feature_engineering.py
│  ├─ geo_utils.py
│  ├─ imbalance.py
│  ├─ metrics.py
│  ├─ models.py
│  ├─ preprocessing.py
│  ├─ visualization.py
│  └─ __init__.py
└─ tests
   └─ __init__.py

```

---

## Task 2 Workflow

1. **Load Data**
   - Feature-engineered dataset from Task 1: `fraud_data_features.csv`.
   - Cleaned using `clean_fraud_data()` from `src/preprocessing.py`.

2. **Separate Features and Target**
   - `X, y = separate_features_target(df, target_col="class")`.

3. **Stratified Train/Test Split**
   - Preserves class distribution in both sets: `stratified_split()`.

4. **Preprocessing Pipeline**
   - Numeric features scaled (StandardScaler)
   - Categorical features one-hot encoded (OneHotEncoder)
   - Built using `build_preprocessor()`.

5. **Handle Class Imbalance**
   - SMOTE applied **only on training data** using `apply_smote()`.

6. **Model Training**
   - Baseline: Logistic Regression (`logistic_regression()`)
   - Ensemble: Random Forest (`random_forest()`)

7. **Evaluation**
   - Metrics: F1-score, AUC-PR, Confusion Matrix
   - Visualizations: Confusion matrices & Precision–Recall curves
   - All computed using `evaluate_model()` and `visualization.py`.

8. **Cross-Validation**
   - Stratified K-Fold (k=5) using `stratified_cv()` to ensure robust estimates.

9. **Model Comparison**
   - Side-by-side table of F1 and AUC-PR
   - Random Forest selected as final model for higher fraud detection performance.

10. **Save Final Model**
    - Saved as `../models/final_fraud_model.pkl` using `joblib`.

---

## Unit Tests

- All modules in `src/` are tested using **pytest**.
- To run tests:

``bash
pytest tests/ --maxfail=1 --disable-warnings -q

## Usage
- Install dependencies:
pip install -r requirements.txt

- Run the notebook:
jupyter notebook notebooks/modeling.ipynb

Follow each section in the notebook to:
- Load and preprocess data
- Train baseline and ensemble models
- Evaluate metrics
- Visualize confusion matrices and PR curves
- Save the final model