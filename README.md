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

