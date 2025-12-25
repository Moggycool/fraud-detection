# src/preprocessing.py
"""Preprocessing utilities for fraud detection dataset."""
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class OneHotEncoderWithUnknownReporter(BaseEstimator, TransformerMixin):
    """Estimator wrapper around sklearn's OneHotEncoder that reports unknown
    categories encountered during `transform` for debugging and data quality checks.

    Parameters mirror sklearn's `OneHotEncoder` for compatibility with
    scikit-learn utilities like `clone` and `get_params`.
    """

    def __init__(self, handle_unknown='ignore', sparse_output=True, min_frequency=5, drop=None):
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.min_frequency = min_frequency
        self.drop = drop
        self.encoder = OneHotEncoder(
            handle_unknown=self.handle_unknown,
            sparse_output=self.sparse_output,
            min_frequency=self.min_frequency,
            drop=self.drop,
        )

    def fit(self, X, y=None):
        # Fit underlying encoder and expose common attributes
        self.encoder.fit(X)
        # expose categories_ to match OneHotEncoder API
        self.categories_ = getattr(self.encoder, 'categories_', None)
        self.feature_names_in_ = getattr(
            self.encoder, 'feature_names_in_', None)
        return self

    def transform(self, X):
        # Normalize input and column names
        if hasattr(X, 'columns'):
            cols = list(X.columns)
            arr = X.values
            X_input = X
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            X_input = arr
            cols = [f'col_{i}' for i in range(arr.shape[1])]

        # Check unknowns per column using fitted categories_
        unknowns_report = {}
        if hasattr(self.encoder, 'categories_'):
            n_check = min(len(self.encoder.categories_), arr.shape[1])
            for i in range(n_check):
                cats = self.encoder.categories_[i]
                present = set(pd.Series(arr[:, i]).dropna().unique())
                known = set(cats)
                unknowns = sorted(list(present - known))
                if unknowns:
                    col_name = cols[i] if i < len(cols) else f'col_{i}'
                    unknowns_report[col_name] = unknowns

        if unknowns_report:
            print("Warning: Found unknown categories during transform:")
            for col, vals in unknowns_report.items():
                print(f" - {col}: {vals}")

        return self.encoder.transform(X_input)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.encoder.get_feature_names_out(input_features)


# -------------------------
# Data Cleaning (from Task 1)
# -------------------------


def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Fraud_Data.csv.

    Steps
    -----
    1. Remove duplicates
    2. Convert timestamps
    3. Drop invalid timestamps
    4. Handle missing values
    5. Validate target variable
    """
    required_cols = {
        "user_id", "signup_time", "purchase_time",
        "purchase_value", "class"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy().drop_duplicates()

    # Convert timestamps
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")

    df = df.dropna(subset=["signup_time", "purchase_time"])

    # Handle categorical missing values
    for col in ["browser", "source", "sex"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Handle numerical missing values
    for col in ["age", "purchase_value"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Validate target
    if not set(df["class"].unique()).issubset({0, 1}):
        raise ValueError("Target column 'class' must be binary (0/1).")

    return df


# -------------------------
# Feature Preprocessing
# -------------------------
def build_preprocessor(
    df: pd.DataFrame,
    num_features: Optional[List[str]] = None,
    cat_features: Optional[List[str]] = None,
    scaler: str = 'standard',
    handle_unknown_cats: str = 'ignore',
    min_frequency: int = 5,
    sparse_output: bool = True
):
    """

    Build preprocessing pipeline for numeric and categorical features.
    Enhanced for fraud detection.

    Parameters
    ----------
    df : DataFrame
        Input data
    num_features : List[str], optional
        List of numeric columns. If None, infer automatically.
    cat_features : List[str], optional
        List of categorical columns. If None, infer automatically.
    scaler : str
        'standard' (StandardScaler) or 'robust' (RobustScaler - better for outliers)
    handle_unknown_cats : str
        How to handle unknown categories: 'ignore' or 'error'
    min_frequency : int
        Minimum frequency for OneHotEncoder categories
    sparse_output : bool
        Whether to return sparse matrices

    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline
    """

    if num_features is None:
        num_features = df.select_dtypes(
            include=["int64", "float64"]).columns.tolist()
    if cat_features is None:
        cat_features = df.select_dtypes(include=["object"]).columns.tolist()
    # Remove target column if present
    if 'class' in num_features:
        num_features.remove('class')
    if 'class' in cat_features:
        cat_features.remove('class')
    # Choose scaler
    if scaler.lower() == 'robust':
        scaler_obj = RobustScaler()  # Better for fraud data with outliers
    else:
        scaler_obj = StandardScaler()

    # Numeric pipeline with imputation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', scaler_obj)
    ])
    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoderWithUnknownReporter(
            handle_unknown=handle_unknown_cats,
            sparse_output=sparse_output,
            min_frequency=min_frequency,
            drop=None
        ))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features)
        ]
    )
# -------------------------
# Feature-Target Separation


def build_preprocessor_fraud_specific(df: pd.DataFrame):
    """
    Build preprocessing pipeline optimized for fraud detection.

    Fraud-specific considerations:
    1. Use RobustScaler for monetary amounts (outliers are important)
    2. High-cardinality features need special handling
    3. Time-based features may need cyclic encoding
    """
    # Auto-detect feature types
    numeric_features = df.select_dtypes(
        include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object"]).columns.tolist()

    # Remove target if present
    for feature_list in [numeric_features, categorical_features]:
        if 'class' in feature_list:
            feature_list.remove('class')

    # Identify potential high-cardinality features
    high_cardinality = []
    for col in categorical_features:
        if df[col].nunique() > 50:
            high_cardinality.append(col)

    # For fraud, we might want different handling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # Robust scaling for fraud amounts
    ])

    # For categorical with reasonable cardinality
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoderWithUnknownReporter(
            handle_unknown='ignore',
            sparse_output=True,
            min_frequency=10,  # Higher threshold for fraud data
            drop=None
        ))
    ])

    transformers = [
        ("num", numeric_transformer, numeric_features),
        ("cat", cat_transformer, [
         c for c in categorical_features if c not in high_cardinality])
    ]

    # Handle high-cardinality features separately if any
    if high_cardinality:
        # For high-cardinality, use frequency encoding or target encoding
        # This is a placeholder - you'd need to implement this
        print(
            f"Warning: High-cardinality features detected: {high_cardinality}")
        print("Consider using frequency encoding or dropping these features.")

    return ColumnTransformer(transformers=transformers)


def separate_features_target(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target with optional column dropping.

    Parameters
    ----------
    df : DataFrame
    target_col : str
        Name of target column
    drop_cols : List[str], optional
        Additional columns to drop from features

    Returns
    -------
    X, y : DataFrame, Series
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])

    # Drop additional columns if specified
    if drop_cols:
        available_drop = [col for col in drop_cols if col in X.columns]
        X = X.drop(columns=available_drop)
        if len(available_drop) < len(drop_cols):
            missing = set(drop_cols) - set(available_drop)
            print(f"Warning: Columns not found for dropping: {missing}")

    y = df[target_col]

    # Validate target distribution
    fraud_ratio = y.mean()
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Fraud rate: {fraud_ratio:.4f} ({fraud_ratio*100:.2f}%)")

    if fraud_ratio < 0.01:
        print("Warning: Very low fraud rate (<1%). Consider anomaly detection methods.")

    return X, y


def create_time_features(df: pd.DataFrame, time_cols: List[str]) -> pd.DataFrame:
    """
    Create time-based features for fraud detection.

    Parameters
    ----------
    df : DataFrame
    time_cols : List[str]
        List of datetime column names

    Returns
    -------
    DataFrame with added time features
    """
    df = df.copy()

    for col in time_cols:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            # Extract time components
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day

            # Cyclical encoding for time features
            df[f'{col}_hour_sin'] = np.sin(2 * np.pi * df[f'{col}_hour'] / 24)
            df[f'{col}_hour_cos'] = np.cos(2 * np.pi * df[f'{col}_hour'] / 24)
            df[f'{col}_dayofweek_sin'] = np.sin(
                2 * np.pi * df[f'{col}_dayofweek'] / 7)
            df[f'{col}_dayofweek_cos'] = np.cos(
                2 * np.pi * df[f'{col}_dayofweek'] / 7)

    return df


# Keep original simple functions for backward compatibility
def build_preprocessor_simple(df, num_features=None, cat_features=None):
    """Simple version for backward compatibility."""
    return build_preprocessor(df, num_features, cat_features, scaler='standard')
