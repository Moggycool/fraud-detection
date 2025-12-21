"""
preprocessing.py

Purpose
-------
Data cleaning and basic preprocessing utilities.
These steps prepare raw datasets for EDA and feature engineering.
"""

import pandas as pd


def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Fraud_Data.csv.

    Steps
    -----
    1. Remove duplicate rows.
    2. Convert timestamp columns to datetime.
    3. Drop rows with invalid critical timestamps.
    4. Handle missing values:
       - Categorical: fill with 'Unknown'
       - Numerical: fill with median
    5. Validate target column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw fraud dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned fraud dataset.
    """
    required_cols = {
        "user_id", "signup_time", "purchase_time",
        "purchase_value", "class"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy().drop_duplicates()

    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")
    df = df.dropna(subset=["signup_time", "purchase_time"])

    for col in ["browser", "source", "sex"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col in ["age", "purchase_value"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def clean_creditcard_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean creditcard.csv.

    Steps
    -----
    1. Remove duplicate rows.
    2. Validate target column.

    Notes
    -----
    - No missing values expected.
    - PCA features (V1–V28) must not be altered.

    Parameters
    ----------
    df : pd.DataFrame
        Raw credit card dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned credit card dataset.
    """
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found.")

    return df.copy().drop_duplicates()
