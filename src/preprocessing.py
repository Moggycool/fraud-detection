"""
preprocessing.py
Purpose
-------
Data cleaning and basic preprocessing utilities.
These steps prepare raw datasets for EDA and feature engineering.

Purpose
-------
Data cleaning and preprocessing utilities for fraud detection datasets.
Fully aligned with Task 1 (Interim-1) requirements.
"""


import pandas as pd


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


def clean_creditcard_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean creditcard.csv.

    Notes
    -----
    - PCA features (V1–V28) must not be altered.
    - Dataset is already scaled.
    """

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found.")

    df = df.copy().drop_duplicates()

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Target column 'Class' must be binary (0/1).")

    return df
