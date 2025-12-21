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

    Parameters
    ----------
    df : pd.DataFrame
        Raw fraud dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned fraud dataset.
    """
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Convert timestamps
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")

    # Drop rows with missing critical timestamps
    df = df.dropna(subset=["signup_time", "purchase_time"])

    # Handle categorical missing values
    for col in ["browser", "source", "sex"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Handle numerical missing values
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

    Note
    ----
    The dataset has no missing values and PCA features (V1–V28)
    should not be altered at this stage.

    Parameters
    ----------
    df : pd.DataFrame
        Raw credit card dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned credit card dataset.
    """
    df = df.copy()
    df = df.drop_duplicates()
    return df
