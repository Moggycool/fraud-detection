"""
feature_engineering.py

Purpose
-------
Create meaningful behavioral and temporal features
that help distinguish fraudulent from legitimate transactions.

Purpose
-------
Create temporal and behavioral features for fraud detection.
"""

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based fraud indicators.
    """

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["purchase_time"]):
        raise TypeError("purchase_time must be datetime.")

    if not pd.api.types.is_datetime64_any_dtype(df["signup_time"]):
        raise TypeError("signup_time must be datetime.")

    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds()

    return df


def add_transaction_velocity(
    df: pd.DataFrame,
    user_col: str = "user_id",
    time_col: str = "purchase_time",
    windows=("1H", "24H"),
) -> pd.DataFrame:
    """
    Add rolling transaction frequency features per user.
    """

    df = df.copy().sort_values([user_col, time_col])
    df.set_index(time_col, inplace=True)

    for window in windows:
        counts = (
            df.groupby(user_col)[user_col]
            .rolling(window)
            .count()
            .reset_index(name=f"transactions_last_{window}")
        )

        df = (
            df.reset_index()
            .merge(counts, on=[user_col, time_col], how="left")
            .set_index(time_col)
        )

    return df.reset_index()
