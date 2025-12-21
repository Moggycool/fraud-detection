"""
geo_utils.py

Purpose
-------
Utilities for geolocation feature creation using IP address ranges.
This enables country-level fraud pattern analysis.

Purpose
-------
IP-to-country geolocation utilities for fraud analysis.
"""

import pandas as pd


def convert_ip_to_int(df: pd.DataFrame, ip_col: str = "ip_address") -> pd.DataFrame:
    """
    Convert IP addresses to integer format.
    """

    df = df.copy()
    df["ip_int"] = pd.to_numeric(df[ip_col], errors="coerce")
    df = df.dropna(subset=["ip_int"])
    df["ip_int"] = df["ip_int"].astype("int64")

    return df


def merge_ip_country(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Range-based IP-to-country merge.
    """

    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    ip_df["lower_bound_ip_address"] = ip_df["lower_bound_ip_address"].astype(
        "int64")
    ip_df["upper_bound_ip_address"] = ip_df["upper_bound_ip_address"].astype(
        "int64")

    ip_df = ip_df.sort_values("lower_bound_ip_address")

    merged = pd.merge_asof(
        fraud_df.sort_values("ip_int"),
        ip_df,
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward"
    )

    merged = merged[
        (merged["ip_int"] >= merged["lower_bound_ip_address"]) &
        (merged["ip_int"] <= merged["upper_bound_ip_address"])
    ]

    merged["country"] = merged["country"].fillna("Unknown")

    return merged
