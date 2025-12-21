"""
geo_utils.py

Purpose
-------
Utilities for geolocation feature creation using IP address ranges.
This enables country-level fraud pattern analysis.
"""

import pandas as pd


def convert_ip_to_int(df: pd.DataFrame, ip_col: str = "ip_address") -> pd.DataFrame:
    """
    Convert IP address column to integer format.

    Parameters
    ----------
    df : pd.DataFrame
        Fraud dataset containing IP addresses.
    ip_col : str
        Column name of IP addresses.

    Returns
    -------
    pd.DataFrame
        Dataset with additional 'ip_int' column.
    """
    df = df.copy()
    df["ip_int"] = df[ip_col].astype("int64")
    return df


def merge_ip_country(
    fraud_df: pd.DataFrame,
    ip_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge fraud transactions with country information
    using inclusive IP address range matching.

    Steps
    -----
    1. Convert IP ranges to integer.
    2. Perform backward as-of merge.
    3. Filter rows where ip_int lies within [lower, upper] bounds.
    4. Assign 'Unknown' to unmatched IPs.

    Parameters
    ----------
    fraud_df : pd.DataFrame
        Fraud dataset with 'ip_int'.
    ip_df : pd.DataFrame
        IP-to-country mapping dataset.

    Returns
    -------
    pd.DataFrame
        Fraud dataset enriched with 'country'.
    """
    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    ip_df["lower_bound_ip_address"] = ip_df["lower_bound_ip_address"].astype(
        "int64")
    ip_df["upper_bound_ip_address"] = ip_df["upper_bound_ip_address"].astype(
        "int64")

    ip_df = ip_df.sort_values("lower_bound_ip_address")

    fraud_df = pd.merge_asof(
        fraud_df.sort_values("ip_int"),
        ip_df,
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward",
    )

    fraud_df = fraud_df[
        (fraud_df["ip_int"] >= fraud_df["lower_bound_ip_address"]) &
        (fraud_df["ip_int"] <= fraud_df["upper_bound_ip_address"])
    ]

    fraud_df["country"] = fraud_df["country"].fillna("Unknown")

    return fraud_df
