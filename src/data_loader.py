"""
data_loader.py

Purpose
-------
Centralized data loading utilities for the fraud detection project.
Separating data I/O from analysis ensures reproducibility and clean notebooks.
"""

from pathlib import Path
import pandas as pd


def load_fraud_data(path: str | Path) -> pd.DataFrame:
    """
    Load the e-commerce fraud dataset (Fraud_Data.csv).

    Parameters
    ----------
    path : str or Path
        Path to the Fraud_Data.csv file.

    Returns
    -------
    pd.DataFrame
        Raw fraud transaction data.
    """
    return pd.read_csv(path)


def load_creditcard_data(path: str | Path) -> pd.DataFrame:
    """
    Load the credit card fraud dataset (creditcard.csv).

    Parameters
    ----------
    path : str or Path
        Path to the creditcard.csv file.

    Returns
    -------
    pd.DataFrame
        Credit card transaction data.
    """
    return pd.read_csv(path)


def load_ip_country_data(path: str | Path) -> pd.DataFrame:
    """
    Load IP address to country mapping dataset.

    Parameters
    ----------
    path : str or Path
        Path to the IpAddress_to_Country.csv file.

    Returns
    -------
    pd.DataFrame
        IP range to country mapping data.
    """
    return pd.read_csv(path)
