# src/data_split.py
"""Module for data splitting functions."""
# Stratified Train/Test Split
from sklearn.model_selection import train_test_split


def stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Perform stratified train-test split to preserve class distribution.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
