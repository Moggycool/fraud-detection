# src/imbalance.py
"""Module for handling class imbalance using SMOTE."""
# SMOTE for Oversampling/SMOTE Handling
import warnings
import scipy.sparse as sp
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import numpy as np


def apply_smote(X_train, y_train, sampling_strategy='auto', method='smote', random_state=42):
    """
    Apply imbalance handling techniques.

    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    sampling_strategy : float, str, or dict
        Sampling strategy for SMOTE
        - 'auto': balance classes
        - float: minority = sampling_strategy * majority
        - dict: specific class ratios
    method : str
        'smote', 'adasyn', 'borderline', 'smotetomek', 'smoteenn'
    random_state : int
        Random seed

    Returns
    -------
    X_resampled, y_resampled
    """
    if method.lower() == 'smote':
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=5
        )
    elif method.lower() == 'adasyn':
        sampler = ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_neighbors=5
        )
    elif method.lower() == 'borderline':
        sampler = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=5,
            m_neighbors=10
        )
    elif method.lower() == 'smotetomek':
        sampler = SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
    elif method.lower() == 'smoteenn':
        sampler = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    original_n = X_train.shape[0]

    # If X_train is sparse, convert to dense for resampling (SMOTE requires dense)
    is_sparse = sp.issparse(X_train)
    if is_sparse:
        X_for_resample = X_train.toarray()
    else:
        X_for_resample = X_train

    try:
        X_res, y_res = sampler.fit_resample(X_for_resample, y_train)
        print(f"Resampled: {original_n} -> {len(X_res)} samples")
        print("Class distribution after resampling:")
        unique, counts = np.unique(y_res, return_counts=True)
        for cls, count in zip(unique, counts):
            print(
                f"  Class {cls}: {count} samples ({count/len(y_res)*100:.1f}%)")
        # If original input was sparse, convert resampled features back to sparse
        if is_sparse:
            X_res = sp.csr_matrix(X_res)
        return X_res, y_res
    except (ValueError, TypeError, MemoryError, RuntimeError) as e:
        warnings.warn(
            f"Resampling failed: {e}. Returning original data. "
            f"If your features are sparse, consider using a dense preprocessor (set sparse_output=False) "
            f"or a different resampling strategy that supports sparse input."
        )
        return X_train, y_train


def apply_hybrid_sampling(X_train, y_train, over_ratio=0.8, under_ratio=0.8, random_state=42):
    """
    Apply hybrid over+under sampling.

    Parameters
    ----------
    over_ratio : float
        Oversampling ratio for minority class
    under_ratio : float
        Undersampling ratio for majority class

    Returns
    -------
    X_resampled, y_resampled
    """
    # First oversample minority
    oversample = SMOTE(sampling_strategy=over_ratio, random_state=random_state)
    X_over, y_over = oversample.fit_resample(X_train, y_train)

    # Then undersample majority
    undersample = RandomUnderSampler(
        sampling_strategy=under_ratio, random_state=random_state)
    X_res, y_res = undersample.fit_resample(X_over, y_over)

    print(f"Original: {len(X_train)} samples")
    print(f"After oversampling: {len(X_over)} samples")
    print(f"After hybrid sampling: {len(X_res)} samples")

    return X_res, y_res


# Keep original simple function for backward compatibility
def apply_smote_simple(X_train, y_train):
    """
    Simple SMOTE application (backward compatibility).
    """
    return apply_smote(X_train, y_train, sampling_strategy='auto', method='smote', random_state=42)
