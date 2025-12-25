# src/cv.py
"""Module for cross-validation utilities."""
# Cross-Validation Utilities with Threshold Optimization
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

import numpy as np
import pandas as pd


def stratified_cv(model, X, y, n_splits=5, scoring=None, threshold=0.5):
    """
    Perform stratified k-fold cross-validation with optional threshold.

    Parameters
    ----------
    model : sklearn-like estimator
        Model to evaluate.
    X : array-like or sparse matrix
        Features (can be dense or sparse).
    y : array-like
        Target labels.
    n_splits : int
        Number of folds.
    scoring : dict or None
        Dictionary of metrics. Default: F1 and AUC-PR.
    threshold : float
        Classification threshold (0.5 default).

    Returns
    -------
    dict
        Mean and standard deviation for each metric.
    """
    if scoring is None:
        scoring = {"f1": "f1", "auc_pr": "average_precision"}

    # Create threshold-aware scorers if needed
    custom_scorers = {}
    for metric_name in scoring.keys():
        if metric_name.startswith('threshold_'):
            # Handle custom threshold metrics
            base_metric = metric_name.replace('threshold_', '')
            if base_metric == 'f1':
                custom_scorers[metric_name] = make_scorer(
                    lambda y_true, y_pred_proba, thr=threshold:
                    f1_score(y_true, (y_pred_proba[:, 1] >= thr).astype(int)),
                    needs_proba=True
                )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_estimator=False,
        return_train_score=False
    )

    summary = {}
    for key in scoring.keys():
        test_key = f"test_{key}"
        summary[f"{key}_mean"] = np.mean(results[test_key])
        summary[f"{key}_std"] = np.std(results[test_key])

    return summary


# NEW: Cross-validation with threshold optimization per fold
def stratified_cv_with_threshold_opt(model, X, y, n_splits=5, random_state=42):
    """
    Perform CV with threshold optimization in each fold.

    Returns threshold-specific metrics.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)

    fold_results = []
    thresholds_tested = np.arange(0.1, 0.95, 0.05)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Train model
        model_clone = clone(model)
        model_clone.fit(X_train_fold, y_train_fold)

        # Get probabilities
        y_proba = model_clone.predict_proba(X_val_fold)[:, 1]

        # Test thresholds
        fold_metrics = []
        for thresh in thresholds_tested:
            y_pred = (y_proba >= thresh).astype(int)

            metrics = {
                'fold': fold,
                'threshold': thresh,
                'f1': f1_score(y_val_fold, y_pred),
                'precision': precision_score(y_val_fold, y_pred, zero_division=0),
                'recall': recall_score(y_val_fold, y_pred)
            }
            fold_metrics.append(metrics)

        fold_results.extend(fold_metrics)

    # Aggregate results
    df_results = pd.DataFrame(fold_results)

    # Find best threshold across folds
    threshold_performance = df_results.groupby('threshold').agg({
        'f1': ['mean', 'std'],
        'precision': 'mean',
        'recall': 'mean'
    }).round(4)

    best_threshold = threshold_performance[('f1', 'mean')].idxmax()
    best_f1_mean = threshold_performance.loc[best_threshold, ('f1', 'mean')]
    best_f1_std = threshold_performance.loc[best_threshold, ('f1', 'std')]

    return {
        'best_threshold': best_threshold,
        'best_f1_mean': best_f1_mean,
        'best_f1_std': best_f1_std,
        'threshold_performance': threshold_performance,
        'detailed_results': df_results
    }


# NEW: Cross-validation for threshold-optimized models
def cv_threshold_optimized(model, X, y, n_splits=5):
    """
    CV specifically for threshold-optimized models.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Clone and train model
        model_clone = clone(model)
        model_clone.fit(X_train_fold, y_train_fold)

        # Find optimal threshold on validation set
        if hasattr(model_clone, 'find_optimal_threshold'):
            optimal_thresh, _ = model_clone.find_optimal_threshold(
                X_val_fold, y_val_fold, metric='f1'
            )
            model_clone.set_threshold(optimal_thresh)

        # Evaluate
        y_pred = model_clone.predict(X_val_fold)

        fold_metrics.append({
            'fold': fold,
            'f1': f1_score(y_val_fold, y_pred),
            'precision': precision_score(y_val_fold, y_pred, zero_division=0),
            'recall': recall_score(y_val_fold, y_pred),
            'optimal_threshold': optimal_thresh if 'optimal_thresh' in locals() else 0.5
        })

    return pd.DataFrame(fold_metrics)
