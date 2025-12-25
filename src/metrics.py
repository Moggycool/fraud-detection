# src/metrics.py
"""Module for model evaluation metrics."""
# Model Evaluation Metrics: F1, AUC-PR, Confusion Matrix
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    average_precision_score,
    precision_score,
    recall_score,
    fbeta_score,
)
import numpy as np
import pandas as pd


def evaluate_model(model, X_test, y_test, threshold=None):
    """
    Compute F1, AUC-PR, and confusion matrix.

    Parameters
    ----------
    model : sklearn-like estimator or ThresholdOptimizedModel
        Model to evaluate
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    threshold : float, optional
        Custom threshold for prediction

    Returns
    -------
    dict
        Evaluation metrics
    """
    # Handle threshold-optimized models and standard estimators.
    # Prefer calling `predict(..., threshold=...)` for wrappers that implement it.
    # Otherwise fall back to using `predict_proba` when a custom threshold is provided.
    if threshold is not None:
        # Try calling predict with threshold kwarg (works for our wrapper)
        try:
            y_pred = model.predict(X_test, threshold=threshold)
        except TypeError:
            # model.predict doesn't accept threshold; try predict_proba
            if hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')):
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_proba >= threshold).astype(int)
            else:
                # As a last resort, see if model supports setting threshold
                if hasattr(model, 'set_threshold') and callable(getattr(model, 'set_threshold')):
                    model.set_threshold(threshold)
                    y_pred = model.predict(X_test)
                else:
                    # Fall back to standard predict (no threshold capability)
                    y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
    # (handled above) standard predict behavior falls through when threshold is None

    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(
        model, 'predict_proba') else None

    metrics = {
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    if y_prob is not None:
        metrics["auc_pr"] = average_precision_score(y_test, y_prob)
        metrics["y_prob"] = y_prob

    return metrics


def evaluate_model_at_thresholds(model, X_test, y_test, thresholds=None):
    """
    Evaluate model performance at multiple thresholds.

    Parameters
    ----------
    model : sklearn-like estimator
    X_test : array-like
    y_test : array-like
    thresholds : list, optional
        List of thresholds to test

    Returns
    -------
    DataFrame
        Metrics at each threshold
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    y_proba = model.predict_proba(X_test)[:, 1]

    results = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        metrics = {
            'threshold': thresh,
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred),
            'accuracy': np.mean(y_pred == y_test)
        }

        # Add confusion matrix components
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp
            })

        results.append(metrics)

    return pd.DataFrame(results)


def find_optimal_threshold(model, X_val, y_val, metric='f1', thresholds=None):
    """
    Find optimal threshold for binary classification.

    Parameters
    ----------
    model : sklearn-like estimator
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    metric : str
        Metric to optimize: 'f1', 'f2', 'precision', 'recall'
    thresholds : list, optional
        Custom thresholds to test

    Returns
    -------
    tuple
        (optimal_threshold, best_score, all_results)
    """
    y_proba = model.predict_proba(X_val)[:, 1]

    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    results = []
    best_score = -1
    optimal_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        if metric == 'f1':
            score = f1_score(y_val, y_pred)
        elif metric == 'f2':
            score = fbeta_score(y_val, y_pred, beta=2)
        elif metric == 'precision':
            score = precision_score(y_val, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_val, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        results.append({
            'threshold': thresh,
            'score': score,
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred)
        })

        if score > best_score:
            best_score = score
            optimal_threshold = thresh

    return optimal_threshold, best_score, pd.DataFrame(results)


def get_business_metrics(y_true, y_pred, cost_fp=10, cost_fn=100):
    """
    Calculate business-oriented metrics.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    cost_fp : float
        Cost of false positive
    cost_fn : float
        Cost of false negative

    Returns
    -------
    dict
        Business metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_cost = (fp * cost_fp) + (fn * cost_fn)
    cost_per_transaction = total_cost / len(y_true)

    return {
        'total_cost': total_cost,
        'cost_per_transaction': cost_per_transaction,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'true_negatives': tn,
        'fraud_capture_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
    }
