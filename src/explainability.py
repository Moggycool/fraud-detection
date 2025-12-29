"""
Model Explainability Utilities for Fraud Detection (SHAP & Feature Importance)
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from IPython.display import display


def plot_builtin_feature_importance(model, feature_names, top_n=10, figsize=(8, 6)):
    """
    Plot the built-in feature importance of an ensemble model.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise AttributeError(
            "Model does not support 'feature_importances_' attribute.")

    # Pair with feature names
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = np.array(feature_names)[indices]
    top_importances = importances[indices]

    plt.figure(figsize=figsize)
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title(f"Top {top_n} Built-in Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    return list(zip(top_features, top_importances))


def compute_shap_values(model, X, model_type="tree"):
    """
    Compute SHAP values for a fitted model and feature matrix.
    model_type: "tree" (e.g. RandomForest) or "linear".
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return explainer, shap_values


def plot_shap_summary(shap_values, X, feature_names=None, plot_type="dot"):
    """
    Plot a global summary plot of SHAP values (all features).
    """
    if feature_names is not None:
        # SHAP plots will use names in X if set
        shap_values.feature_names = feature_names
    shap.summary_plot(
        shap_values, X, feature_names=feature_names, plot_type=plot_type)


def plot_shap_force(explainer, shap_values, X, sample_idx, feature_names=None):
    """
    Plot a SHAP force plot for the given sample(s).
    sample_idx can be a single int or list of ints.
    """
    # Single instance
    if isinstance(sample_idx, int):
        idxs = [sample_idx]
    else:
        idxs = list(sample_idx)
    for idx in idxs:
        display(shap.plots.force(
            explainer.expected_value,
            shap_values.values[idx],
            X[idx],
            feature_names=feature_names
        ))


def get_test_case_indices(y_true, y_pred, y_prob=None):
    """
    Find indices of true positives, false positives, and false negatives.
    Returns a dict: {"TP": idx, "FP": idx, "FN": idx}
    """
    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]

    # reference y_prob to avoid "unused argument" warnings (no-op)
    if y_prob is not None:
        _ = y_prob

    # For demo, pick first example in each group (if available)
    return {
        "TP": int(tp[0]) if len(tp) else None,
        "FP": int(fp[0]) if len(fp) else None,
        "FN": int(fn[0]) if len(fn) else None,
    }
