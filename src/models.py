# src/models.py
"""Module defining machine learning models."""
# Model Definitions/Builder: Logistic Regression and Random Forest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score
import numpy as np


def logistic_regression():
    """ Logistic Regression model with balanced class weights. """
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )


def random_forest(n_estimators=200, max_depth=None):
    """ Random Forest model with balanced class weights. """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


# NEW: Threshold-optimized model wrapper
class ThresholdOptimizedModel:
    """
    Wrapper for sklearn models with threshold optimization.
    """

    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.optimal_threshold = threshold

    def fit(self, X, y):
        """Fit the underlying model."""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """Get probability predictions."""
        return self.model.predict_proba(X)

    def predict(self, X, threshold=None):
        """Predict with custom or optimal threshold."""
        if threshold is None:
            threshold = self.threshold

        probas = self.model.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)

    def set_threshold(self, threshold):
        """Set prediction threshold."""
        self.threshold = threshold
        return self

    def find_optimal_threshold(self, X_val, y_val, metric='f1', thresholds=None):
        """
        Find optimal threshold on validation set.

        Parameters
        ----------
        X_val : array-like
            Validation features
        y_val : array-like
            Validation labels
        metric : str
            'f1', 'f2', or 'cost'
        thresholds : list, optional
            Custom thresholds to test

        Returns
        -------
        optimal_threshold : float
            Best threshold
        best_score : float
            Best metric score
        """
        y_proba = self.model.predict_proba(X_val)[:, 1]

        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05)

        best_score = -1
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)

            if metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif metric == 'f2':
                score = fbeta_score(y_val, y_pred, beta=2)
            elif metric == 'cost':
                # You can implement custom cost function
                score = self._calculate_cost_score(y_val, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_thresh = thresh

        self.optimal_threshold = best_thresh
        self.threshold = best_thresh
        return best_thresh, best_score

    def _calculate_cost_score(self, y_true, y_pred):
        """Example cost function - customize based on business needs."""
        _, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
        # Example: Cost of missed fraud = 100, cost of false alarm = 10
        total_cost = (fn * 100) + (fp * 10)
        return -total_cost  # Negative because we want to minimize cost


# NEW: Factory function for threshold-optimized models
def threshold_optimized_random_forest(n_estimators=200, max_depth=None, threshold=0.65):
    """
    Create a Random Forest with threshold optimization capability.

    Parameters
    ----------
    threshold : float, default=0.65
        Optimal threshold based on your analysis
    """
    rf = random_forest(n_estimators=n_estimators, max_depth=max_depth)
    return ThresholdOptimizedModel(rf, threshold=threshold)
