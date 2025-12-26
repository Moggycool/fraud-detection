"""Module for tuning machine learning models using grid search and cross-validation."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, average_precision_score, precision_score, recall_score


def tune_random_forest(X, y, cv=5, search_type="grid", n_iter=20):
    """
    Tunes a RandomForestClassifier using grid or randomized search.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv : int, default=5
        Number of cross-validation folds
    search_type : str, default="grid"
        Type of search: "grid" for GridSearchCV, "random" for RandomizedSearchCV
    n_iter : int, default=20
        Number of iterations for randomized search

    Returns:
    --------
    best_estimator : RandomForestClassifier
        Best model from tuning
    best_params : dict
        Best hyperparameters
    cv_results : pd.DataFrame
        Detailed cross-validation results
    """

    # Define parameter grids
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
        "bootstrap": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None]
    }

    # Create stratified cross-validation object
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Define scoring metrics
    scoring = {
        'f1': make_scorer(f1_score),
        'average_precision': make_scorer(average_precision_score),
        'f1_weighted': make_scorer(f1_score, average='weighted')
    }

    # Initialize base model (single-threaded to avoid nested parallelism)
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=1
    )

    if search_type == "grid":
        # Grid search on a subset of parameters for efficiency
        search_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 20, 30],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced", None]
        }

        grid = GridSearchCV(
            rf,
            search_grid,
            scoring=scoring,
            refit='average_precision',  # Use AUC-PR for final selection
            cv=cv_strategy,
            n_jobs=-1,
            pre_dispatch='2*n_jobs',
            verbose=1
        )

    else:  # randomized search
        grid = RandomizedSearchCV(
            rf,
            param_grid,
            n_iter=n_iter,
            scoring=scoring,
            refit='average_precision',
            cv=cv_strategy,
            n_jobs=-1,
            pre_dispatch='2*n_jobs',
            random_state=42,
            verbose=1
        )

    # Perform search
    print(f"Performing {search_type} search with {cv}-fold CV...")
    grid.fit(X, y)

    # Convert CV results to DataFrame
    cv_results = pd.DataFrame(grid.cv_results_)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV score (AUC-PR): {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, cv_results


def tune_logistic_regression(X, y, cv=5):
    """Tunes a Logistic Regression model (if needed)."""

    # Use solver/penalty combinations that are compatible to avoid deprecation
    # and inconsistent-value warnings from sklearn. Restrict to 'liblinear'
    # which supports both 'l1' and 'l2' penalties.
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ['l1', 'l2'],
        "solver": ['liblinear'],
        "class_weight": ["balanced", None]
    }

    lr = LogisticRegression(random_state=42, max_iter=1000)

    grid = GridSearchCV(
        lr,
        param_grid,
        scoring='average_precision',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    return grid.best_estimator_, grid.best_params_


def perform_comprehensive_cv(model, X, y, thresholds=None, cv=5):
    """
    Performs comprehensive cross-validation with threshold analysis.

    Parameters:
    -----------
    model : sklearn estimator
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Target
    thresholds : list, optional
        Thresholds to evaluate
    cv : int, default=5
        Number of folds

    Returns:
    --------
    cv_results : dict
        Cross-validation results including threshold analysis
    """

    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    results = {
        'fold_scores': [],
        'threshold_results': {thresh: [] for thresh in thresholds},
        'best_thresholds': []
    }

    for train_idx, val_idx in cv_strategy.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Train model
        model_copy = clone_model(model)
        model_copy.fit(X_train_fold, y_train_fold)

        # Get probabilities
        y_proba = model_copy.predict_proba(X_val_fold)[:, 1]

        # Evaluate at different thresholds
        fold_threshold_scores = []
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            f1 = f1_score(y_val_fold, y_pred)
            results['threshold_results'][thresh].append(f1)
            fold_threshold_scores.append((thresh, f1))

        # Find best threshold for this fold
        best_thresh = max(fold_threshold_scores, key=lambda x: x[1])[0]
        results['best_thresholds'].append(best_thresh)

        # Calculate metrics at default threshold
        y_pred_default = model_copy.predict(X_val_fold)
        fold_scores = {
            'f1': f1_score(y_val_fold, y_pred_default),
            'auc_pr': average_precision_score(y_val_fold, y_proba),
            'precision': precision_score(y_val_fold, y_pred_default),
            'recall': recall_score(y_val_fold, y_pred_default)
        }
        results['fold_scores'].append(fold_scores)

    # Calculate summary statistics
    results['mean_f1'] = np.mean([s['f1'] for s in results['fold_scores']])
    results['std_f1'] = np.std([s['f1'] for s in results['fold_scores']])
    results['mean_auc_pr'] = np.mean(
        [s['auc_pr'] for s in results['fold_scores']])
    results['best_threshold'] = np.mean(results['best_thresholds'])

    return results


def clone_model(model):
    """Creates a clone of a model with same parameters."""
    from copy import deepcopy
    return deepcopy(model)
