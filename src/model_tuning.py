"""Module for tuning machine learning models using grid search."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def tune_random_forest(X, y, cv=3):
    """Tunes a RandomForestClassifier using grid search."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    grid = GridSearchCV(
        rf,
        param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_
