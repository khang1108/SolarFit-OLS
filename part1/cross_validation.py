"""
Cross-validation functions.

Functions:
    kfold_cv(X, y, k)          -- k-fold cross-validation with OLS
    kfold_cv_ridge(X, y, lam, k)  -- k-fold CV for Ridge regression
    model_selection_cv(X, y, k, models)  -- Compare models via k-fold CV
"""

from dataclasses import dataclass
from typing import List, Callable, Dict
from math import sqrt
import random


@dataclass
class CVResult:
    """Container for cross-validation results."""
    k: int
    model_name: str
    cv_scores: List[float]
    mean_cv_score: float
    std_cv_score: float
    train_scores: List[float]
    test_scores: List[float]


def kfold_cv(X: List[List[float]], y: List[float], k: int = 5, metric: str = "mse") -> CVResult:
    """
    Perform k-fold cross-validation for OLS regression.

    Parameters:
        X      : List[List[float]]  -- design matrix (n x p+1)
        y      : List[float]        -- response vector (n,)
        k      : int                -- number of folds (default 5)
        metric : str                -- evaluation metric: "mse", "rmse", "mae", "r2"

    Returns:
        CVResult with CV scores for each fold
    """
    n = len(y)
    if k > n or k < 2:
        raise ValueError(f"k must be between 2 and {n}")

    # Split data into k folds
    fold_indices = _stratified_k_fold(n, k)
    cv_scores = []
    train_scores = []
    test_scores = []

    for fold_idx, test_idx in enumerate(fold_indices):
        # Create train/test split
        X_train, y_train = [], []
        X_test, y_test = [], []

        test_set = set(test_idx)
        for i in range(n):
            if i in test_set:
                X_test.append(X[i])
                y_test.append(y[i])
            else:
                X_train.append(X[i])
                y_train.append(y[i])

        # Fit OLS on training set
        try:
            from ols_implementation import ols_fit
            ols_result = ols_fit(X_train, y_train)
            if not ols_result.success:
                cv_scores.append(float('nan'))
                continue

            # Evaluate on test set
            y_pred_test = ols_result.y_hat
            score_test = _calculate_metric(y_test, y_pred_test, metric)
            test_scores.append(score_test)

            # Evaluate on train set
            score_train = _calculate_metric(y_train, ols_result.y_hat, metric)
            train_scores.append(score_train)

            cv_scores.append(score_test)

        except Exception as e:
            print(f"Fold {fold_idx}: Error - {e}")
            cv_scores.append(float('nan'))

    # Calculate statistics
    valid_scores = [s for s in cv_scores if s == s]  # Remove NaN
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else float('nan')
    if len(valid_scores) > 1:
        var = sum((s - mean_score)**2 for s in valid_scores) / (len(valid_scores) - 1)
        std_score = sqrt(var)
    else:
        std_score = float('nan')

    return CVResult(
        k=k,
        model_name="OLS",
        cv_scores=cv_scores,
        mean_cv_score=mean_score,
        std_cv_score=std_score,
        train_scores=train_scores,
        test_scores=test_scores,
    )


def kfold_cv_ridge(
    X: List[List[float]],
    y: List[float],
    lam: float,
    k: int = 5,
    metric: str = "mse"
) -> CVResult:
    """
    Perform k-fold cross-validation for Ridge regression.

    Parameters:
        X      : List[List[float]]  -- design matrix (n x p+1)
        y      : List[float]        -- response vector (n,)
        lam    : float              -- ridge regularization parameter
        k      : int                -- number of folds (default 5)
        metric : str                -- evaluation metric

    Returns:
        CVResult for Ridge with given lambda
    """
    n = len(y)
    if k > n or k < 2:
        raise ValueError(f"k must be between 2 and {n}")

    fold_indices = _stratified_k_fold(n, k)
    cv_scores = []

    for fold_idx, test_idx in enumerate(fold_indices):
        X_train, y_train = [], []
        X_test, y_test = [], []

        test_set = set(test_idx)
        for i in range(n):
            if i in test_set:
                X_test.append(X[i])
                y_test.append(y[i])
            else:
                X_train.append(X[i])
                y_train.append(y[i])

        try:
            from regularization import ridge_fit
            ridge_result = ridge_fit(X_train, y_train, lam)
            if not ridge_result.success:
                cv_scores.append(float('nan'))
                continue

            # Predict on test set
            y_pred_test = [sum(X_test[i][j] * ridge_result.coefficients[j]
                              for j in range(len(ridge_result.coefficients)))
                          for i in range(len(X_test))]

            score = _calculate_metric(y_test, y_pred_test, metric)
            cv_scores.append(score)

        except Exception as e:
            cv_scores.append(float('nan'))

    valid_scores = [s for s in cv_scores if s == s]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else float('nan')
    var = sum((s - mean_score)**2 for s in valid_scores) / (len(valid_scores) - 1) \
        if len(valid_scores) > 1 else float('nan')
    std_score = sqrt(var) if var == var else float('nan')

    return CVResult(
        k=k,
        model_name=f"Ridge(lambda={lam})",
        cv_scores=cv_scores,
        mean_cv_score=mean_score,
        std_cv_score=std_score,
        train_scores=[],
        test_scores=[],
    )


def _stratified_k_fold(n: int, k: int) -> List[List[int]]:
    """
    Create k-fold indices. Simple random split.
    """
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        folds.append(indices[start:end])

    return folds


def _calculate_metric(y_true: List[float], y_pred: List[float], metric: str) -> float:
    """
    Calculate evaluation metric.
    """
    if len(y_true) != len(y_pred):
        return float('nan')

    n = len(y_true)

    if metric == "mse":
        return sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n

    elif metric == "rmse":
        mse = sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n
        return sqrt(mse)

    elif metric == "mae":
        return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

    elif metric == "r2":
        y_mean = sum(y_true) / n
        ss_tot = sum((y_true[i] - y_mean)**2 for i in range(n))
        ss_res = sum((y_true[i] - y_pred[i])**2 for i in range(n))
        if ss_tot != 0:
            return 1.0 - (ss_res / ss_tot)
        else:
            return 0.0

    else:
        raise ValueError(f"Unknown metric: {metric}")


@dataclass
class ModelComparisonResult:
    """Container for comparing multiple models via CV."""
    models: Dict[str, CVResult]
    best_model: str
    best_score: float


def model_selection_cv(
    X: List[List[float]],
    y: List[float],
    k: int = 5,
    models: Dict[str, Callable] = None
) -> ModelComparisonResult:
    """
    Compare multiple models using k-fold cross-validation.

    Parameters:
        X      : List[List[float]]  -- design matrix
        y      : List[float]        -- response vector
        k      : int                -- number of folds
        models : Dict[str, Callable] -- dict of model name -> function

    Returns:
        ModelComparisonResult with best model
    """
    if models is None:
        models = {
            "OLS": lambda X, y: kfold_cv(X, y, k, metric="rmse"),
            "Ridge(lam=0.1)": lambda X, y: kfold_cv_ridge(X, y, 0.1, k, metric="rmse"),
            "Ridge(lam=1.0)": lambda X, y: kfold_cv_ridge(X, y, 1.0, k, metric="rmse"),
        }

    results = {}
    for name, model_func in models.items():
        try:
            results[name] = model_func(X, y)
        except Exception as e:
            print(f"Error with {name}: {e}")

    # Find best model (lowest mean CV score)
    best_model = min(results.keys(), key=lambda m: results[m].mean_cv_score)
    best_score = results[best_model].mean_cv_score

    return ModelComparisonResult(
        models=results,
        best_model=best_model,
        best_score=best_score,
    )
