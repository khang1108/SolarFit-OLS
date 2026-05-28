"""
=============================================================================
Model Evaluation: Cross-Validation and Test Set Metrics
=============================================================================
Provides functions for:
  1. Train-Validation split
  2. k-fold cross-validation
  3. Hyperparameter tuning (lambda selection for Ridge/Lasso)
  4. Feature importance analysis
  5. Residual diagnostics

=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class CrossValidationResult:
    """Results from k-fold cross-validation."""
    model_name: str
    fold_mae: List[float]
    fold_rmse: List[float]
    fold_r2: List[float]
    mean_mae: float
    std_mae: float
    mean_rmse: float
    std_rmse: float
    mean_r2: float
    std_r2: float


@dataclass
class FeatureImportanceResult:
    """Feature importance ranking."""
    feature_names: List[str]
    coefficients: np.ndarray
    abs_coefficients: np.ndarray
    ranking: List[Tuple[str, float]]  # (feature_name, abs_coef)


class ModelEvaluator:
    """
    Comprehensive evaluation tools for regression models.

    Usage:
        evaluator = ModelEvaluator()
        cv_result = evaluator.kfold_cv(X_train, y_train, k=5, fit_func=fit_ols)
        fi_result = evaluator.feature_importance(feature_names, coef)
    """

    @staticmethod
    def train_val_split(
        X: np.ndarray,
        y: np.ndarray,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and validation sets.

        Args:
            X: Feature matrix (n, p)
            y: Target vector (n,)
            val_size: Proportion for validation (0-1)
            random_state: Random seed

        Returns:
            X_train, X_val, y_train, y_val
        """
        np.random.seed(random_state)
        n = X.shape[0]
        val_size_n = int(n * val_size)

        indices = np.arange(n)
        np.random.shuffle(indices)

        train_idx = indices[val_size_n:]
        val_idx = indices[:val_size_n]

        return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

    @staticmethod
    def kfold_split(
        n: int,
        k: int = 5,
        random_state: int = 42
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate k-fold train/test index pairs.

        Args:
            n: Number of samples
            k: Number of folds
            random_state: Random seed

        Returns:
            List of (train_indices, test_indices) tuples
        """
        np.random.seed(random_state)
        fold_size = n // k
        indices = np.arange(n)
        np.random.shuffle(indices)

        folds = []
        for i in range(k):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < k - 1 else n

            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

            folds.append((train_idx, test_idx))

        return folds

    @staticmethod
    def kfold_cv(
        X: np.ndarray,
        y: np.ndarray,
        fit_func: Callable,
        k: int = 5,
        model_name: str = "Model",
        random_state: int = 42
    ) -> CrossValidationResult:
        """
        Perform k-fold cross-validation.

        Args:
            X: Feature matrix (n, p)
            y: Target vector (n,)
            fit_func: Function that takes (X_train, y_train) and returns y_pred function
            k: Number of folds
            model_name: Name of model for reporting
            random_state: Random seed

        Returns:
            CrossValidationResult with per-fold and aggregate metrics
        """
        folds = ModelEvaluator.kfold_split(X.shape[0], k, random_state)

        fold_mae = []
        fold_rmse = []
        fold_r2 = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit and predict
            beta = fit_func(X_train, y_train)
            y_pred = X_test @ beta

            # Calculate metrics
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            tss = np.sum((y_test - np.mean(y_test)) ** 2)
            rss = np.sum((y_test - y_pred) ** 2)
            r2 = 1 - (rss / tss) if tss > 0 else 0

            fold_mae.append(mae)
            fold_rmse.append(rmse)
            fold_r2.append(r2)

        return CrossValidationResult(
            model_name=model_name,
            fold_mae=fold_mae,
            fold_rmse=fold_rmse,
            fold_r2=fold_r2,
            mean_mae=np.mean(fold_mae),
            std_mae=np.std(fold_mae),
            mean_rmse=np.mean(fold_rmse),
            std_rmse=np.std(fold_rmse),
            mean_r2=np.mean(fold_r2),
            std_r2=np.std(fold_r2)
        )

    @staticmethod
    def feature_importance(
        feature_names: List[str],
        coefficients: np.ndarray,
        top_n: int = 20
    ) -> FeatureImportanceResult:
        """
        Rank features by absolute coefficient magnitude.

        Args:
            feature_names: List of feature names
            coefficients: Coefficient vector (p,)
            top_n: Number of top features to return

        Returns:
            FeatureImportanceResult with ranking
        """
        abs_coef = np.abs(coefficients)
        ranking = sorted(
            zip(feature_names, abs_coef),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return FeatureImportanceResult(
            feature_names=feature_names,
            coefficients=coefficients,
            abs_coefficients=abs_coef,
            ranking=ranking
        )

    @staticmethod
    def residual_diagnostics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute residual diagnostics.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            X: Feature matrix (optional, for VIF calculation)

        Returns:
            Dictionary with diagnostic metrics
        """
        residuals = y_true - y_pred

        diagnostics = {
            "residuals": residuals,
            "mean_residual": np.mean(residuals),
            "std_residual": np.std(residuals),
            "min_residual": np.min(residuals),
            "max_residual": np.max(residuals),
            "autocorr_lag1": np.corrcoef(residuals[:-1], residuals[1:])[0, 1],
            "normality_test": _shapiro_test(residuals),
        }

        return diagnostics

    @staticmethod
    def hyperparameter_tuning(
        X: np.ndarray,
        y: np.ndarray,
        fit_func: Callable,
        param_name: str,
        param_values: List[float],
        k: int = 5,
        metric: str = "r2"
    ) -> Tuple[float, List[float]]:
        """
        Find best hyperparameter value using k-fold CV.

        Args:
            X: Feature matrix
            y: Target vector
            fit_func: Function that takes (X, y, param_value) and returns y_pred function
            param_name: Name of parameter (for reporting)
            param_values: List of parameter values to try
            k: Number of folds
            metric: Metric to optimize ("mae", "rmse", "r2")

        Returns:
            (best_param, scores_per_param)
        """
        scores_per_param = []
        best_score = -np.inf if metric == "r2" else np.inf
        best_param = param_values[0]

        folds = ModelEvaluator.kfold_split(X.shape[0], k)

        for param_val in param_values:
            fold_scores = []

            for train_idx, test_idx in folds:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Fit and predict
                beta = fit_func(X_train, y_train, param_val)
                y_pred = X_test @ beta

                # Calculate metric
                if metric == "r2":
                    tss = np.sum((y_test - np.mean(y_test)) ** 2)
                    rss = np.sum((y_test - y_pred) ** 2)
                    score = 1 - (rss / tss) if tss > 0 else 0
                elif metric == "rmse":
                    score = np.sqrt(np.mean((y_test - y_pred) ** 2))
                else:  # mae
                    score = np.mean(np.abs(y_test - y_pred))

                fold_scores.append(score)

            mean_score = np.mean(fold_scores)
            scores_per_param.append(mean_score)

            # Update best
            if metric == "r2":
                if mean_score > best_score:
                    best_score = mean_score
                    best_param = param_val
            else:
                if mean_score < best_score:
                    best_score = mean_score
                    best_param = param_val

        return best_param, scores_per_param


def _shapiro_test(residuals: np.ndarray) -> Dict:
    """Simplified Shapiro-Wilk test for normality."""
    from scipy import stats

    try:
        stat, pvalue = stats.shapiro(residuals[:min(5000, len(residuals))])
        return {"statistic": stat, "p_value": pvalue, "normal": pvalue > 0.05}
    except:
        return {"statistic": np.nan, "p_value": np.nan, "normal": None}


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from data_pipeline import DataPipeline, PipelineConfig
    from models import RegressionModels

    print("Testing Evaluator Module\n")

    # Load data
    config = PipelineConfig(data_dir="data", missing_method="mean")
    pipeline = DataPipeline(config)
    pipe_result = pipeline.run()

    X_train = pipe_result.X_train
    y_train = pipe_result.y_train

    # Define fit functions for CV
    def fit_ols(X, y):
        from numpy.linalg import lstsq
        return lstsq(X, y, rcond=None)[0]

    def fit_ridge(X, y, lam):
        from numpy.linalg import inv
        return inv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y

    # Test k-fold CV
    print("=" * 70)
    print("K-FOLD CROSS-VALIDATION (k=5)")
    print("=" * 70)

    cv_ols = ModelEvaluator.kfold_cv(
        X_train, y_train,
        fit_func=fit_ols,
        k=5,
        model_name="OLS"
    )

    print(f"\n{cv_ols.model_name}:")
    print(f"  MAE:  {cv_ols.mean_mae:.2f} ± {cv_ols.std_mae:.2f}")
    print(f"  RMSE: {cv_ols.mean_rmse:.2f} ± {cv_ols.std_rmse:.2f}")
    print(f"  R²:   {cv_ols.mean_r2:.4f} ± {cv_ols.std_r2:.4f}")

    # Test feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Top 20)")
    print("=" * 70)

    beta_ols = fit_ols(X_train, y_train)
    feature_names = pipe_result.feature_names
    fi_result = ModelEvaluator.feature_importance(feature_names, beta_ols, top_n=20)

    print("\nTop 20 features by |coefficient|:")
    for i, (feat, coef) in enumerate(fi_result.ranking, 1):
        print(f"  {i:2d}. {feat:40s}: {coef:12.2f}")

    # Test hyperparameter tuning
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING: Ridge λ selection")
    print("=" * 70)

    lambdas = [1, 10, 100, 1000, 10000]
    best_lam, scores = ModelEvaluator.hyperparameter_tuning(
        X_train, y_train,
        fit_func=fit_ridge,
        param_name="lambda",
        param_values=lambdas,
        k=5,
        metric="r2"
    )

    print(f"\n  Best λ: {best_lam}")
    print(f"  CV Scores (R²):")
    for lam, score in zip(lambdas, scores):
        print(f"    λ={lam:5d}: R²={score:.4f}")
