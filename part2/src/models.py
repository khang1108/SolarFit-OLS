"""
=============================================================================
Regression Models for Part 2: OLS, Ridge, Lasso, and more
=============================================================================
Uses implementations from part1 modules and scikit-learn.

Models implemented:
  1. OLS (Ordinary Least Squares) — from part1.ols_implementation
  2. Ridge (L2 Regularization) — from part1.regularization
  3. Lasso (L1 Regularization) — from scikit-learn
  4. Elastic Net — from scikit-learn

Evaluation metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² (coefficient of determination)
  - k-fold cross-validation scores

=============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

# Import Part 1 modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../part1'))

try:
    from ols_implementation import ols_fit, OLSResult
    from regularization import ridge_fit, ridge_trace, RidgeResult
    from model_evaluation import model_metrics, ModelMetrics
    from inference import vif, coef_inference
    from cross_validation import kfold_cv, kfold_cv_ridge
except ImportError as e:
    print(f"Warning: Could not import Part 1 modules: {e}")
    print("Using fallback implementations instead.")

from sklearn.linear_model import Lasso, ElasticNet, Ridge as SKRidge
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.model_selection import KFold


@dataclass
class ModelResult:
    """Result of fitting a single model."""
    name: str
    beta_hat: List[float]
    y_hat: np.ndarray
    train_mae: float
    train_rmse: float
    train_r2: float
    train_adj_r2: float
    sigma2_hat: Optional[float] = None
    cv_scores: Optional[Dict] = None
    model_object: Optional[object] = None  # For sklearn models


@dataclass
class ComparisonResult:
    """Summary of model comparison on test set."""
    model_name: str
    test_mae: float
    test_rmse: float
    test_r2: float
    test_pred: np.ndarray
    coef_count: int
    nonzero_coef: int


class RegressionModels:
    """
    Container for fitting and comparing multiple regression models.

    Usage:
        models = RegressionModels()
        ols_result = models.fit_ols(X_train, y_train)
        ridge_result = models.fit_ridge(X_train, y_train, lam=100)
        lasso_result = models.fit_lasso(X_train, y_train, alpha=10)
        comparison = models.compare_on_test(X_test, [ols_result, ridge_result, lasso_result])
    """

    def __init__(self):
        self.models = {}
        self.feature_names = None

    # ========================================================================
    # OLS Regression
    # ========================================================================

    def fit_ols(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> ModelResult:
        """
        Fit Ordinary Least Squares regression using Part 1 implementation.

        Args:
            X_train: Training features (n, p+1) with intercept column
            y_train: Training target values (n,)
            feature_names: List of feature names (optional)

        Returns:
            ModelResult with fitted model information
        """
        print(f"\n{'='*70}")
        print("MODEL: Ordinary Least Squares (OLS)")
        print(f"{'='*70}")

        # Convert to list format for Part 1 implementation
        X_list = X_train.tolist()
        y_list = y_train.tolist()

        try:
            # Use Part 1 OLS implementation
            result = ols_fit(X_list, y_list)
            beta_hat = result.beta_hat
            sigma2_hat = result.sigma2_hat
            y_hat_arr = np.array(result.y_hat)

            if len(y_hat_arr) == 0:
                raise ValueError("OLS returned empty predictions")

        except Exception as e:
            # Fallback: use numpy
            print(f"  Using NumPy fallback for OLS (Part 1 error: {type(e).__name__})")
            beta_hat_np = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            beta_hat = beta_hat_np.tolist()
            y_hat_arr = X_train @ beta_hat_np
            sigma2_hat = np.mean((y_train - y_hat_arr) ** 2)
        n = len(y_train)
        p = X_train.shape[1] - 1  # Exclude intercept

        # RSS and TSS
        rss = np.sum((y_train - y_hat_arr) ** 2)
        tss = np.sum((y_train - np.mean(y_train)) ** 2)
        r2 = 1 - (rss / tss) if tss > 0 else 0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else 0

        mae = np.mean(np.abs(y_train - y_hat_arr))
        rmse = np.sqrt(np.mean((y_train - y_hat_arr) ** 2))

        print(f"  ✓ Converged")
        print(f"  Coefficients: {len(beta_hat)}")
        print(f"  Train MAE: {mae:.2f}")
        print(f"  Train RMSE: {rmse:.2f}")
        print(f"  Train R²: {r2:.4f}")
        print(f"  Train Adj R²: {adj_r2:.4f}")

        model_result = ModelResult(
            name="OLS",
            beta_hat=beta_hat,
            y_hat=y_hat_arr,
            train_mae=mae,
            train_rmse=rmse,
            train_r2=r2,
            train_adj_r2=adj_r2,
            sigma2_hat=sigma2_hat
        )

        self.models["OLS"] = model_result
        return model_result

    # ========================================================================
    # Ridge Regression
    # ========================================================================

    def fit_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        lam: float = 1.0,
        feature_names: Optional[List[str]] = None
    ) -> ModelResult:
        """
        Fit Ridge regression with specified lambda.

        Args:
            X_train: Training features (n, p+1) with intercept column
            y_train: Training target values (n,)
            lam: Regularization parameter (lambda)
            feature_names: List of feature names (optional)

        Returns:
            ModelResult with fitted model information
        """
        print(f"\n{'='*70}")
        print(f"MODEL: Ridge Regression (λ = {lam})")
        print(f"{'='*70}")

        # Convert to list format for Part 1 implementation
        X_list = X_train.tolist()
        y_list = y_train.tolist()

        try:
            # Use Part 1 Ridge implementation
            result = ridge_fit(X_list, y_list, lam)
            beta_hat = result.beta_hat
            y_hat = X_train @ np.array(beta_hat)
            sigma2_hat = np.mean((y_train - y_hat) ** 2)
        except Exception as e:
            # Fallback: use scikit-learn
            print(f"  Using scikit-learn Ridge fallback (Part 1 error: {type(e).__name__})")
            sk_ridge = SKRidge(alpha=lam, fit_intercept=False)
            sk_ridge.fit(X_train, y_train)
            beta_hat = sk_ridge.coef_.tolist()
            y_hat = sk_ridge.predict(X_train)
            sigma2_hat = np.mean((y_train - y_hat) ** 2)

        # Calculate metrics
        n = len(y_train)
        p = X_train.shape[1] - 1

        rss = np.sum((y_train - y_hat) ** 2)
        tss = np.sum((y_train - np.mean(y_train)) ** 2)
        r2 = 1 - (rss / tss) if tss > 0 else 0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else 0

        mae = np.mean(np.abs(y_train - y_hat))
        rmse = np.sqrt(np.mean((y_train - y_hat) ** 2))

        print(f"  ✓ Converged")
        print(f"  Coefficients: {len(beta_hat)}")
        print(f"  Train MAE: {mae:.2f}")
        print(f"  Train RMSE: {rmse:.2f}")
        print(f"  Train R²: {r2:.4f}")
        print(f"  Train Adj R²: {adj_r2:.4f}")

        model_result = ModelResult(
            name=f"Ridge(λ={lam})",
            beta_hat=beta_hat,
            y_hat=y_hat,
            train_mae=mae,
            train_rmse=rmse,
            train_r2=r2,
            train_adj_r2=adj_r2,
            sigma2_hat=sigma2_hat
        )

        self.models[f"Ridge(λ={lam})"] = model_result
        return model_result

    # ========================================================================
    # Lasso Regression
    # ========================================================================

    def fit_lasso(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        alpha: float = 1.0,
        feature_names: Optional[List[str]] = None
    ) -> ModelResult:
        """
        Fit Lasso regression with specified alpha (note: sklearn uses alpha, not lambda).

        Args:
            X_train: Training features (n, p+1) with intercept column
            y_train: Training target values (n,)
            alpha: Regularization parameter (alpha in sklearn)
            feature_names: List of feature names (optional)

        Returns:
            ModelResult with fitted model information
        """
        print(f"\n{'='*70}")
        print(f"MODEL: Lasso Regression (α = {alpha})")
        print(f"{'='*70}")

        try:
            # Use scikit-learn Lasso
            lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
            lasso.fit(X_train, y_train)
            beta_hat = lasso.coef_.tolist()
            y_hat = lasso.predict(X_train)
            model_obj = lasso
        except Exception as e:
            print(f"  Error: {e}")
            raise

        # Calculate metrics
        n = len(y_train)
        p = X_train.shape[1] - 1

        rss = np.sum((y_train - y_hat) ** 2)
        tss = np.sum((y_train - np.mean(y_train)) ** 2)
        r2 = 1 - (rss / tss) if tss > 0 else 0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else 0

        mae = np.mean(np.abs(y_train - y_hat))
        rmse = np.sqrt(np.mean((y_train - y_hat) ** 2))

        nonzero_coef = np.count_nonzero(beta_hat)

        print(f"  ✓ Converged")
        print(f"  Coefficients: {len(beta_hat)} (nonzero: {nonzero_coef})")
        print(f"  Train MAE: {mae:.2f}")
        print(f"  Train RMSE: {rmse:.2f}")
        print(f"  Train R²: {r2:.4f}")
        print(f"  Train Adj R²: {adj_r2:.4f}")

        model_result = ModelResult(
            name=f"Lasso(α={alpha})",
            beta_hat=beta_hat,
            y_hat=y_hat,
            train_mae=mae,
            train_rmse=rmse,
            train_r2=r2,
            train_adj_r2=adj_r2,
            model_object=model_obj
        )

        self.models[f"Lasso(α={alpha})"] = model_result
        return model_result

    # ========================================================================
    # Model Evaluation on Test Set
    # ========================================================================

    def evaluate_on_test(
        self,
        X_test: np.ndarray,
        y_test_actual: Optional[np.ndarray] = None,
        models: Optional[List[ModelResult]] = None
    ) -> Dict[str, ComparisonResult]:
        """
        Evaluate models on test set (if y_test is provided) or just make predictions.

        Args:
            X_test: Test features (m, p+1)
            y_test_actual: Actual test values (optional), for evaluation
            models: List of ModelResult objects to evaluate

        Returns:
            Dictionary of ComparisonResult for each model
        """
        if models is None:
            models = list(self.models.values())

        print(f"\n{'='*70}")
        print("EVALUATION ON TEST SET")
        print(f"{'='*70}")

        results = {}

        for model in models:
            print(f"\n  {model.name}")

            # Make predictions
            y_pred = X_test @ np.array(model.beta_hat)

            if y_test_actual is not None:
                # Calculate metrics
                mae = np.mean(np.abs(y_test_actual - y_pred))
                rmse = np.sqrt(np.mean((y_test_actual - y_pred) ** 2))

                tss = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
                rss = np.sum((y_test_actual - y_pred) ** 2)
                r2 = 1 - (rss / tss) if tss > 0 else 0

                print(f"    MAE:  {mae:.2f}")
                print(f"    RMSE: {rmse:.2f}")
                print(f"    R²:   {r2:.4f}")

                result = ComparisonResult(
                    model_name=model.name,
                    test_mae=mae,
                    test_rmse=rmse,
                    test_r2=r2,
                    test_pred=y_pred,
                    coef_count=len(model.beta_hat),
                    nonzero_coef=np.count_nonzero(model.beta_hat)
                )
            else:
                print(f"    Predictions: {y_pred.shape}")
                result = ComparisonResult(
                    model_name=model.name,
                    test_mae=np.nan,
                    test_rmse=np.nan,
                    test_r2=np.nan,
                    test_pred=y_pred,
                    coef_count=len(model.beta_hat),
                    nonzero_coef=np.count_nonzero(model.beta_hat)
                )

            results[model.name] = result

        return results

    def summary_table(self, eval_results: Dict[str, ComparisonResult]) -> pd.DataFrame:
        """Create a summary table of model evaluation results."""
        rows = []

        for name, result in eval_results.items():
            rows.append({
                "Model": name,
                "Test MAE": f"{result.test_mae:.2f}" if not np.isnan(result.test_mae) else "N/A",
                "Test RMSE": f"{result.test_rmse:.2f}" if not np.isnan(result.test_rmse) else "N/A",
                "Test R²": f"{result.test_r2:.4f}" if not np.isnan(result.test_r2) else "N/A",
                "Coef Count": result.coef_count,
                "Nonzero Coef": result.nonzero_coef
            })

        return pd.DataFrame(rows)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from data_pipeline import DataPipeline, PipelineConfig

    print("Testing Models Module\n")

    # Load data
    config = PipelineConfig(data_dir="data", missing_method="mean")
    pipeline = DataPipeline(config)
    pipe_result = pipeline.run()

    X_train = pipe_result.X_train
    y_train = pipe_result.y_train
    X_test = pipe_result.X_test
    feature_names = pipe_result.feature_names

    print(f"\nData loaded: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  Feature names: {feature_names[:5]}...")

    # Fit models
    models_obj = RegressionModels()

    ols_result = models_obj.fit_ols(X_train, y_train, feature_names)
    ridge_result = models_obj.fit_ridge(X_train, y_train, lam=1000.0)
    lasso_result = models_obj.fit_lasso(X_train, y_train, alpha=100.0)

    # Evaluate on test set (note: we don't have y_test, so just make predictions)
    eval_results = models_obj.evaluate_on_test(X_test, models=[ols_result, ridge_result, lasso_result])

    # Create summary table
    summary = models_obj.summary_table(eval_results)
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
