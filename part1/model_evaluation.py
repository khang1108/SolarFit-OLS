"""
Model evaluation functions.

Functions:
    model_metrics(y, y_hat, p)      -- Calculate RSS, TSS, R², F-statistic
    residual_plots(X, y, beta_hat)  -- Generate 4 diagnostic plots
"""

from dataclasses import dataclass
from typing import List, Tuple
from math import sqrt


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    n: int
    p: int
    rss: float
    tss: float
    ess: float
    r2: float
    r2_adj: float
    sigma2_hat: float
    f_statistic: float
    f_pvalue: float
    rmse: float


def model_metrics(y: List[float], y_hat: List[float], p: int) -> ModelMetrics:
    """
    Calculate model evaluation metrics: RSS, TSS, R², adjusted R², RMSE, F-statistic.

    Parameters:
        y      : List[float]  -- observed values (shape n,)
        y_hat  : List[float]  -- predicted values (shape n,)
        p      : int          -- number of features (excluding intercept)

    Returns:
        ModelMetrics
    """
    n = len(y)
    if len(y_hat) != n:
        raise ValueError("y and y_hat must have same length")

    # Calculate mean of y
    y_mean = sum(y) / n

    # RSS (residual sum of squares)
    residuals = [y[i] - y_hat[i] for i in range(n)]
    rss = sum(e**2 for e in residuals)

    # TSS (total sum of squares)
    tss = sum((y[i] - y_mean)**2 for i in range(n))

    # ESS (explained sum of squares)
    ess = tss - rss

    # R² = 1 - RSS/TSS = ESS/TSS
    r2 = 1.0 - (rss / tss) if tss != 0 else 0.0

    # Adjusted R² = 1 - [RSS/(n-p-1)] / [TSS/(n-1)]
    dof_residual = n - p - 1
    dof_total = n - 1
    if dof_residual > 0 and tss != 0:
        r2_adj = 1.0 - (rss / dof_residual) / (tss / dof_total)
    else:
        r2_adj = r2

    # Sigma² estimate (noise variance)
    if dof_residual > 0:
        sigma2_hat = rss / dof_residual
    else:
        sigma2_hat = float('nan')

    # RMSE (root mean squared error)
    rmse = sqrt(sigma2_hat) if sigma2_hat > 0 else 0.0

    # F-statistic = (TSS - RSS) / p / (RSS / (n - p - 1))
    if dof_residual > 0 and p > 0:
        f_stat = (ess / p) / (rss / dof_residual)
    else:
        f_stat = float('nan')

    # F p-value: would need scipy.stats.f.sf(f_stat, p, dof_residual)
    # For now, store nan
    f_pvalue = float('nan')

    return ModelMetrics(
        n=n,
        p=p,
        rss=rss,
        tss=tss,
        ess=ess,
        r2=r2,
        r2_adj=r2_adj,
        sigma2_hat=sigma2_hat,
        f_statistic=f_stat,
        f_pvalue=f_pvalue,
        rmse=rmse,
    )


@dataclass
class ResidualPlotsData:
    """Container for residual analysis data."""
    residuals: List[float]
    fitted_values: List[float]
    standardized_residuals: List[float]
    qqplot_data: Tuple[List[float], List[float]]


def residual_plots(
    X: List[List[float]],
    y: List[float],
    beta_hat: List[float]
) -> ResidualPlotsData:
    """
    Generate data for 4 diagnostic residual plots:
        1. Residuals vs Fitted values
        2. Q-Q plot (Normal probability plot)
        3. Scale-Location plot (sqrt(|standardized residuals|) vs Fitted)
        4. Residuals vs Leverage (influential points)

    Parameters:
        X        : List[List[float]]  -- design matrix (n x p+1)
        y        : List[float]        -- response vector (n,)
        beta_hat : List[float]        -- OLS coefficients (p+1,)

    Returns:
        ResidualPlotsData with arrays for plotting
    """
    n = len(y)
    if len(beta_hat) != len(X[0]):
        raise ValueError("beta_hat length must match number of columns in X")

    # Calculate fitted values
    fitted = [sum(X[i][j] * beta_hat[j] for j in range(len(beta_hat))) for i in range(n)]

    # Calculate residuals
    residuals = [y[i] - fitted[i] for i in range(n)]

    # Standard deviation of residuals
    residual_mean = sum(residuals) / n
    residual_var = sum((e - residual_mean)**2 for e in residuals) / (n - 1)
    residual_sd = sqrt(residual_var) if residual_var > 0 else 1.0

    # Standardized residuals
    standardized = [e / residual_sd for e in residuals]

    # Sort for Q-Q plot (quantile-quantile)
    sorted_std_residuals = sorted(standardized)
    theoretical_quantiles = _normal_quantiles(n)

    qqplot_data = (theoretical_quantiles, sorted_std_residuals)

    return ResidualPlotsData(
        residuals=residuals,
        fitted_values=fitted,
        standardized_residuals=standardized,
        qqplot_data=qqplot_data,
    )


def _normal_quantiles(n: int) -> List[float]:
    """
    Generate approximate normal quantiles for Q-Q plot.
    Uses simple inverse-normal approximation.
    """
    from math import erfinv, pi
    quantiles = []
    for i in range(1, n + 1):
        p = i / (n + 1)
        if p <= 0 or p >= 1:
            quantiles.append(0.0)
        else:
            try:
                q = sqrt(2) * erfinv(2 * p - 1)
                quantiles.append(q)
            except (ValueError, ZeroDivisionError):
                quantiles.append(0.0)
    return quantiles
