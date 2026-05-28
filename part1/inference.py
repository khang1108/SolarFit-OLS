"""
Coefficient inference functions.

Functions:
    coef_inference(X, y, beta_hat, sigma2)  -- Calculate standard errors, t-stats, p-values, CI
    vif(X)                                   -- Calculate variance inflation factors
"""

from dataclasses import dataclass
from typing import List, Tuple
from math import sqrt


@dataclass
class CoefficientInference:
    """Container for coefficient inference results."""
    coefficients: List[float]
    std_errors: List[float]
    t_statistics: List[float]
    p_values: List[float]
    ci_lower: List[float]
    ci_upper: List[float]
    alpha: float


def coef_inference(
    X: List[List[float]],
    y: List[float],
    beta_hat: List[float],
    sigma2: float,
    alpha: float = 0.05
) -> CoefficientInference:
    """
    Calculate standard errors, t-statistics, p-values, and confidence intervals.

    Uses the formula:
        se(beta_j) = sigma_hat * sqrt((X'X)^{-1}_{jj})
        t_j = beta_j / se(beta_j)  ~ t_{n-p-1}
        CI_j = beta_j ± t_{alpha/2, n-p-1} * se(beta_j)

    Parameters:
        X        : List[List[float]]  -- design matrix (n x p+1)
        y        : List[float]        -- response vector (n,)
        beta_hat : List[float]        -- OLS coefficients (p+1,)
        sigma2   : float              -- estimated noise variance
        alpha    : float              -- significance level (default 0.05)

    Returns:
        CoefficientInference
    """
    n = len(y)
    p = len(beta_hat) - 1  # number of features (excluding intercept)

    # Compute X'X and its inverse
    k = len(X[0])
    XtX = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            XtX[i][j] = sum(X[row][i] * X[row][j] for row in range(n))

    # Invert X'X (simplified, using numpy-like approach)
    try:
        XtX_inv = _matrix_inverse(XtX)
    except:
        return CoefficientInference(
            coefficients=beta_hat,
            std_errors=[float('nan')] * len(beta_hat),
            t_statistics=[float('nan')] * len(beta_hat),
            p_values=[float('nan')] * len(beta_hat),
            ci_lower=[float('nan')] * len(beta_hat),
            ci_upper=[float('nan')] * len(beta_hat),
            alpha=alpha,
        )

    # Calculate standard errors
    sigma_hat = sqrt(sigma2) if sigma2 > 0 else 0.0
    std_errors = [sigma_hat * sqrt(max(0, XtX_inv[i][i])) for i in range(k)]

    # Calculate t-statistics
    t_stats = []
    for i in range(k):
        if std_errors[i] > 0:
            t_stats.append(beta_hat[i] / std_errors[i])
        else:
            t_stats.append(float('nan'))

    # Calculate critical t-value (approximate using normal for large n)
    dof = n - p - 1
    t_crit = _t_critical(alpha / 2, dof)

    # Calculate confidence intervals
    ci_lower = []
    ci_upper = []
    for i in range(k):
        margin = t_crit * std_errors[i]
        ci_lower.append(beta_hat[i] - margin)
        ci_upper.append(beta_hat[i] + margin)

    # P-values (approximated using t-distribution)
    p_values = [_t_pvalue(abs(t_stats[i]), dof) for i in range(k)]

    return CoefficientInference(
        coefficients=beta_hat,
        std_errors=std_errors,
        t_statistics=t_stats,
        p_values=p_values,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha,
    )


@dataclass
class VIFResult:
    """Container for VIF analysis."""
    vif_values: List[float]
    column_names: List[str]
    max_vif: float
    has_multicollinearity: bool


def vif(X: List[List[float]], threshold: float = 10.0) -> VIFResult:
    """
    Calculate Variance Inflation Factors (VIF) for each variable.

    VIF_j = 1 / (1 - R²_j) where R²_j is from regressing X_j on other X's.
    VIF > 10 suggests multicollinearity.

    Parameters:
        X         : List[List[float]]  -- design matrix (n x p+1)
        threshold : float              -- VIF threshold for multicollinearity (default 10)

    Returns:
        VIFResult
    """
    n = len(X)
    p = len(X[0])

    vif_values = []
    for j in range(p):
        # R² from regressing X_j on others
        r2_j = _calculate_r2_excluding_col(X, j)
        vif_j = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else float('inf')
        vif_values.append(vif_j)

    max_vif = max(v for v in vif_values if v != float('inf'))
    has_multicollinearity = max_vif > threshold

    col_names = [f"X{i}" for i in range(p)]

    return VIFResult(
        vif_values=vif_values,
        column_names=col_names,
        max_vif=max_vif,
        has_multicollinearity=has_multicollinearity,
    )


def _matrix_inverse(A: List[List[float]]) -> List[List[float]]:
    """
    Compute matrix inverse using Gauss-Jordan elimination.
    Simple implementation for small matrices.
    """
    n = len(A)
    # Create augmented matrix [A | I]
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for col in range(n):
        # Find pivot
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError(f"Singular matrix at column {col}")

        # Swap rows
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Scale pivot row
        pivot_val = aug[col][col]
        aug[col] = [v / pivot_val for v in aug[col]]

        # Eliminate column
        for row in range(n):
            if row != col:
                f = aug[row][col]
                aug[row] = [aug[row][j] - f * aug[col][j] for j in range(2 * n)]

    return [row[n:] for row in aug]


def _calculate_r2_excluding_col(X: List[List[float]], exclude_col: int) -> float:
    """
    Regress column exclude_col on all other columns, return R².
    Simplified version using least squares.
    """
    try:
        import numpy as np
        X_arr = np.array(X, dtype=float)
        y_col = X_arr[:, exclude_col]
        X_others = np.delete(X_arr, exclude_col, axis=1)

        # Simple OLS
        try:
            beta = np.linalg.lstsq(X_others, y_col, rcond=None)[0]
            y_pred = X_others @ beta
            ss_res = np.sum((y_col - y_pred)**2)
            ss_tot = np.sum((y_col - np.mean(y_col))**2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            return float(r2)
        except:
            return 0.0
    except ImportError:
        return 0.0


def _t_critical(alpha_half: float, dof: int) -> float:
    """
    Approximate critical t-value for confidence interval.
    For large dof, approximates to normal distribution.
    """
    try:
        from scipy import stats
        return float(stats.t.ppf(1 - alpha_half, dof))
    except:
        # Fallback: approximate with normal (good for dof > 30)
        if dof > 30:
            return 1.96  # 95% CI
        else:
            # Simple approximation
            return 2.0 + 4.0 / dof


def _t_pvalue(t_stat: float, dof: int) -> float:
    """
    Calculate two-sided p-value for t-statistic.
    """
    try:
        from scipy import stats
        return float(2 * (1 - stats.t.cdf(abs(t_stat), dof)))
    except:
        # Fallback: no p-value available
        return float('nan')
