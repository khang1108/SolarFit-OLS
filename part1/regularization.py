"""
Regularization methods: Ridge and Lasso regression.

Functions:
    ridge_fit(X, y, lam)  -- Ridge regression with regularization parameter lambda
    ridge_trace(X, y, lambdas)  -- Compute ridge coefficients across lambda range
"""

from dataclasses import dataclass
from typing import List, Tuple
from math import sqrt


@dataclass
class RidgeResult:
    """Container for ridge regression results."""
    coefficients: List[float]
    lambda_val: float
    rss: float
    ridge_penalty: float
    total_loss: float
    success: bool
    message: str


def ridge_fit(X: List[List[float]], y: List[float], lam: float) -> RidgeResult:
    """
    Fit Ridge regression: minimize ||y - X*beta||² + lambda * ||beta||²

    Ridge solution: beta_ridge = (X'X + lambda*I)^{-1} X'y

    Parameters:
        X   : List[List[float]]  -- design matrix (n x p+1)
        y   : List[float]        -- response vector (n,)
        lam : float              -- regularization parameter (lambda >= 0)

    Returns:
        RidgeResult
    """
    if lam < 0:
        return RidgeResult(
            coefficients=[],
            lambda_val=lam,
            rss=float('nan'),
            ridge_penalty=float('nan'),
            total_loss=float('nan'),
            success=False,
            message="lambda must be non-negative",
        )

    n = len(y)
    p = len(X[0])

    try:
        # Compute X'X
        XtX = [[0.0] * p for _ in range(p)]
        for i in range(p):
            for j in range(p):
                XtX[i][j] = sum(X[row][i] * X[row][j] for row in range(n))

        # Add lambda*I to diagonal: X'X + lambda*I
        XtX_ridge = [XtX[i][:] for i in range(p)]
        for i in range(p):
            XtX_ridge[i][i] += lam

        # Compute X'y
        Xty = [sum(X[row][i] * y[row] for row in range(n)) for i in range(p)]

        # Solve (X'X + lambda*I) beta = X'y
        beta_ridge = _solve_linear_system(XtX_ridge, Xty)

        # Calculate metrics
        y_pred = [sum(X[i][j] * beta_ridge[j] for j in range(p)) for i in range(n)]
        residuals = [y[i] - y_pred[i] for i in range(n)]
        rss = sum(e**2 for e in residuals)
        ridge_penalty = sum(b**2 for b in beta_ridge[1:])  # exclude intercept
        total_loss = rss + lam * ridge_penalty

        return RidgeResult(
            coefficients=beta_ridge,
            lambda_val=lam,
            rss=rss,
            ridge_penalty=ridge_penalty,
            total_loss=total_loss,
            success=True,
            message=f"Ridge solved with lambda={lam}. RSS={rss:.6g}, penalty={ridge_penalty:.6g}",
        )

    except Exception as exc:
        return RidgeResult(
            coefficients=[],
            lambda_val=lam,
            rss=float('nan'),
            ridge_penalty=float('nan'),
            total_loss=float('nan'),
            success=False,
            message=f"ridge_fit failed: {exc}",
        )


@dataclass
class RidgeTraceData:
    """Container for ridge trace data."""
    lambdas: List[float]
    coefficients_trace: List[List[float]]
    rss_trace: List[float]
    gcv_trace: List[float]


def ridge_trace(X: List[List[float]], y: List[float], lambdas: List[float]) -> RidgeTraceData:
    """
    Compute ridge regression coefficients across a range of lambda values.

    Parameters:
        X       : List[List[float]]  -- design matrix (n x p+1)
        y       : List[float]        -- response vector (n,)
        lambdas : List[float]        -- range of lambda values to evaluate

    Returns:
        RidgeTraceData with coefficients and RSS for each lambda
    """
    coefficients_trace = []
    rss_trace = []
    gcv_trace = []

    for lam in lambdas:
        result = ridge_fit(X, y, lam)
        if result.success:
            coefficients_trace.append(result.coefficients)
            rss_trace.append(result.rss)
            # GCV (Generalized Cross-Validation): RSS / (1 - tr(H_ridge)/n)^2
            # Simplified approximation
            gcv = result.rss  # placeholder
            gcv_trace.append(gcv)
        else:
            coefficients_trace.append([])
            rss_trace.append(float('nan'))
            gcv_trace.append(float('nan'))

    return RidgeTraceData(
        lambdas=lambdas,
        coefficients_trace=coefficients_trace,
        rss_trace=rss_trace,
        gcv_trace=gcv_trace,
    )


def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve linear system Ax = b using Gauss elimination.
    Simple implementation suitable for small systems.
    """
    n = len(A)
    if len(b) != n:
        raise ValueError("A and b dimensions don't match")

    # Make copies
    mat = [A[i][:] for i in range(n)]
    rhs = b[:]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        pivot_row = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot_row][col]) < 1e-12:
            raise ValueError(f"Singular matrix at column {col}")

        # Swap rows
        mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
        rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]

        # Eliminate below
        for row in range(col + 1, n):
            if abs(mat[col][col]) < 1e-12:
                continue
            f = mat[row][col] / mat[col][col]
            for j in range(col, n):
                mat[row][j] -= f * mat[col][j]
            rhs[row] -= f * rhs[col]

    # Back substitution
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        x[row] = rhs[row]
        for col in range(row + 1, n):
            x[row] -= mat[row][col] * x[col]
        if abs(mat[row][row]) > 1e-12:
            x[row] /= mat[row][row]

    return x
