"""Kiểm thử ridge_fit và lasso_fit."""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "part1"))

from ols_implementation import ols_fit  # pyright: ignore[reportMissingImports]
from regularization import lasso_fit, ridge_fit  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data(seed: int = 3):
    rng = np.random.default_rng(seed)
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 2.0 + 1.5 * x1 - x2 + 0.5 * rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2, x3]).tolist()
    return X, y.tolist()


# ---------------------------------------------------------------------------
# ridge_fit
# ---------------------------------------------------------------------------

def test_ridge_fit_lambda0_converges_to_ols():
    """λ=0: Ridge = OLS (sai số < 1e-8)."""
    X, y = _data()
    ols = ols_fit(X, y)
    ridge = ridge_fit(X, y, lam=0.0)
    for b_ols, b_ridge in zip(ols.beta_hat, ridge.coefficients):
        assert abs(b_ols - b_ridge) < 1e-8


def test_ridge_fit_intercept_not_penalized():
    """ridge_penalty không bao gồm hệ số tự do (beta[0])."""
    X, y = _data()
    lam = 5.0
    result = ridge_fit(X, y, lam=lam)
    # Penalty chỉ trên beta[1:]
    expected_penalty = sum(b ** 2 for b in result.coefficients[1:])
    assert abs(result.ridge_penalty - expected_penalty) < 1e-10


def test_ridge_fit_larger_lambda_shrinks_coefficients():
    """λ lớn hơn → chuẩn hệ số nhỏ hơn (shrinkage)."""
    X, y = _data()
    coef_small = ridge_fit(X, y, lam=0.01).coefficients[1:]
    coef_large = ridge_fit(X, y, lam=100.0).coefficients[1:]
    norm_small = sum(b ** 2 for b in coef_small)
    norm_large = sum(b ** 2 for b in coef_large)
    assert norm_large < norm_small


def test_ridge_fit_matches_numpy_closed_form():
    """So sánh với công thức đóng NumPy làm oracle: max error < 1e-8.

    Scratch không phạt hệ số tự do → P = diag(0, 1, ..., 1).
    Oracle: (XᵀX + λP)⁻¹Xᵀy tính trực tiếp bằng numpy.
    """
    X, y = _data()
    lam = 2.0
    result = ridge_fit(X, y, lam=lam)

    Xnp = np.array(X)
    ynp = np.array(y)
    p = Xnp.shape[1]
    P = np.diag([0.0] + [1.0] * (p - 1))  # không phạt hệ số tự do
    beta_oracle = np.linalg.solve(Xnp.T @ Xnp + lam * P, Xnp.T @ ynp)
    for b_s, b_ref in zip(result.coefficients, beta_oracle):
        assert abs(b_s - b_ref) < 1e-8


# ---------------------------------------------------------------------------
# lasso_fit
# ---------------------------------------------------------------------------

def test_lasso_fit_large_lambda_gives_zero_coefficients():
    """λ đủ lớn: tất cả hệ số hồi quy = 0 (hệ số tự do không bị ảnh hưởng)."""
    X, y = _data()
    result = lasso_fit(X, y, lam=1e6)
    for b in result.coefficients[1:]:
        assert abs(b) < 1e-6


def test_lasso_fit_lambda0_near_ols():
    """λ=0: Lasso ≈ OLS (sai số < 1e-4 do coordinate descent tolerance)."""
    X, y = _data()
    ols = ols_fit(X, y)
    lasso = lasso_fit(X, y, lam=0.0)
    for b_ols, b_lasso in zip(ols.beta_hat, lasso.coefficients):
        assert abs(b_ols - b_lasso) < 1e-4


def test_lasso_fit_sparsity_increases_with_lambda():
    """Lambda tăng → số hệ số khác không giảm (hoặc giữ nguyên)."""
    X, y = _data()
    nonzero_small = sum(1 for b in lasso_fit(X, y, lam=0.1).coefficients[1:] if abs(b) > 1e-8)
    nonzero_large = sum(1 for b in lasso_fit(X, y, lam=5.0).coefficients[1:] if abs(b) > 1e-8)
    assert nonzero_large <= nonzero_small


def test_lasso_fit_matches_sklearn_approximate():
    """So sánh hệ số Lasso scratch vs sklearn: max error < 0.01."""
    X, y = _data()
    lam = 1.0
    result = lasso_fit(X, y, lam=lam)

    Xnp = np.array(X)
    ynp = np.array(y)
    # sklearn Lasso tối thiểu (1/(2n))*RSS + alpha*||β||₁
    # scratch tối thiểu RSS + lam*||β||₁ → alpha_sklearn = lam / (2n)
    n = len(y)
    sk = SklearnLasso(alpha=lam / (2 * n), fit_intercept=True, max_iter=10000, tol=1e-8)
    sk.fit(Xnp[:, 1:], ynp)  # sklearn tự thêm hệ số tự do, bỏ cột 1
    sk_coef = [sk.intercept_] + list(sk.coef_)
    for b_s, b_sk in zip(result.coefficients, sk_coef):
        assert abs(b_s - b_sk) < 0.01
