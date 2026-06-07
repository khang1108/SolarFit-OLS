"""Kiểm thử ols_fit và hat_matrix."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "part1"))

from ols_implementation import hat_matrix, ols_fit  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_xy():
    """y = 2 + 3*x, n=20, nghiệm biết trước."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 20)
    X = np.column_stack([np.ones(20), x]).tolist()
    y = (2.0 + 3.0 * x).tolist()
    return X, y


def _noisy_xy(seed: int = 42):
    """y = 1 + 2*x1 - x2 + noise, n=50."""
    rng = np.random.default_rng(seed)
    n = 50
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - x2 + 0.1 * rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2]).tolist()
    return X, y.tolist()


# ---------------------------------------------------------------------------
# ols_fit
# ---------------------------------------------------------------------------

def test_ols_fit_known_coefficients():
    """Với dữ liệu không có nhiễu, beta_hat phải khớp chính xác."""
    X, y = _simple_xy()
    result = ols_fit(X, y)
    assert result.success
    assert abs(result.beta_hat[0] - 2.0) < 1e-8
    assert abs(result.beta_hat[1] - 3.0) < 1e-8


def test_ols_fit_zero_residuals_perfect_fit():
    """Dữ liệu không có nhiễu: RSS phải bằng 0 và residuals ≈ 0."""
    X, y = _simple_xy()
    result = ols_fit(X, y)
    assert result.rss < 1e-15
    assert all(abs(e) < 1e-8 for e in result.residuals)


def test_ols_fit_residuals_orthogonal_to_X():
    """Normal Equations đảm bảo X'e = 0."""
    X, y = _noisy_xy()
    result = ols_fit(X, y)
    Xnp = np.array(X)
    e = np.array(result.residuals)
    xte = Xnp.T @ e
    assert np.max(np.abs(xte)) < 1e-8


def test_ols_fit_matches_numpy_lstsq():
    """So sánh beta_hat với numpy.linalg.lstsq làm oracle."""
    X, y = _noisy_xy()
    result = ols_fit(X, y)
    beta_np, _, _, _ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    for b_scratch, b_np in zip(result.beta_hat, beta_np):
        assert abs(b_scratch - b_np) < 1e-8


def test_ols_fit_sigma2_unbiased():
    """sigma2_hat = RSS / (n - p - 1): kiểm tra mẫu số đúng."""
    X, y = _noisy_xy()
    result = ols_fit(X, y)
    n = len(y)
    p = len(result.beta_hat) - 1
    expected = result.rss / (n - p - 1)
    assert abs(result.sigma2_hat - expected) < 1e-12


def test_ols_fit_singular_matrix_fails_gracefully():
    """Hai cột giống nhau → X'X suy biến → success=False."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=10).tolist()
    X = [[1.0, xi, xi] for xi in x]  # cột 2 = cột 1: hoàn toàn phụ thuộc tuyến tính
    y = [float(i) for i in range(10)]
    result = ols_fit(X, y)
    assert not result.success


# ---------------------------------------------------------------------------
# hat_matrix
# ---------------------------------------------------------------------------

def test_hat_matrix_idempotent():
    """H² = H (tính lũy đẳng)."""
    X, _ = _noisy_xy()
    res = hat_matrix(X)
    H = np.array(res.H)
    diff = np.max(np.abs(H @ H - H))
    assert diff < 1e-8


def test_hat_matrix_symmetric():
    """H = H' (tính đối xứng)."""
    X, _ = _noisy_xy()
    res = hat_matrix(X)
    H = np.array(res.H)
    assert np.max(np.abs(H - H.T)) < 1e-10


def test_hat_matrix_trace_equals_rank():
    """trace(H) = p+1 (số tham số ước lượng)."""
    X, _ = _noisy_xy()
    res = hat_matrix(X)
    p_plus_1 = len(X[0])  # số cột của X
    assert abs(res.trace_H - p_plus_1) < 1e-8


def test_hat_matrix_projects_yhat():
    """Hy = ŷ: hat matrix áp lên y cho ra fitted values."""
    X, y = _noisy_xy()
    ols = ols_fit(X, y)
    H = np.array(hat_matrix(X).H)
    y_arr = np.array(y)
    y_hat_from_H = H @ y_arr
    y_hat_from_ols = np.array(ols.y_hat)
    assert np.max(np.abs(y_hat_from_H - y_hat_from_ols)) < 1e-8
