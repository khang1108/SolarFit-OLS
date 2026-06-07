"""Kiểm thử coef_inference và vif."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "part1"))

from inference import coef_inference, vif  # pyright: ignore[reportMissingImports]
from ols_implementation import ols_fit  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(seed: int = 1):
    rng = np.random.default_rng(seed)
    n = 60
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 3.0 + 1.5 * x1 - 0.8 * x2 + 0.5 * rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2]).tolist()
    return X, y.tolist()


# ---------------------------------------------------------------------------
# coef_inference
# ---------------------------------------------------------------------------

def test_coef_inference_se_matches_numpy():
    """Sai số chuẩn scratch vs numpy oracle: max error < 1e-8."""
    X, y = _make_data()
    ols = ols_fit(X, y)
    inf_result = coef_inference(X, y, ols.beta_hat, ols.sigma2_hat)

    Xnp = np.array(X)
    cov_matrix = ols.sigma2_hat * np.linalg.inv(Xnp.T @ Xnp)
    se_np = np.sqrt(np.diag(cov_matrix))
    for se_s, se_n in zip(inf_result.std_errors, se_np):
        assert abs(se_s - se_n) < 1e-8


def test_coef_inference_pvalue_matches_scipy():
    """p-value scratch vs scipy.stats.t: max error < 1e-8."""
    X, y = _make_data()
    ols = ols_fit(X, y)
    inf_result = coef_inference(X, y, ols.beta_hat, ols.sigma2_hat)
    n = len(y)
    p = len(ols.beta_hat) - 1
    dof = n - p - 1
    Xnp = np.array(X)
    cov_matrix = ols.sigma2_hat * np.linalg.inv(Xnp.T @ Xnp)
    se_np = np.sqrt(np.diag(cov_matrix))
    t_np = np.array(ols.beta_hat) / se_np
    for pv_s, t_n in zip(inf_result.p_values, t_np):
        pv_scipy = float(2 * stats.t.sf(abs(t_n), dof))
        assert abs(pv_s - pv_scipy) < 1e-8


def test_coef_inference_ci_contains_true_beta():
    """99% CI phải chứa hệ số thực khi dữ liệu Tạo từ mô hình đã biết."""
    X, y = _make_data(seed=1)
    ols = ols_fit(X, y)
    inf_result = coef_inference(X, y, ols.beta_hat, ols.sigma2_hat, alpha=0.01)
    true_beta = [3.0, 1.5, -0.8]
    for j, true_b in enumerate(true_beta):
        assert inf_result.ci_lower[j] <= true_b <= inf_result.ci_upper[j]


def test_coef_inference_tstat_direction():
    """t-statistic phải cùng dấu với hệ số (vì SE > 0)."""
    X, y = _make_data()
    ols = ols_fit(X, y)
    inf_result = coef_inference(X, y, ols.beta_hat, ols.sigma2_hat)
    for b, t in zip(ols.beta_hat, inf_result.t_statistics):
        if abs(b) > 1e-10:
            assert (b > 0) == (t > 0)


# ---------------------------------------------------------------------------
# vif
# ---------------------------------------------------------------------------

def test_vif_independent_columns_near_one():
    """Các cột feature độc lập → VIF ≈ 1 (bỏ qua cột hệ số tự do index 0)."""
    rng = np.random.default_rng(5)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2, x3]).tolist()
    result = vif(X)
    # Bỏ qua index 0 (hệ số tự do) vì VIF của cột hằng không xác định
    for v in result.vif_values[1:]:
        assert v < 2.0


def test_vif_correlated_columns_inflated():
    """Cột có tương quan cao (r≈0.99) → VIF >> 1."""
    rng = np.random.default_rng(6)
    n = 100
    x1 = rng.normal(size=n)
    x2 = 0.99 * x1 + 0.01 * rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2]).tolist()
    result = vif(X)
    # Cột x1 và x2 phải có VIF >> 10
    assert result.vif_values[1] > 10
    assert result.vif_values[2] > 10


def test_vif_length_matches_columns():
    """Số VIF phải bằng số cột của X."""
    X, _ = _make_data()
    result = vif(X)
    assert len(result.vif_values) == len(X[0])
