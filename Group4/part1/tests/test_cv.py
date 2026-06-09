"""Kiểm thử kfold_cv và kfold_cv_ridge."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "part1"))

from cross_validation import CVResult, kfold_cv, kfold_cv_ridge  # pyright: ignore[reportMissingImports]
from ols_implementation import ols_fit  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data(seed: int = 9):
    rng = np.random.default_rng(seed)
    n = 60
    x = rng.normal(size=n)
    y = 2.0 + 3.0 * x + 0.5 * rng.normal(size=n)
    X = np.column_stack([np.ones(n), x]).tolist()
    return X, y.tolist()


# ---------------------------------------------------------------------------
# kfold_cv
# ---------------------------------------------------------------------------

def test_kfold_cv_same_folds_same_result():
    """Cùng fold_indices → kết quả hoàn toàn giống nhau (determinism)."""
    X, y = _data()
    result1 = kfold_cv(X, y, k=5, metric="mse", random_state=0)
    result2 = kfold_cv(X, y, k=5, metric="mse", random_state=0)
    assert abs(result1.mean_cv_score - result2.mean_cv_score) < 1e-12
    assert result1.cv_scores == result2.cv_scores


def test_kfold_cv_fold_indices_override_random_state():
    """Khi truyền fold_indices, random_state bị bỏ qua — kết quả phải giống nhau."""
    X, y = _data()
    n = len(y)
    # Tạo fold thủ công: 5 fold đều nhau
    indices = [list(range(i, n, 5)) for i in range(5)]
    result_a = kfold_cv(X, y, k=5, metric="mse", fold_indices=indices, random_state=0)
    result_b = kfold_cv(X, y, k=5, metric="mse", fold_indices=indices, random_state=99)
    assert result_a.cv_scores == result_b.cv_scores


def test_kfold_cv_returns_correct_k_folds():
    """CVResult phải có đúng k điểm số."""
    X, y = _data()
    k = 4
    result = kfold_cv(X, y, k=k, metric="mse", random_state=1)
    assert len(result.cv_scores) == k
    assert result.k == k


def test_kfold_cv_mse_nonnegative():
    """MSE phải >= 0 trên mọi fold."""
    X, y = _data()
    result = kfold_cv(X, y, k=5, metric="mse", random_state=2)
    assert all(s >= 0.0 for s in result.cv_scores)


def test_kfold_cv_r2_improves_with_stronger_signal():
    """Signal mạnh hơn (ít noise) → CV R² cao hơn."""
    rng = np.random.default_rng(11)
    n = 80
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x]).tolist()
    # noise lớn
    y_noisy = (2.0 + 3.0 * x + 5.0 * rng.normal(size=n)).tolist()
    # noise nhỏ
    y_clean = (2.0 + 3.0 * x + 0.01 * rng.normal(size=n)).tolist()

    r2_noisy = kfold_cv(X, y_noisy, k=5, metric="r2", random_state=0).mean_cv_score
    r2_clean = kfold_cv(X, y_clean, k=5, metric="r2", random_state=0).mean_cv_score
    assert r2_clean > r2_noisy


# ---------------------------------------------------------------------------
# kfold_cv_ridge
# ---------------------------------------------------------------------------

def test_kfold_cv_ridge_lambda0_close_to_ols_cv():
    """λ=0 cho Ridge CV ≈ OLS CV (sai số do floating point nhỏ)."""
    X, y = _data()
    result_ols = kfold_cv(X, y, k=5, metric="mse", random_state=42)
    result_ridge = kfold_cv_ridge(X, y, lam=0.0, k=5, metric="mse", random_state=42)
    assert abs(result_ols.mean_cv_score - result_ridge.mean_cv_score) < 1e-6


def test_kfold_cv_ridge_returns_cvresult():
    """kfold_cv_ridge phải trả về CVResult hợp lệ."""
    X, y = _data()
    result = kfold_cv_ridge(X, y, lam=1.0, k=5, random_state=0)
    assert isinstance(result, CVResult)
    assert len(result.cv_scores) == 5
    assert result.mean_cv_score >= 0.0
