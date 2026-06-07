"""Kiểm thử Generalized Cross-Validation của Ridge."""

import sys
from pathlib import Path

import numpy as np  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part1"))

from regularization import (  # pyright: ignore[reportMissingImports]
    _ridge_effective_degrees_of_freedom,
    ridge_trace,
)


def _demo_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    n_rows = 80
    x1 = rng.normal(size=n_rows)
    x2 = 0.95 * x1 + 0.05 * rng.normal(size=n_rows)
    x3 = rng.normal(size=n_rows)
    x4 = rng.normal(size=n_rows)
    X = np.column_stack([np.ones(n_rows), x1, x2, x3, x4])
    y = X @ np.array([2.0, 3.0, 0.0, 1.5, 0.0]) + 0.7 * rng.normal(size=n_rows)
    return X, y


def test_effective_degrees_of_freedom_matches_numpy_svd() -> None:
    X, _ = _demo_data()
    centered = X[:, 1:] - X[:, 1:].mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)

    for lam in [0.0, 0.01, 1.0, 1000.0]:
        expected = 1.0 + float(
            np.sum(singular_values ** 2 / (singular_values ** 2 + lam))
        )
        actual = _ridge_effective_degrees_of_freedom(X.tolist(), lam)
        assert actual == pytest.approx(expected, abs=1e-9)


def test_gcv_has_interior_minimum_on_demo_data() -> None:
    X, y = _demo_data()
    lambdas = np.logspace(-2, 3, 40).tolist()
    trace = ridge_trace(X.tolist(), y.tolist(), lambdas)

    best_index = int(np.argmin(trace.gcv_trace))
    assert 0 < best_index < len(lambdas) - 1
    assert all(
        trace.gcv_trace[index] >= trace.gcv_trace[index + 1] - 1e-12
        for index in range(best_index)
    )
    assert all(
        trace.gcv_trace[index] <= trace.gcv_trace[index + 1] + 1e-12
        for index in range(best_index, len(lambdas) - 1)
    )
    assert trace.effective_df_trace[0] > trace.effective_df_trace[-1]


def test_gcv_uses_mean_rss_and_effective_degrees_of_freedom() -> None:
    X, y = _demo_data()
    trace = ridge_trace(X.tolist(), y.tolist(), [1.0])

    n_rows = len(y)
    expected = (trace.rss_trace[0] / n_rows) / (
        1.0 - trace.effective_df_trace[0] / n_rows
    ) ** 2
    assert trace.gcv_trace[0] == pytest.approx(expected, abs=1e-12)
