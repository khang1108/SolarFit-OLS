"""Kiểm chứng các model scratch Part 2 bằng NumPy/sklearn tham chiếu."""

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from part2.analysis import (
    _fit_ols_beta,
    _fit_scratch_regularized_beta,
    _kfold_indices,
    _polynomial_degree_two,
)
from part1.regularization import elasticnet_fit


def _regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(120, 4))
    X = np.column_stack([np.ones(len(features)), features])
    y = X @ np.array([1.5, 2.0, 0.0, -1.0, 0.5]) + rng.normal(scale=0.1, size=120)
    return X, y


def test_part1_ols_adapter_matches_numpy_reference() -> None:
    X, y = _regression_data()
    scratch = _fit_ols_beta(X, y)
    reference = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(scratch, reference, rtol=1e-8, atol=1e-8)


def test_vectorized_lasso_and_elasticnet_match_sklearn_reference() -> None:
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    X, y = _regression_data()
    alpha = 0.02

    lasso_scratch = _fit_scratch_regularized_beta(X, y, alpha, l1_ratio=1.0)
    lasso_reference = sklearn_linear.Lasso(
        alpha=alpha, fit_intercept=True, max_iter=100000, tol=1e-10
    ).fit(X[:, 1:], y)
    np.testing.assert_allclose(
        lasso_scratch,
        np.r_[lasso_reference.intercept_, lasso_reference.coef_],
        rtol=2e-4,
        atol=2e-4,
    )

    elastic_scratch = _fit_scratch_regularized_beta(X, y, alpha, l1_ratio=0.7)
    elastic_reference = sklearn_linear.ElasticNet(
        alpha=alpha,
        l1_ratio=0.7,
        fit_intercept=True,
        max_iter=100000,
        tol=1e-10,
    ).fit(X[:, 1:], y)
    np.testing.assert_allclose(
        elastic_scratch,
        np.r_[elastic_reference.intercept_, elastic_reference.coef_],
        rtol=2e-4,
        atol=2e-4,
    )


def test_part1_elasticnet_matches_vectorized_scratch_adapter() -> None:
    X, y = _regression_data()
    alpha = 0.02
    l1_ratio = 0.7
    n_rows = X.shape[0]
    pure_python = elasticnet_fit(
        X.tolist(),
        y.tolist(),
        lambda_l1=2.0 * n_rows * alpha * l1_ratio,
        lambda_l2=n_rows * alpha * (1.0 - l1_ratio),
        max_iter=10000,
        tol=1e-9,
    )
    vectorized = _fit_scratch_regularized_beta(X, y, alpha, l1_ratio)
    assert pure_python.success
    np.testing.assert_allclose(
        np.asarray(pure_python.coefficients),
        vectorized,
        rtol=2e-4,
        atol=2e-4,
    )


def test_polynomial_expansion_matches_sklearn_reference() -> None:
    sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")
    X = np.array([[2.0, 3.0], [4.0, 5.0]])
    scratch, names = _polynomial_degree_two(X, ["x", "y"])
    reference_transformer = sklearn_preprocessing.PolynomialFeatures(
        degree=2, include_bias=False
    )
    reference = reference_transformer.fit_transform(X)
    np.testing.assert_allclose(scratch, reference)
    assert names == ["x", "y", "x^2", "x y", "y^2"]


def test_manual_kfold_is_reproducible_and_partitions_every_row() -> None:
    first = _kfold_indices(23, 5, random_state=42)
    second = _kfold_indices(23, 5, random_state=42)
    assert all(
        np.array_equal(first_train, second_train)
        and np.array_equal(first_valid, second_valid)
        for (first_train, first_valid), (second_train, second_valid) in zip(first, second)
    )
    validation_rows = np.concatenate([valid for _, valid in first])
    np.testing.assert_array_equal(np.sort(validation_rows), np.arange(23))
