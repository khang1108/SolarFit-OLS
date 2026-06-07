"""Kiểm thử DataPipeline: fit/transform API, missing values, scaler."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from part2.data_pipeline import DataPipeline, PipelineConfig  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_df():
    return pd.DataFrame({
        "ID": list("abcde"),
        "nights": [1.0, 2.0, np.nan, 4.0, 5.0],
        "purpose": ["holiday", "business", "holiday", "business", None],
        "total_cost": [100.0, 200.0, 150.0, 300.0, 250.0],
    })


def _config():
    return PipelineConfig(missing_method="median", log_numeric_features=False)


# ---------------------------------------------------------------------------
# fit học state từ train only
# ---------------------------------------------------------------------------

def test_fit_stores_category_vocabulary():
    """Sau fit, category_vocabularies chứa đúng các giá trị từ train."""
    pipeline = DataPipeline(_config()).fit(_train_df())
    assert "purpose" in pipeline.category_vocabularies
    vocab = pipeline.category_vocabularies["purpose"]
    assert "holiday" in vocab and "business" in vocab


def test_fit_vocabulary_excludes_null():
    """NaN trong train không được đưa vào vocabulary."""
    pipeline = DataPipeline(_config()).fit(_train_df())
    vocab = pipeline.category_vocabularies.get("purpose", [])
    assert all(v is not None for v in vocab)


def test_fit_stores_imputation_values():
    """numeric_imputation_values được thiết lập sau fit."""
    pipeline = DataPipeline(_config()).fit(_train_df())
    assert len(pipeline.numeric_imputation_values) > 0


def test_fit_stores_scaler_params():
    """scaler_mean và scaler_std được gán sau fit."""
    pipeline = DataPipeline(_config()).fit(_train_df())
    assert pipeline.scaler_mean is not None
    assert pipeline.scaler_std is not None
    assert pipeline._is_fitted


# ---------------------------------------------------------------------------
# transform ứng dụng state đã học
# ---------------------------------------------------------------------------

def test_transform_shape_consistent():
    """X_train và X_test phải có cùng số cột."""
    train = _train_df()
    test = pd.DataFrame({
        "ID": ["x", "y"],
        "nights": [3.0, 7.0],
        "purpose": ["holiday", "business"],
    })
    pipeline = DataPipeline(_config()).fit(train)
    X_train = pipeline.transform(train).X
    X_test = pipeline.transform(test).X
    assert X_train.shape[1] == X_test.shape[1]


def test_transform_unseen_category_zeros():
    """Category chưa thấy trong train → tất cả one-hot columns = 0."""
    pipeline = DataPipeline(_config()).fit(_train_df())
    new_row = pd.DataFrame({
        "ID": ["z"],
        "nights": [2.0],
        "purpose": ["completely_new_value"],
    })
    result = pipeline.transform(new_row)
    cat_indices = [
        i for i, name in enumerate(pipeline.feature_names)
        if pipeline.feature_types.get(name) == "categorical"
    ]
    assert cat_indices
    np.testing.assert_array_equal(result.X[0, cat_indices], np.zeros(len(cat_indices)))


def test_transform_missing_numeric_imputed():
    """NaN trong numeric feature được fill bằng giá trị từ fit (không raise)."""
    pipeline = DataPipeline(_config()).fit(_train_df())
    row_with_nan = pd.DataFrame({
        "ID": ["x"],
        "nights": [np.nan],
        "purpose": ["holiday"],
    })
    result = pipeline.transform(row_with_nan)
    # Không nên có NaN trong X sau transform
    assert not np.any(np.isnan(result.X))


# ---------------------------------------------------------------------------
# fit_transform convenience
# ---------------------------------------------------------------------------

def test_fit_transform_equals_fit_then_transform():
    """fit_transform(train) = fit(train).transform(train)."""
    train = _train_df()
    pipeline_a = DataPipeline(_config())
    X_a = pipeline_a.fit_transform(train).X

    pipeline_b = DataPipeline(_config()).fit(train)
    X_b = pipeline_b.transform(train).X

    np.testing.assert_allclose(X_a, X_b)
