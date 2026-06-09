"""Kiểm thử chống data leakage và tính toàn vẹn holdout."""

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

def _config():
    return PipelineConfig(missing_method="median", log_numeric_features=False)


def _base_train():
    return pd.DataFrame({
        "ID": list("abcdef"),
        "nights": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "purpose": ["holiday", "business", "holiday", "business", "holiday", "business"],
        "total_cost": [100.0, 200.0, 150.0, 300.0, 250.0, 180.0],
    })


# ---------------------------------------------------------------------------
# D3-1: Thay đổi test set không làm thay đổi X_train
# ---------------------------------------------------------------------------

def test_different_test_sets_do_not_change_X_train():
    """B1 regression guard: thay đổi test không ảnh hưởng X_train."""
    train = _base_train()
    config = _config()

    pipeline_a = DataPipeline(config).fit(train)
    X_train_a = pipeline_a.transform(train).X

    test_small = pd.DataFrame({"ID": ["x"], "nights": [2.0], "purpose": ["holiday"]})
    test_large = pd.DataFrame({
        "ID": ["p", "q", "r", "s"],
        "nights": [1.0, 99.0, np.nan, 5.0],
        "purpose": ["holiday", "unseen_cat", "business", "another_unseen"],
    })

    pipeline_b = DataPipeline(config).fit(train)
    X_train_b = pipeline_b.transform(train).X
    pipeline_b.transform(test_small)
    pipeline_b.transform(test_large)

    np.testing.assert_allclose(X_train_a, X_train_b)


def test_vocabulary_fixed_after_fit():
    """category_vocabularies không đổi sau khi transform được gọi nhiều lần."""
    train = _base_train()
    pipeline = DataPipeline(_config()).fit(train)
    vocab_before = {k: list(v) for k, v in pipeline.category_vocabularies.items()}

    for _ in range(3):
        pipeline.transform(pd.DataFrame({
            "ID": ["z"],
            "nights": [1.0],
            "purpose": ["unseen_category"],
        }))

    for col, vocab in vocab_before.items():
        assert pipeline.category_vocabularies[col] == vocab


# ---------------------------------------------------------------------------
# D3-2: Scaler params cố định sau fit
# ---------------------------------------------------------------------------

def test_scaler_params_fixed_after_fit():
    """scaler_mean và scaler_std không thay đổi sau transform."""
    train = _base_train()
    pipeline = DataPipeline(_config()).fit(train)
    mean_before = pipeline.scaler_mean.copy() # pyright: ignore
    std_before = pipeline.scaler_std.copy() # pyright: ignore

    # transform với dữ liệu có range rất khác
    extreme_test = pd.DataFrame({
        "ID": ["x"],
        "nights": [1e9],
        "purpose": ["holiday"],
    })
    pipeline.transform(extreme_test)

    np.testing.assert_array_equal(pipeline.scaler_mean, mean_before)
    np.testing.assert_array_equal(pipeline.scaler_std, std_before)


# ---------------------------------------------------------------------------
# D3-3: Feature names cố định (số cột không thay đổi khi test set thay đổi)
# ---------------------------------------------------------------------------

def test_feature_names_stable_across_transforms():
    """feature_names không thay đổi sau các lần transform khác nhau."""
    train = _base_train()
    pipeline = DataPipeline(_config()).fit(train)
    names_after_fit = list(pipeline.feature_names)

    pipeline.transform(pd.DataFrame({"ID": ["x"], "nights": [1.0], "purpose": ["holiday"]}))
    assert pipeline.feature_names == names_after_fit

    pipeline.transform(pd.DataFrame({"ID": ["y"], "nights": [2.0], "purpose": ["new_cat"]}))
    assert pipeline.feature_names == names_after_fit


# ---------------------------------------------------------------------------
# D3-4: Holdout transform không phụ thuộc vào competition test
# ---------------------------------------------------------------------------

def test_holdout_transform_independent_of_test_set():
    """Holdout được transform bằng pipeline fit trên dev_train — kết quả không đổi
    bất kể competition test có khác nhau."""
    train = _base_train()
    holdout = pd.DataFrame({
        "ID": ["h1", "h2"],
        "nights": [2.0, 4.0],
        "purpose": ["holiday", "business"],
    })
    test_v1 = pd.DataFrame({"ID": ["t1"], "nights": [1.0], "purpose": ["holiday"]})
    test_v2 = pd.DataFrame({"ID": ["t2"], "nights": [99.0], "purpose": ["extra_class"]})

    pipeline_a = DataPipeline(_config()).fit(train)
    X_holdout_a = pipeline_a.transform(holdout).X
    pipeline_a.transform(test_v1)

    pipeline_b = DataPipeline(_config()).fit(train)
    X_holdout_b = pipeline_b.transform(holdout).X
    pipeline_b.transform(test_v2)

    np.testing.assert_allclose(X_holdout_a, X_holdout_b)
