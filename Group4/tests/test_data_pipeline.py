"""Kiểm thử fit/transform API và các hàng rào chống data leakage."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from part2.analysis import FinalModelResult, _evaluate_holdout_once
from part2.data_pipeline import DataPipeline, PipelineConfig


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": ["a", "b", "c", "d"],
            "nights": [1.0, 2.0, np.nan, 4.0],
            "purpose": ["holiday", "business", "holiday", None],
            "total_cost": [100.0, 200.0, 150.0, 300.0],
        }
    )


def test_test_categories_and_values_cannot_change_transformed_train() -> None:
    train = _train_frame()
    config = PipelineConfig(missing_method="median", log_numeric_features=False)

    first = DataPipeline(config).fit(train)
    X_train_first = first.transform(train).X
    first.transform(
        pd.DataFrame({"ID": ["x"], "nights": [10.0], "purpose": ["exclusive"]})
    )

    second = DataPipeline(config).fit(train)
    X_train_second = second.transform(train).X
    second.transform(
        pd.DataFrame({"ID": ["y"], "nights": [1e12], "purpose": ["another_new"]})
    )

    np.testing.assert_allclose(X_train_first, X_train_second)
    assert first.feature_names == second.feature_names
    np.testing.assert_allclose(first.scaler_mean, second.scaler_mean)
    np.testing.assert_allclose(first.scaler_std, second.scaler_std)


def test_unknown_category_becomes_zero_one_hot_row() -> None:
    pipeline = DataPipeline(
        PipelineConfig(missing_method="median", log_numeric_features=False)
    ).fit(_train_frame())
    transformed = pipeline.transform(
        pd.DataFrame({"ID": ["x"], "nights": [2.0], "purpose": ["exclusive"]})
    )

    categorical_indices = [
        index
        for index, name in enumerate(pipeline.feature_names)
        if pipeline.feature_types.get(name) == "categorical"
    ]
    assert categorical_indices
    np.testing.assert_array_equal(
        transformed.X[0, categorical_indices],
        np.zeros(len(categorical_indices)),
    )


def test_run_matches_manual_fit_transform(tmp_path: Path) -> None:
    train = _train_frame()
    test = pd.DataFrame(
        {
            "ID": ["x", "y"],
            "nights": [3.0, 7.0],
            "purpose": ["holiday", "exclusive"],
        }
    )
    train.to_csv(tmp_path / "Train.csv", index=False)
    test.to_csv(tmp_path / "Test.csv", index=False)
    config = PipelineConfig(
        data_dir=str(tmp_path),
        missing_method="median",
        log_numeric_features=False,
    )

    run_result = DataPipeline(config).run()
    manual_pipeline = DataPipeline(config).fit(train)
    manual_train = manual_pipeline.transform(train)
    manual_test = manual_pipeline.transform(test)

    np.testing.assert_allclose(run_result.X_train, manual_train.X)
    np.testing.assert_allclose(run_result.X_test, manual_test.X)
    np.testing.assert_allclose(run_result.y_train, manual_train.y)
    assert run_result.feature_names == manual_pipeline.feature_names


def test_holdout_metrics_are_computed_before_competition_predictions_are_kept() -> None:
    result = FinalModelResult(
        name="demo",
        slug="demo",
        train_pred=np.array([10.0, 20.0]),
        test_pred=np.array([11.0, 19.0, 999.0]),
        train_mae=0.0,
        train_rmse=0.0,
        train_r2=1.0,
        feature_count=1,
        nonzero_coef=1,
        detail="test",
    )

    _evaluate_holdout_once([result], np.array([10.0, 20.0]), holdout_rows=2)

    assert result.holdout_mae == 1.0
    assert result.holdout_rmse == 1.0
    assert result.holdout_r2 == 0.96
    np.testing.assert_array_equal(result.test_pred, np.array([999.0]))
