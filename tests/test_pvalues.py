"""Kiểm thử p-value Student-t và F-test được triển khai từ đầu."""

import math
import sys
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]


ROOT = Path(__file__).resolve().parents[1]
PART1 = ROOT / "part1"
sys.path.insert(0, str(PART1))

from model_evaluation import model_metrics  # pyright: ignore[reportMissingImports]
from statistical_distributions import (  # pyright: ignore[reportMissingImports]
    f_survival_probability,
    student_t_critical,
    student_t_two_sided_pvalue,
)


def test_student_t_boundaries_symmetry_and_monotonicity() -> None:
    assert student_t_two_sided_pvalue(0.0, 10) == 1.0
    assert student_t_two_sided_pvalue(float("inf"), 10) == 0.0
    assert student_t_two_sided_pvalue(2.5, 10) == student_t_two_sided_pvalue(-2.5, 10)

    values = [student_t_two_sided_pvalue(t_stat, 10) for t_stat in [0, 1, 2, 5]]
    assert values == sorted(values, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_f_boundaries_monotonicity_and_perfect_fit() -> None:
    assert f_survival_probability(0.0, 2, 20) == 1.0
    assert f_survival_probability(float("inf"), 2, 20) == 0.0

    values = [f_survival_probability(f_stat, 2, 20) for f_stat in [0, 1, 5, 20]]
    assert values == sorted(values, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in values)

    metrics = model_metrics([1.0, 3.0, 5.0], [1.0, 3.0, 5.0], p=1)
    assert math.isinf(metrics.f_statistic)
    assert metrics.f_pvalue == 0.0


def test_invalid_distribution_inputs() -> None:
    with pytest.raises(ValueError):
        student_t_two_sided_pvalue(1.0, 0)
    with pytest.raises(ValueError):
        student_t_critical(0.0, 10)
    with pytest.raises(ValueError):
        f_survival_probability(-1.0, 2, 10)
    with pytest.raises(ValueError):
        f_survival_probability(1.0, 0, 10)


def test_student_t_matches_scipy() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    for dof in [1, 2, 5, 30, 1000]:
        for t_stat in [0.0, 0.1, 1.96, 5.0, 50.0]:
            expected = float(2.0 * scipy_stats.t.sf(abs(t_stat), dof))
            actual = student_t_two_sided_pvalue(t_stat, dof)
            assert actual == pytest.approx(expected, rel=1e-8, abs=1e-10)


def test_f_pvalue_matches_scipy() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    for df_numerator, df_denominator in [(1, 1), (2, 20), (5, 30), (20, 100)]:
        for f_stat in [0.0, 0.01, 1.0, 5.0, 1000.0]:
            expected = float(
                scipy_stats.f.sf(f_stat, df_numerator, df_denominator)
            )
            actual = f_survival_probability(
                f_stat, df_numerator, df_denominator
            )
            assert actual == pytest.approx(expected, rel=1e-8, abs=1e-10)


def test_core_pvalue_modules_do_not_import_scipy() -> None:
    core_sources = [
        PART1 / "statistical_distributions.py",
        PART1 / "model_evaluation.py",
        PART1 / "inference.py",
        ROOT / "part2" / "analysis.py",
        ROOT / "part2" / "models.py",
    ]
    for source_path in core_sources:
        source = source_path.read_text(encoding="utf-8")
        core_source = source.split('if __name__ == "__main__":', maxsplit=1)[0]
        assert "from scipy import stats" not in core_source
        assert "import scipy.stats" not in core_source
