"""Kiểm thử chẩn đoán giả thiết Gauss-Markov."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part1"))

from gauss_markov_sim import verify_assumptions  # pyright: ignore[reportMissingImports]
from ols_implementation import ols_fit  # pyright: ignore[reportMissingImports]


def test_gm1_is_diagnostic_not_hard_coded_boolean() -> None:
    X = [[1.0, float(value)] for value in range(-5, 6)]
    y = [2.0 + 3.0 * row[1] for row in X]
    fit = ols_fit(X, y)

    assumptions = verify_assumptions(X, y, fit.beta_hat, fit.sigma2_hat)

    assert assumptions["GM1_Linearity"] is None
    assert "Residuals vs Fitted" in assumptions["GM1_Note"]
    assert len(assumptions["GM1_FittedValues"]) == len(y)
    assert len(assumptions["GM1_BinCenters"]) == len(
        assumptions["GM1_BinMeanResiduals"]
    )
