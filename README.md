# Data Fitting & OLS — From Scratch to Zindi

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-only%20for%20arrays-013243?style=flat-square&logo=numpy)
![LaTeX](https://img.shields.io/badge/Report-pdflatex-008080?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Applied Mathematics in Statistics — HCMUS, Faculty of Information Technology.

This project builds ordinary least squares regression entirely from scratch in Part 1, then applies those implementations to a real-world Zindi competition (Tanzania Tourism Expenditure) in Part 2. No `numpy.linalg.solve`, no `sklearn.linear_model` — every algorithm from Gauss-Jordan inversion to coordinate descent is hand-written and verified against library oracles.

---

## Results

**Part 2 — Tanzania Tourism Expenditure (Zindi)**

| Model | Holdout MAE (TZS) | CV R² | Features |
|---|---|---|---|
| OLS | 4,743,845 | 0.281 | 151 |
| Ridge (α=10) | 4,749,367 | 0.294 | 151 |
| Lasso (α=0.00316) | 4,786,367 | 0.293 | 49 / 151 |
| ElasticNet | 4,761,273 | 0.294 | 77 / 151 |
| Poly(2) + ElasticNet | 4,764,671 | 0.291 | 81 / 161 |
| Kernel Ridge RBF | 4,943,734 | 0.296 | — |
| **Ensemble (mean)** | **4,738,269** | — | 5 models |

All models trained on `log1p(total_cost)` and evaluated in original TZS. Hyperparameters selected via nested cross-validation (5-fold outer, 3-fold inner). Holdout set (20% of Train.csv) was held out and evaluated exactly once.

---

## Project Structure

```
.
├── part1/                          # From-scratch implementations
│   ├── ols_implementation.py       # Normal equations via Gauss-Jordan, hat matrix
│   ├── inference.py                # t-test, F-test, VIF, confidence intervals
│   ├── statistical_distributions.py# Student-t, F CDF via regularized incomplete beta
│   ├── regularization.py           # Ridge (closed-form), Lasso (coordinate descent)
│   ├── model_evaluation.py         # R², adjusted R², F-statistic, residual diagnostics
│   ├── cross_validation.py         # k-fold CV, ridge CV, model selection
│   ├── monte_carlo_gauss_markov.py # Monte Carlo verification of BLUE property
│   ├── gauss_markov_sim.py         # Gauss-Markov assumption simulation tools
│   ├── tests/                      # Unit tests for all modules
│   ├── outputs/                    # Diagnostic plots (residuals, regularization paths)
│   └── part1_notebook.ipynb        # End-to-end walkthrough with theory and results
│
├── part2/                          # Tanzania Tourism application
│   ├── data_pipeline.py            # Imputation, one-hot encoding, StandardScaler
│   ├── models.py                   # RegressionModels — wraps Part 1 scratch solvers
│   ├── evaluate.py                 # CrossValidationResult, ModelEvaluator
│   ├── analysis.py                 # Full pipeline runner, KernelRidgeRBF, Ensemble
│   ├── elasticnet_polynomial.py    # Polynomial feature expansion + ElasticNet
│   ├── eda.py                      # VIF computation using Part 1 ols_fit
│   ├── shap_analysis.py            # SHAP values via Part 1 Ridge wrapper
│   ├── tests/                      # Integration and leakage tests
│   ├── outputs/                    # model_comparison.csv, feature_importance.csv, submissions/
│   └── part2_notebook.ipynb        # End-to-end pipeline notebook
│
├── report/                         # LuaLaTeX report
│   ├── report.tex                  # Main document
│   ├── sections/                   # part1.tex, gauss-markov.tex, part2.tex, conclusion.tex
│   ├── images/                     # All figures referenced in the report
│   └── ref.bib                     # Bibliography
│
├── docs/                           # Project requirements (PDF)
└── requirements.txt
```

---

## Key Design Principles

**Scratch-first, oracle-verified.** Every core algorithm is implemented without calling `numpy.linalg` solvers or `sklearn` estimators. NumPy, SciPy, and scikit-learn are used strictly as independent oracles: after computing a result from scratch, the implementation checks agreement to machine-epsilon precision.

**No data leakage.** The `DataPipeline` fits imputation statistics, one-hot vocabulary, and StandardScaler parameters on the training set only, then transforms the test set using those frozen parameters. The holdout split (20%) is evaluated exactly once at the very end, never during hyperparameter selection.

**Structured error returns, not silent fallbacks.** Functions like `ols_fit` return `OLSResult(success=False)` on singular input instead of silently switching to an alternative solver. There are no try/except fallback patterns in the codebase.

---

## Installation

```bash
git clone <repo-url>
cd DataFitting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data files (`Train.csv`, `Test.csv`) from the [Zindi Tanzania Tourism competition](https://zindi.africa/competitions/tanzania-tourism-prediction) go in `part2/data/`.

---

## Quick Start

**Part 1 — Run the notebook:**
```bash
cd part1
jupyter notebook part1_notebook.ipynb
```

**Part 2 — Run the full pipeline:**
```python
# from part2/
import analysis
output_paths = analysis.run()   # trains 8 models, writes outputs/
```

**Run tests:**
```bash
pytest part1/tests/ part2/tests/ -v
```

**Build the report:**
```bash
cd report
latexmk -pdf report.tex
```

---

## Team

| Name | Student ID | Contribution |
|---|---|---|
| Nguyen Phuc Khang | 24120068 | Project lead; OLS, Ridge/Lasso, inference implementation; numerical verification; report integration |
| Hoang Trong Nghia | 24120103 | OLS theory, hat matrix, Gauss-Markov assumptions, Monte Carlo simulation |
| Mai Hoang Nhat | 24120109 | EDA, missing value analysis, data shift, model assumption diagnostics |
| Nguyen Hoang Nhat | 24120110 | Data pipeline, preprocessing, feature engineering, Part 2 model training |
| Vo Phung Nhat Long | 24120088 | Cross-validation, model comparison, feature importance, submission pipeline |
