# Phần 1: Cài Đặt OLS và Các Hàm Liên Quan

Phần này chứa các hàm cài đặt từ đầu (from scratch) cho phương pháp Ordinary Least Squares (OLS) và các công cụ phân tích liên quan, theo đúng yêu cầu của đề bài.

## 📁 Cấu trúc thư mục

```
part1/
├── __init__.py                  # Package initialization
├── ols_implementation.py        # Core OLS solver
├── model_evaluation.py          # Model metrics & residual analysis
├── inference.py                 # Coefficient inference & VIF
├── regularization.py            # Ridge regression
├── cross_validation.py          # k-fold cross-validation
├── gauss_markov_sim.py          # Gauss-Markov simulation
├── part1_notebook.ipynb         # Comprehensive demo notebook
└── README.md                    # This file
```

## 📚 Module Descriptions

### 1. `ols_implementation.py`
**Core OLS implementation with no external dependencies (pure Python)**

Functions:
- `ols_fit(X, y)` — Compute OLS coefficients β̂ = (X'X)⁻¹X'y
- `hat_matrix(X, tol)` — Compute projection matrix H = X(X'X)⁻¹X'
- `run_ols_analysis(X, y)` — Run both OLS and hat matrix analysis
- Helper functions: `_transpose`, `_matmul`, `_matvec`, `_mat_inv`, etc.

Key features:
- Uses Normal Equations approach
- Gauss-Jordan elimination with partial pivoting for matrix inversion
- Automatic singularity and multicollinearity detection
- Returns detailed `OLSResult` and `HatMatrixResult` dataclasses

### 2. `model_evaluation.py`
**Model evaluation metrics and residual analysis**

Functions:
- `model_metrics(y, y_hat, p)` — Calculate RSS, TSS, R², R̄², RMSE, F-statistic
- `residual_plots(X, y, beta_hat)` — Prepare data for 4 diagnostic plots

Returns:
- `ModelMetrics`: RSS, TSS, R², adjusted R², RMSE, F-statistic
- `ResidualPlotsData`: Residuals, fitted values, Q-Q plot data

### 3. `inference.py`
**Coefficient inference: standard errors, t-tests, confidence intervals**

Functions:
- `coef_inference(X, y, beta_hat, sigma2, alpha=0.05)` — Calculate standard errors, t-statistics, p-values, confidence intervals
- `vif(X, threshold=10.0)` — Calculate Variance Inflation Factors

Returns:
- `CoefficientInference`: se(β̂), t-statistics, p-values, 95% CI
- `VIFResult`: VIF values and multicollinearity detection

### 4. `regularization.py`
**Ridge regression and regularization methods**

Functions:
- `ridge_fit(X, y, lam)` — Ridge regression: β̂ᵣ = (X'X + λI)⁻¹X'y
- `ridge_trace(X, y, lambdas)` — Compute ridge coefficients across λ range

Returns:
- `RidgeResult`: Coefficients, RSS, ridge penalty, total loss
- `RidgeTraceData`: Coefficients and RSS for multiple λ values

### 5. `cross_validation.py`
**k-fold cross-validation for model evaluation**

Functions:
- `kfold_cv(X, y, k=5, metric="mse")` — k-fold CV for OLS
- `kfold_cv_ridge(X, y, lam, k=5)` — k-fold CV for Ridge
- `model_selection_cv(X, y, k=5, models=None)` — Compare multiple models

Supported metrics: MSE, RMSE, MAE, R²

Returns:
- `CVResult`: CV scores, mean/std, train/test scores per fold
- `ModelComparisonResult`: Best model and scores

### 6. `gauss_markov_sim.py`
**Gauss-Markov theorem verification via Monte Carlo simulation**

Functions:
- `simulate_gauss_markov(X, beta_true, sigma, n_simulations)` — Verify:
  - Unbiasedness: E[β̂] = β
  - Efficiency: OLS has minimum variance
- `verify_assumptions(X, y, beta_hat, sigma2)` — Check GM1-GM5 assumptions

Returns:
- `GaussMarkovSimulation`: Mean/std of estimates, bias, MSE, theoretical variance
- `dict`: Verification of each Gauss-Markov assumption

## 🚀 Quick Start

### Using the functions directly

```python
from part1 import ols_fit, hat_matrix, model_metrics, coef_inference, vif

# Load your data
X_list = ...  # List[List[float]], shape (n, p+1), first column = 1
y_list = ...  # List[float], shape (n,)

# Fit OLS
result = ols_fit(X_list, y_list)
print(f"β̂ = {result.beta_hat}")
print(f"σ̂² = {result.sigma2_hat}")

# Evaluate model
metrics = model_metrics(y_list, result.y_hat, p=len(X_list[0])-1)
print(f"R² = {metrics.r2:.4f}")

# Coefficient inference
inference = coef_inference(X_list, y_list, result.beta_hat, result.sigma2_hat)
print(f"Standard errors: {inference.std_errors}")

# Check multicollinearity
vif_result = vif(X_list)
print(f"Max VIF: {vif_result.max_vif:.2f}")
```

### Running the demo notebook

```bash
# In the part1 directory
jupyter notebook part1_notebook.ipynb
```

## ✅ Verification

All functions have been verified against NumPy:
- `ols_fit`: ||β̂ - numpy|| < 1e-8 ✓
- `hat_matrix`: ||H - numpy|| < 1e-8 ✓
- `model_metrics`: Matches R² formula exactly ✓
- `coef_inference`: SE and t-stats match theory ✓

## 📋 Requirements

**Core**:
- Python 3.6+
- Pure Python (no NumPy required for main functions)

**For testing and examples**:
- NumPy
- Matplotlib (for visualizations)
- SciPy (optional, for p-value calculations)
- Jupyter (for notebook)

## 📝 Mathematical Background

### OLS Problem
Minimize: ||y - Xβ||² subject to X ∈ ℝⁿˣ⁽ᵖ⁺¹⁾

Solution: β̂ = (X'X)⁻¹X'y

### Gauss-Markov Assumptions (GM1-GM5)
1. **Linearity**: y = Xβ + ε
2. **No multicollinearity**: rank(X) = p+1
3. **Exogeneity**: E[ε|X] = 0
4. **Homoscedasticity**: Var(ε|X) = σ²I
5. **Normality**: ε ~ N(0, σ²I) [for inference]

### Key Results
- **BLUE Theorem**: Under GM1-GM4, OLS is Best Linear Unbiased Estimator
- **Efficiency**: OLS has minimum variance among linear unbiased estimators
- **Distribution**: β̂ ~ N(β, σ²(X'X)⁻¹) under GM5

## 🔍 Implementation Details

### Matrix Inversion
Uses Gauss-Jordan elimination with partial pivoting for numerical stability.

### Rank Computation
Gaussian elimination with threshold (EPS_PIVOT = 1e-12) for singularity detection.

### VIF Calculation
VIF_j = 1/(1 - R²_j) where R²_j is from auxiliary regression.

## 📖 References

The implementations follow standard references:
- Strang (2023): Introduction to Linear Algebra
- Wooldridge (2013): Introductory Econometrics
- Greene (2012): Econometric Analysis
- James et al. (2013): An Introduction to Statistical Learning

## ✨ Features

- ✓ Pure Python implementation (no NumPy for core functions)
- ✓ Numerical stability (Gauss-Jordan with pivoting)
- ✓ Comprehensive error handling
- ✓ Detailed result dataclasses
- ✓ NumPy verification available
- ✓ k-fold cross-validation
- ✓ Ridge regression and regularization
- ✓ Gauss-Markov theorem verification
- ✓ Complete coefficient inference (SE, t, p-values, CI)

## 📞 Contact

For questions or issues, refer to the main project documentation.
