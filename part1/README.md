# Phần 1: Cài Đặt OLS và Các Hàm Liên Quan

Phần này chứa các hàm cài đặt từ đầu (from scratch) cho phương pháp Ordinary Least Squares (OLS) và các công cụ phân tích liên quan, theo đúng yêu cầu của đề bài.

> **Thiết kế:** mỗi file là một script độc lập, **không phải package**. Chạy trực tiếp `python <tên_file>.py` (từ trong thư mục `part1/`) là file sẽ tự in ra kết quả minh họa tương ứng với một tiêu chí chấm điểm. Các hàm lõi vẫn thuần Python; chỉ khối demo `__main__` mới dùng NumPy/Matplotlib/SciPy để kiểm chứng và vẽ đồ thị.

## 📁 Cấu trúc thư mục

```
part1/
├── ols_implementation.py        # OLS từ đầu + hat matrix (kiểm chứng NumPy)
├── inference.py                 # Suy luận hệ số: t-test, p-value, F-test, VIF
├── model_evaluation.py          # Chỉ số đánh giá + 4 đồ thị chẩn đoán phần dư
├── regularization.py            # Ridge & Lasso + vẽ ridge trace / lasso path
├── cross_validation.py          # k-fold CV và so sánh mô hình
├── gauss_markov_sim.py          # Kiểm chứng Gauss-Markov bằng Monte Carlo
├── monte_carlo_gauss_markov.py  # Monte Carlo so sánh OLS với ước lượng khác
├── outputs/                     # Đồ thị PNG sinh ra khi chạy demo
├── part1_notebook.ipynb         # Notebook tổng hợp có markdown giải thích
└── README.md                    # File này
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
**Ridge regression, Lasso và các phương pháp chuẩn hóa**

Functions:
- `ridge_fit(X, y, lam)` — Ridge regression: β̂ᵣ = (X'X + λI)⁻¹X'y
- `ridge_trace(X, y, lambdas)` — Quỹ đạo hệ số Ridge trên dải λ
- `lasso_fit(X, y, lam)` — Lasso bằng coordinate descent với soft-thresholding
- `lasso_path(X, y, lambdas)` — Quỹ đạo Lasso với warm start (λ giảm dần)

Returns:
- `RidgeResult` / `LassoResult`: hệ số, RSS, số hạng phạt, tổng hàm mục tiêu
- `RidgeTraceData` / `LassoTraceData`: hệ số theo từng λ (để vẽ trace/path)

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

## 🚀 Cách chạy

### Chạy từng file để ra kết quả (mỗi file ứng với một tiêu chí chấm điểm)

```bash
cd part1
python ols_implementation.py     # OLS từ đầu + hat matrix, kiểm chứng với NumPy
python inference.py              # t-stat, p-value, F-test, VIF (kiểm chứng SciPy)
python model_evaluation.py       # 4 đồ thị chẩn đoán phần dư → outputs/residual_diagnostics.png
python regularization.py         # Ridge & Lasso, vẽ ridge trace → outputs/regularization_paths.png
python cross_validation.py       # k-fold CV và so sánh OLS vs Ridge
python gauss_markov_sim.py       # Monte Carlo kiểm chứng OLS không chệch & phương sai
python monte_carlo_gauss_markov.py  # Monte Carlo so sánh OLS với ước lượng tuyến tính khác
```

| File | Tiêu chí rubric | Đầu ra chính |
|------|-----------------|--------------|
| `ols_implementation.py` | Cài đặt OLS từ đầu + Hat matrix | β̂, H, sai số so với NumPy < 1e-8; `idem_err`, `sym_err` |
| `inference.py` | Kiểm định hệ số (t, F) | Bảng t-stat/p-value/CI, F-test, đối chiếu SciPy |
| `model_evaluation.py` | Phân tích phần dư (4 biểu đồ) | `outputs/residual_diagnostics.png` + nhận xét |
| `regularization.py` | Regularization + vẽ ridge trace | `outputs/regularization_paths.png` |
| `cross_validation.py` | Cross-validation | Điểm CV từng fold, so sánh mô hình |
| `gauss_markov_sim.py` | Minh họa Gauss–Markov | E[β̂] vs β, sd mẫu vs sd lý thuyết |

### Dùng lại hàm trong code khác

Vì đây không phải package, ta import theo tên module trực tiếp (cần `part1/` nằm trong `sys.path`):

```python
import sys; sys.path.insert(0, "duong/dan/toi/part1")
from ols_implementation import ols_fit
from model_evaluation import model_metrics
from inference import coef_inference, vif

result = ols_fit(X_list, y_list)            # X_list: List[List[float]] có cột 1 ở đầu
metrics = model_metrics(y_list, result.y_hat, p=len(X_list[0]) - 1)
inf = coef_inference(X_list, y_list, result.beta_hat, result.sigma2_hat)
print(f"R² = {metrics.r2:.4f}, max VIF = {vif(X_list).max_vif:.2f}")
```

### Chạy notebook tổng hợp

```bash
cd part1
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
