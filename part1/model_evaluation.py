"""
Module đánh giá chất lượng mô hình hồi quy OLS thông qua các chỉ số thống kê.

Module này cung cấp hai hàm chính: model_metrics tính tập hợp đầy đủ các chỉ số
đánh giá mô hình bao gồm RSS, TSS, ESS, R², R² hiệu chỉnh, F-statistic và RMSE;
và residual_plots tạo dữ liệu cho 4 đồ thị chẩn đoán phần dư chuẩn (residuals vs
fitted, Q-Q plot, scale-location, leverage). 
"""

from dataclasses import dataclass
from typing import List, Tuple
from math import inf, sqrt
try:
    from .ols_implementation import ols_fit
    from .statistical_distributions import f_survival_probability
except ImportError:  # Cho phép chạy trực tiếp file
    from ols_implementation import ols_fit
    from statistical_distributions import f_survival_probability

import os
import sys
import numpy as np  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]


@dataclass
class ModelMetrics:
    """Dataclass chứa toàn bộ chỉ số đánh giá chất lượng mô hình hồi quy."""
    n: int             # Số quan sát
    p: int             # Số biến giải thích (không tính hệ số tự do)
    rss: float         # Residual Sum of Squares = ||y - ŷ||²
    tss: float         # Total Sum of Squares = ||y - ȳ||²
    ess: float         # Explained Sum of Squares = TSS - RSS
    r2: float          # Hệ số xác định R² = 1 - RSS/TSS
    r2_adj: float      # R² hiệu chỉnh = 1 - [RSS/(n-p-1)] / [TSS/(n-1)]
    sigma2_hat: float  # Ước lượng phương sai nhiễu σ̂² = RSS/(n-p-1)
    f_statistic: float # F-statistic = (ESS/p) / (RSS/(n-p-1))
    f_pvalue: float    # P-value đuôi phải của F-statistic, tính từ đầu
    rmse: float        # Root Mean Squared Error = sqrt(σ̂²)


def model_metrics(y: List[float], y_hat: List[float], p: int) -> ModelMetrics:
    """Tính toàn bộ các chỉ số đánh giá mô hình hồi quy từ giá trị quan sát và dự báo.

    Hàm này thực hiện phân rã phương sai TSS = ESS + RSS, từ đó tính R², R² hiệu
    chỉnh, F-statistic và RMSE. 
    
    F-statistic kiểm định giả thuyết H₀: β₁ = β₂ = ... = βₚ = 0,
    phân phối F(p, n-p-1) dưới H₀ khi phần dư có phân phối chuẩn.

    Args:
        y: Vector giá trị quan sát kích thước (n,).
        y_hat: Vector giá trị dự báo ŷ = Xβ̂ kích thước (n,).
        p: Số biến giải thích không tính hệ số tự do (tức là số cột của X trừ 1).

    Returns:
        ModelMetrics chứa RSS, TSS, ESS, R², R² hiệu chỉnh, σ̂², F-statistic,
        F p-value và RMSE.

    Raises:
        ValueError: Khi y và y_hat có độ dài khác nhau.
    """
    n = len(y)
    if len(y_hat) != n:
        raise ValueError("y and y_hat must have same length")

    # Tính trung bình ȳ — mức cơ sở (baseline) so với mô hình chỉ có hệ số tự do
    y_mean = sum(y) / n

    # RSS: phần dư không được giải thích bởi mô hình, là hàm mà OLS tối thiểu hóa
    residuals = [y[i] - y_hat[i] for i in range(n)]
    rss = sum(e**2 for e in residuals)

    # TSS: tổng độ biến thiên của y so với trung bình, không phụ thuộc mô hình
    tss = sum((y[i] - y_mean)**2 for i in range(n))

    # ESS: phần độ biến thiên được giải thích; đảm bảo TSS = ESS + RSS
    ess = tss - rss

    # R² = ESS/TSS: tỷ lệ phương sai được giải thích, luôn trong [0, 1] khi có hệ số tự do
    r2 = 1.0 - (rss / tss) if tss != 0 else 0.0

    # R² hiệu chỉnh: chia theo bậc tự do để phạt khi thêm biến không ý nghĩa
    dof_residual = n - p - 1
    dof_total = n - 1
    if dof_residual > 0 and tss != 0:
        r2_adj = 1.0 - (rss / dof_residual) / (tss / dof_total)
    else:
        r2_adj = r2

    # σ̂²: ước lượng không chệch phương sai nhiễu; bậc tự do n-p-1 chứ không phải n
    if dof_residual > 0:
        sigma2_hat = rss / dof_residual
    else:
        sigma2_hat = float('nan')

    # RMSE = sqrt(σ̂²): sai số trung bình theo đơn vị gốc của y, dễ diễn giải hơn MSE
    rmse = sqrt(sigma2_hat) if sigma2_hat > 0 else 0.0

    # F-statistic = (ESS/p) / (RSS/(n-p-1)): tỷ số giữa phương sai giải thích được
    # và phương sai phần dư; phân phối F(p, n-p-1) dưới H₀: β₁ = ... = βₚ = 0
    if dof_residual > 0 and p > 0 and rss == 0.0:
        f_stat = inf if ess > 0.0 else float("nan")
    elif dof_residual > 0 and p > 0:
        f_stat = max(0.0, (ess / p) / (rss / dof_residual))
    else:
        f_stat = float('nan')

    # P-value của F-test được tính từ đầu qua hàm beta không đầy đủ chuẩn hóa.
    f_pvalue = float('nan')
    if dof_residual > 0 and p > 0:
        f_pvalue = f_survival_probability(f_stat, p, dof_residual)

    return ModelMetrics(
        n=n,
        p=p,
        rss=rss,
        tss=tss,
        ess=ess,
        r2=r2,
        r2_adj=r2_adj,
        sigma2_hat=sigma2_hat,
        f_statistic=f_stat,
        f_pvalue=f_pvalue,
        rmse=rmse,
    )


@dataclass
class ResidualPlotsData:
    """Dataclass chứa dữ liệu phục vụ 4 đồ thị chẩn đoán phần dư chuẩn trong OLS.

    Bốn đồ thị kiểm tra trực quan các giả thiết Gauss-Markov sau khi ước lượng: 
    residuals vs fitted phát hiện xu hướng phi tuyến và phương sai không đồng nhất; 
    
    Q-Q plot kiểm tra phân phối chuẩn của phần dư (cần thiết cho suy luận thống kê); 
    scale-location giúp phát hiện heteroscedasticity; 
    leverage cho biết quan sát nào có ảnh hưởng lớn (influential points). 
    """
    residuals: List[float]                        # Vector phần dư e = y - ŷ
    fitted_values: List[float]                    # Giá trị dự báo ŷ = Xβ̂
    standardized_residuals: List[float]           # Phần dư chuẩn hóa e_i / s_e
    qqplot_data: Tuple[List[float], List[float]]  # (quantile lý thuyết, quantile mẫu)


def residual_plots(
    X: List[List[float]],
    y: List[float],
    beta_hat: List[float]
) -> ResidualPlotsData:
    """Tạo dữ liệu cho 4 đồ thị chẩn đoán phần dư chuẩn của mô hình OLS.

    Hàm này tính toán các mảng số cần thiết để vẽ 4 đồ thị chẩn đoán theo quy
    ước thống kê: 
        (1) Residuals vs Fitted phát hiện phi tuyến hoặc vi phạm GM4,
        (2) Q-Q plot kiểm tra phân phối chuẩn của phần dư (giả thiết GM5 cho suy luận), 
        (3) Scale-Location phát hiện heteroscedasticity bằng cách nhìn vào
            sqrt(|e_std|) vs ŷ, 
        (4) Cook's distance hoặc leverage phát hiện influential
        points. 
        
    Phần dư được chuẩn hóa bằng cách chia cho độ lệch chuẩn mẫu s_e,
    trong khi quantile lý thuyết được tính bằng xấp xỉ inverse normal dựa trên
    vị trí thứ hạng i/(n+1).

    Args:
        X: Ma trận thiết kế kích thước n×(p+1) dạng list 2D.
        y: Vector quan sát kích thước (n,).
        beta_hat: Vector hệ số OLS ước lượng β̂ kích thước (p+1,).

    Returns:
        ResidualPlotsData chứa phần dư, giá trị dự báo, phần dư chuẩn hóa và
        dữ liệu Q-Q plot.

    Raises:
        ValueError: Khi độ dài beta_hat không khớp với số cột của X.
    """
    n = len(y)
    if len(beta_hat) != len(X[0]):
        raise ValueError("beta_hat length must match number of columns in X")

    # Calculate fitted values
    fitted = [sum(X[i][j] * beta_hat[j] for j in range(len(beta_hat))) for i in range(n)]

    # Calculate residuals
    residuals = [y[i] - fitted[i] for i in range(n)]

    # Tính độ lệch chuẩn mẫu của phần dư để chuẩn hóa
    residual_mean = sum(residuals) / n
    residual_var = sum((e - residual_mean)**2 for e in residuals) / (n - 1)
    residual_sd = sqrt(residual_var) if residual_var > 0 else 1.0

    # Chuẩn hóa phần dư bằng cách chia cho s_e; dùng để phát hiện outlier (|e_std| > 2)
    standardized = [e / residual_sd for e in residuals]

    # Q-Q plot so sánh quantile mẫu (được sắp xếp) với quantile lý thuyết chuẩn;
    # nếu phần dư có phân phối chuẩn, các điểm sẽ nằm gần đường thẳng y = x
    sorted_std_residuals = sorted(standardized)
    theoretical_quantiles = _normal_quantiles(n)

    qqplot_data = (theoretical_quantiles, sorted_std_residuals)

    return ResidualPlotsData(
        residuals=residuals,
        fitted_values=fitted,
        standardized_residuals=standardized,
        qqplot_data=qqplot_data,
    )


def _normal_quantiles(n: int) -> List[float]:
    """Tạo danh sách quantile lý thuyết của phân phối chuẩn N(0,1) để dùng trong Q-Q plot.

    Với n quan sát, quantile thứ i được tính tại xác suất p = i/(n+1) theo quy
    ước Hazen (tránh p = 0 và p = 1). Nghịch đảo của hàm phân phối tích lũy N(0,1),
    tức hàm probit Φ⁻¹(p), được tính bằng hàm trợ giúp _norm_ppf cài đặt thuật toán
    hữu tỉ Acklam với sai số tuyệt đối dưới 1.15e-9, nhờ đó nhóm không phụ thuộc vào
    thư viện ngoài mà vẫn giữ độ chính xác đủ dùng cho Q-Q plot.

    Args:
        n: Số lượng quantile cần tạo, bằng số quan sát trong mẫu.

    Returns:
        Danh sách n giá trị quantile lý thuyết N(0,1), được sắp xếp tăng dần.
    """
    quantiles = []
    for i in range(1, n + 1):
        p = i / (n + 1)
        if p <= 0 or p >= 1:
            quantiles.append(0.0)
        else:
            quantiles.append(_norm_ppf(p))
    return quantiles


def _norm_ppf(p: float) -> float:
    """Xấp xỉ hàm phân vị (probit) Φ⁻¹(p) của phân phối chuẩn chuẩn tắc N(0,1).

    Hàm cài đặt thuật toán hữu tỉ của Peter Acklam: trước hết chia miền (0, 1)
    thành ba vùng gồm đuôi dưới, vùng trung tâm và đuôi trên, sau đó dùng các đa
    thức bậc cao đã được hiệu chỉnh hệ số cho từng vùng, nhờ đó đạt sai số tuyệt
    đối dưới 1.15e-9 mà chỉ cần các phép toán cơ bản của math chuẩn. 

    Args:
        p: Xác suất tích lũy trong khoảng mở (0, 1).

    Returns:
        Giá trị z sao cho Φ(z) = p.
    """
    from math import log

    # Hệ số của các đa thức xấp xỉ, theo công bố gốc của Acklam.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00]
    p_low, p_high = 0.02425, 1.0 - 0.02425

    if p < p_low:                       # Đuôi dưới: dùng biến đổi căn logarit
        q = sqrt(-2.0 * log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:                   # Vùng trung tâm: đa thức hữu tỉ bậc năm
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:                               # Đuôi trên: đối xứng với đuôi dưới
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
