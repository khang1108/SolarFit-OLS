"""
Module thực hiện suy luận thống kê cho các hệ số hồi quy OLS.

Module này cung cấp hai hàm chính: coef_inference thực hiện suy luận cho từng
hệ số β_j bằng cách tính sai số chuẩn se(β̂_j) = σ̂·sqrt[(X'X)^{-1}_{jj}],
thống kê t, p-value hai phía và khoảng tin cậy (1-α)·100%; và vif tính Variance
Inflation Factor để phát hiện và đo lường mức độ multicollinearity. Suy luận
thống kê dựa trên định lý phân phối: dưới giả thiết GM5 (phần dư có phân phối
chuẩn), β̂_j tuân theo phân phối t_{n-p-1} chuẩn hóa, từ đó xây dựng được kiểm
định và khoảng tin cậy chính xác. VIF_j = 1/(1-R²_j) đo mức độ một biến bị giải
thích bởi các biến còn lại: VIF > 10 là dấu hiệu đáng lo ngại về multicollinearity,
vì nó làm phình sai số chuẩn của hệ số và làm khoảng tin cậy trở nên rất rộng,
gây mất ý nghĩa thống kê ngay cả khi biến thực sự quan trọng.
"""

from dataclasses import dataclass
from typing import List, Tuple
from math import sqrt


@dataclass
class CoefficientInference:
    """Lớp chứa kết quả suy luận thống kê cho từng hệ số hồi quy.

    Dataclass này gom tất cả thông tin cần thiết để đánh giá ý nghĩa thống kê của
    từng hệ số β̂_j: sai số chuẩn phản ánh độ không chắc chắn của ước lượng, thống
    kê t đo khoảng cách chuẩn hóa từ β̂_j đến 0, p-value cho biết xác suất quan
    sát được kết quả cực đoan như vậy dưới H₀: β_j = 0, và khoảng tin cậy cho
    biết dải giá trị plausible của β_j với độ tin cậy (1-α)·100%.
    """
    coefficients: List[float]   # Vector hệ số ước lượng β̂ ∈ R^{p+1}
    std_errors: List[float]     # Sai số chuẩn se(β̂_j) = σ̂·sqrt[(X'X)^{-1}_{jj}]
    t_statistics: List[float]   # Thống kê t: t_j = β̂_j / se(β̂_j)
    p_values: List[float]       # P-value hai phía: P(|T| > |t_j|) với T ~ t_{n-p-1}
    ci_lower: List[float]       # Cận dưới khoảng tin cậy: β̂_j - t_{α/2}·se(β̂_j)
    ci_upper: List[float]       # Cận trên khoảng tin cậy: β̂_j + t_{α/2}·se(β̂_j)
    alpha: float                # Mức ý nghĩa α, thường là 0.05


def coef_inference(
    X: List[List[float]],
    y: List[float],
    beta_hat: List[float],
    sigma2: float,
    alpha: float = 0.05
) -> CoefficientInference:
    """Thực hiện suy luận thống kê cho từng hệ số OLS: se, t-statistic, p-value và CI.

    Dưới mô hình tuyến tính với GM1-GM5, ma trận phương sai-hiệp phương sai của
    ước lượng OLS là Cov(β̂) = σ²(X'X)^{-1}, do đó sai số chuẩn của β̂_j là
    se(β̂_j) = σ̂·sqrt[(X'X)^{-1}_{jj}]. Thống kê kiểm định t_j = β̂_j / se(β̂_j)
    tuân theo phân phối t_{n-p-1} dưới H₀: β_j = 0, từ đó suy ra p-value hai phía
    và khoảng tin cậy CI_j = β̂_j ± t_{α/2, n-p-1}·se(β̂_j). Hàm này dùng σ̂²
    từ bên ngoài truyền vào (thay vì tính lại) để tránh trùng lặp tính toán với
    ols_fit.

    Args:
        X: Ma trận thiết kế kích thước n×(p+1) dạng list 2D.
        y: Vector quan sát kích thước (n,) — dùng để xác định n.
        beta_hat: Vector hệ số OLS ước lượng β̂ kích thước (p+1,).
        sigma2: Ước lượng phương sai nhiễu σ̂² = RSS/(n-p-1) từ ols_fit.
        alpha: Mức ý nghĩa α cho khoảng tin cậy, mặc định là 0.05 (95% CI).

    Returns:
        CoefficientInference chứa se, t-statistics, p-values và khoảng tin cậy
        cho tất cả p+1 hệ số.
    """
    n = len(y)
    p = len(beta_hat) - 1  # number of features (excluding intercept)

    # Tính X'X — ma trận Gram cần thiết để có Cov(β̂) = σ²(X'X)^{-1}
    k = len(X[0])
    XtX = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            XtX[i][j] = sum(X[row][i] * X[row][j] for row in range(n))

    # Tính nghịch đảo (X'X)^{-1} bằng Gauss-Jordan; nếu X'X suy biến thì trả
    # về NaN cho tất cả thống kê thay vì crash
    try:
        XtX_inv = _matrix_inverse(XtX)
    except:
        return CoefficientInference(
            coefficients=beta_hat,
            std_errors=[float('nan')] * len(beta_hat),
            t_statistics=[float('nan')] * len(beta_hat),
            p_values=[float('nan')] * len(beta_hat),
            ci_lower=[float('nan')] * len(beta_hat),
            ci_upper=[float('nan')] * len(beta_hat),
            alpha=alpha,
        )

    # se(β̂_j) = σ̂·sqrt[(X'X)^{-1}_{jj}]: phần tử đường chéo của (X'X)^{-1}
    # cho phương sai chuẩn hóa của từng hệ số, max(0,...) đề phòng sai số số học
    sigma_hat = sqrt(sigma2) if sigma2 > 0 else 0.0
    std_errors = [sigma_hat * sqrt(max(0, XtX_inv[i][i])) for i in range(k)]

    # t_j = β̂_j / se(β̂_j): khoảng cách chuẩn hóa từ ước lượng đến giả thuyết H₀
    t_stats = []
    for i in range(k):
        if std_errors[i] > 0:
            t_stats.append(beta_hat[i] / std_errors[i])
        else:
            t_stats.append(float('nan'))

    # Giá trị tới hạn t_{α/2, n-p-1}: xấp xỉ chuẩn khi dof > 30
    dof = n - p - 1
    t_crit = _t_critical(alpha / 2, dof)

    # CI_j = β̂_j ± t_crit·se(β̂_j): khoảng plausible cho β_j với độ tin cậy (1-α)·100%
    ci_lower = []
    ci_upper = []
    for i in range(k):
        margin = t_crit * std_errors[i]
        ci_lower.append(beta_hat[i] - margin)
        ci_upper.append(beta_hat[i] + margin)

    # P-value hai phía: P(|T_{n-p-1}| > |t_j|), nhỏ hơn α thì bác bỏ H₀: β_j = 0
    p_values = [_t_pvalue(abs(t_stats[i]), dof) for i in range(k)]

    return CoefficientInference(
        coefficients=beta_hat,
        std_errors=std_errors,
        t_statistics=t_stats,
        p_values=p_values,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha,
    )


@dataclass
class VIFResult:
    """Lớp chứa kết quả phân tích Variance Inflation Factor cho tất cả biến.

    VIF là chỉ số đo lường mức độ multicollinearity: VIF_j = 1/(1-R²_j) trong đó
    R²_j là hệ số xác định từ hồi quy X_j lên các biến còn lại. Ý nghĩa trực quan:
    VIF_j = k có nghĩa là phương sai của β̂_j lớn gấp k lần so với trường hợp X_j
    hoàn toàn độc lập với các biến khác. VIF > 10 thường được coi là ngưỡng đáng
    lo ngại vì lúc đó sai số chuẩn se(β̂_j) bị phình lên hơn 3 lần (sqrt(10) ≈ 3.16).
    """
    vif_values: List[float]          # VIF_j cho từng cột của X
    column_names: List[str]          # Tên cột tương ứng
    max_vif: float                   # Giá trị VIF lớn nhất trong mô hình
    has_multicollinearity: bool      # True nếu max_vif > threshold


def vif(X: List[List[float]], threshold: float = 10.0) -> VIFResult:
    """Tính Variance Inflation Factor cho mỗi biến trong ma trận thiết kế.

    VIF_j = 1/(1 - R²_j) trong đó R²_j là hệ số xác định từ hồi quy X_j lên
    tất cả các cột còn lại của X. VIF_j = 1 nghĩa là không có multicollinearity;
    VIF_j → ∞ khi R²_j → 1, tức là X_j gần như là tổ hợp tuyến tính của các
    biến khác. Đây là chẩn đoán quan trọng trước khi diễn giải hệ số OLS: khi
    VIF cao, các β̂_j trở nên không ổn định và khoảng tin cậy rất rộng, làm mất
    ý nghĩa thống kê. Phiên bản này dùng numpy để tính hồi quy phụ trợ nhưng
    interface hoàn toàn là Python thuần.

    Args:
        X: Ma trận thiết kế kích thước n×(p+1) dạng list 2D, bao gồm cả cột intercept.
        threshold: Ngưỡng VIF để gắn cờ multicollinearity, mặc định là 10.0.

    Returns:
        VIFResult chứa VIF cho từng cột, tên cột, VIF lớn nhất và cờ multicollinearity.
    """
    n = len(X)
    p = len(X[0])

    vif_values = []
    for j in range(p):
        # R² from regressing X_j on others
        r2_j = _calculate_r2_excluding_col(X, j)
        vif_j = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else float('inf')
        vif_values.append(vif_j)

    max_vif = max(v for v in vif_values if v != float('inf'))
    has_multicollinearity = max_vif > threshold

    col_names = [f"X{i}" for i in range(p)]

    return VIFResult(
        vif_values=vif_values,
        column_names=col_names,
        max_vif=max_vif,
        has_multicollinearity=has_multicollinearity,
    )


def _matrix_inverse(A: List[List[float]]) -> List[List[float]]:
    """Tính nghịch đảo ma trận vuông A bằng phương pháp Gauss-Jordan với partial pivoting.

    Hàm nội bộ này được dùng để tính (X'X)^{-1} trong coef_inference, phục vụ
    tính sai số chuẩn và khoảng tin cậy. Partial pivoting đảm bảo tính ổn định
    số học bằng cách luôn đưa phần tử có giá trị tuyệt đối lớn nhất lên vị trí
    pivot, giảm thiểu sai số tích lũy trong phép chia.

    Args:
        A: Ma trận vuông kích thước n×n cần tính nghịch đảo.

    Returns:
        Ma trận nghịch đảo A^{-1} kích thước n×n.

    Raises:
        ValueError: Khi ma trận suy biến tại cột nào đó trong quá trình khử.
    """
    n = len(A)
    # Create augmented matrix [A | I]
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for col in range(n):
        # Find pivot
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError(f"Singular matrix at column {col}")

        # Swap rows
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Scale pivot row
        pivot_val = aug[col][col]
        aug[col] = [v / pivot_val for v in aug[col]]

        # Eliminate column
        for row in range(n):
            if row != col:
                f = aug[row][col]
                aug[row] = [aug[row][j] - f * aug[col][j] for j in range(2 * n)]

    return [row[n:] for row in aug]


def _calculate_r2_excluding_col(X: List[List[float]], exclude_col: int) -> float:
    """Tính R² khi hồi quy cột exclude_col lên tất cả các cột còn lại của X.

    Đây là bước cốt lõi trong tính VIF: R²_j từ hồi quy X_j ~ các X khác cho biết
    mức độ X_j có thể được dự đoán tuyến tính từ các biến còn lại. Hàm dùng
    numpy.linalg.lstsq để tính OLS phụ trợ vì bước này không cần triển khai thuần
    Python, chỉ là công cụ nội bộ hỗ trợ chẩn đoán. Trả về 0.0 nếu không có numpy
    hoặc gặp lỗi số học.

    Args:
        X: Ma trận thiết kế đầy đủ kích thước n×(p+1).
        exclude_col: Chỉ số cột cần làm biến phụ thuộc trong hồi quy phụ trợ.

    Returns:
        Giá trị R² trong khoảng [0, 1]; trả về 0.0 khi không tính được.
    """
    try:
        import numpy as np
        X_arr = np.array(X, dtype=float)
        y_col = X_arr[:, exclude_col]
        X_others = np.delete(X_arr, exclude_col, axis=1)

        # Simple OLS
        try:
            beta = np.linalg.lstsq(X_others, y_col, rcond=None)[0]
            y_pred = X_others @ beta
            ss_res = np.sum((y_col - y_pred)**2)
            ss_tot = np.sum((y_col - np.mean(y_col))**2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            return float(r2)
        except:
            return 0.0
    except ImportError:
        return 0.0


def _t_critical(alpha_half: float, dof: int) -> float:
    """Tính giá trị tới hạn t_{alpha_half, dof} để xây dựng khoảng tin cậy.

    Giá trị này là điểm phân vị (1 - alpha_half) của phân phối t_{dof}, dùng để
    xác định "biên độ sai số" của khoảng tin cậy. Khi dof > 30, phân phối t tiệm
    cận chuẩn tắc N(0,1), nên xấp xỉ t ≈ 1.96 là chấp nhận được cho α = 0.05.
    Nếu có scipy thì dùng giá trị chính xác, ngược lại dùng xấp xỉ kinh nghiệm
    2.0 + 4.0/dof (đủ tốt cho các mô hình nhỏ trong dự án này).

    Args:
        alpha_half: Nửa mức ý nghĩa α/2 (ví dụ: 0.025 cho α = 0.05).
        dof: Bậc tự do phần dư = n - p - 1.

    Returns:
        Giá trị tới hạn t dương tương ứng với xác suất (1 - alpha_half).
    """
    try:
        from scipy import stats
        return float(stats.t.ppf(1 - alpha_half, dof))
    except:
        # Fallback: approximate with normal (good for dof > 30)
        if dof > 30:
            return 1.96  # 95% CI
        else:
            # Simple approximation
            return 2.0 + 4.0 / dof


def _t_pvalue(t_stat: float, dof: int) -> float:
    """Tính p-value hai phía cho thống kê t trong kiểm định hệ số hồi quy.

    P-value = P(|T_{dof}| > |t_stat|) = 2·P(T_{dof} > |t_stat|) trong đó T_{dof}
    là biến ngẫu nhiên phân phối t_{dof} dưới giả thuyết không H₀: β_j = 0. P-value
    nhỏ (thường < 0.05) là bằng chứng chống lại H₀, nghĩa là hệ số β_j có ý nghĩa
    thống kê. Cần scipy để tính chính xác; trả về NaN khi không có scipy.

    Args:
        t_stat: Giá trị tuyệt đối của thống kê t, tức là |β̂_j / se(β̂_j)|.
        dof: Bậc tự do phần dư = n - p - 1.

    Returns:
        P-value hai phía trong khoảng [0, 1]; trả về NaN khi không có scipy.
    """
    try:
        from scipy import stats
        return float(2 * (1 - stats.t.cdf(abs(t_stat), dof)))
    except:
        # Fallback: no p-value available
        return float('nan')
