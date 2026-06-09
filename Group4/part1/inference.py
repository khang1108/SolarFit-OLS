"""
File thực hiện suy luận thống kê cho các hệ số hồi quy OLS.

File này cung cấp hai hàm chính: coef_inference thực hiện suy luận cho từng
hệ số β_j bằng cách tính sai số chuẩn se(β̂_j) = σ̂·sqrt[(X'X)^{-1}_{jj}],
thống kê t, p-value hai phía và khoảng tin cậy (1-α)·100%; và vif tính Variance
Inflation Factor để phát hiện và đo lường mức độ multicollinearity.
"""

from dataclasses import dataclass
from typing import List
from math import inf, isfinite, sqrt
try:
    from .statistical_distributions import student_t_critical, student_t_two_sided_pvalue
except ImportError:  # Cho phép chạy trực tiếp file
    from statistical_distributions import student_t_critical, student_t_two_sided_pvalue


@dataclass
class CoefficientInference:
    """Dataclass chứa kết quả suy luận thống kê cho từng hệ số hồi quy.

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

    Mô hình tuyến tính với GM1-GM5, ma trận phương sai-hiệp phương sai của
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

    Raises:
        ValueError: Khi dữ liệu đầu vào không hợp lệ hoặc không còn bậc tự do
            phần dư để thực hiện suy luận.
    """
    _validate_inference_inputs(X, y, beta_hat, sigma2, alpha)

    n = len(y)
    p = len(beta_hat) - 1  # number of features (excluding hệ số tự do)

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
    except ValueError:
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
    t_crit = student_t_critical(alpha / 2, dof)

    # CI_j = β̂_j ± t_crit·se(β̂_j): khoảng plausible cho β_j với độ tin cậy (1-α)·100%
    ci_lower = []
    ci_upper = []
    for i in range(k):
        margin = t_crit * std_errors[i]
        ci_lower.append(beta_hat[i] - margin)
        ci_upper.append(beta_hat[i] + margin)

    # P-value hai phía: P(|T_{n-p-1}| > |t_j|), nhỏ hơn α thì bác bỏ H₀: β_j = 0
    p_values = [student_t_two_sided_pvalue(t_stats[i], dof) for i in range(k)]

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
    """Dataclass chứa kết quả phân tích Variance Inflation Factor cho tất cả biến."""

    vif_values: List[float]          # VIF_j cho từng cột của X
    column_names: List[str]          # Tên cột tương ứng
    max_vif: float                   # Giá trị VIF lớn nhất trong mô hình
    has_multicollinearity: bool      # True nếu max_vif > threshold


def vif(X: List[List[float]], threshold: float = 10.0) -> VIFResult:
    """Tính Variance Inflation Factor cho mỗi biến trong ma trận thiết kế.

    VIF_j = 1/(1 - R²_j) trong đó R²_j là hệ số xác định từ hồi quy X_j lên
    tất cả các cột còn lại của X. VIF_j = 1 nghĩa là không có multicollinearity;
    VIF_j → ∞ khi R²_j → 1.
    
    VIF cao, các β̂_j trở nên không ổn định và khoảng tin cậy rất rộng, làm mất
    ý nghĩa thống kê. 

    Args:
        X: Ma trận thiết kế kích thước n×(p+1) dạng list 2D, bao gồm cả cột hệ số tự do.
        threshold: Ngưỡng VIF để gắn cờ multicollinearity, mặc định là 10.0.

    Returns:
        VIFResult chứa VIF cho từng cột, tên cột, VIF lớn nhất và cờ multicollinearity.

    Raises:
        ValueError: Khi X rỗng, các hàng không cùng số cột hoặc threshold không dương.
    """
    _validate_design_matrix(X)
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    n = len(X)
    p = len(X[0])

    vif_values = []
    for j in range(p):
        # Hồi quy phụ trợ X_j theo tất cả cột còn lại, sau đó dùng
        # VIF_j = 1 / (1 - R²_j). Cột hằng như hệ số tự do có VIF không xác định.
        r2_j = _calculate_r2_excluding_col(X, j)
        if not isfinite(r2_j):
            vif_j = float("nan")
        elif r2_j >= 1.0 - 1e-12:
            vif_j = inf
        else:
            # Chặn R² tại 0 để hạn chế sai số làm tròn rất nhỏ khiến VIF < 1.
            vif_j = 1.0 / (1.0 - max(0.0, r2_j))
        vif_values.append(vif_j)

    # Bỏ qua NaN của hệ số tự do nhưng giữ inf vì đó là dấu hiệu đa cộng tuyến hoàn hảo.
    comparable_vifs = [value for value in vif_values if value == value]
    max_vif = max(comparable_vifs) if comparable_vifs else float("nan")
    has_multicollinearity = max_vif > threshold if max_vif == max_vif else False

    col_names = [f"X{i}" for i in range(p)]

    return VIFResult(
        vif_values=vif_values,
        column_names=col_names,
        max_vif=max_vif,
        has_multicollinearity=has_multicollinearity,
    )


def _matrix_inverse(A: List[List[float]]) -> List[List[float]]:
    """Tính nghịch đảo ma trận vuông A bằng phương pháp Gauss-Jordan với partial pivoting.

    Hàm tính (X'X)^{-1} trong coef_inference, phục vụ tính sai số chuẩn và khoảng tin cậy. 

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

    Tính VIF: R²_j từ hồi quy X_j ~ các X khác cho biết
    mức độ X_j có thể được Prediction tuyến tính từ các biến còn lại. 

    Args:
        X: Ma trận thiết kế đầy đủ kích thước n×(p+1).
        exclude_col: Chỉ số cột cần làm biến phụ thuộc trong hồi quy phụ trợ.

    Returns:
        Giá trị R² của hồi quy phụ trợ; trả về NaN cho cột hằng và 1.0 khi
        ma trận hồi quy phụ trợ suy biến do đa cộng tuyến hoàn hảo.
    """
    n = len(X)
    p = len(X[0])
    if exclude_col < 0 or exclude_col >= p:
        raise ValueError("exclude_col is outside the design matrix")

    # Tách cột đang xét làm biến phụ thuộc và giữ các cột khác làm biến giải thích.
    y_col = [row[exclude_col] for row in X]
    X_others = [
        [value for column, value in enumerate(row) if column != exclude_col]
        for row in X
    ]

    # SST bằng 0 đối với cột hằng, điển hình là hệ số tự do. Khi đó R² và VIF
    # không được định nghĩa nên trả NaN thay vì gán một giá trị gây hiểu nhầm.
    y_mean = sum(y_col) / n
    ss_tot = sum((value - y_mean) ** 2 for value in y_col)
    if ss_tot <= 1e-15:
        return float("nan")

    # Nếu không còn biến giải thích, dự báo tốt nhất trong mô hình rỗng là 0.
    if not X_others[0]:
        ss_res = sum(value**2 for value in y_col)
        return 1.0 - ss_res / ss_tot

    # Tự xây dựng normal equations (Z'Z)γ = Z'y cho hồi quy phụ trợ.
    q = len(X_others[0])
    ZtZ = [[0.0] * q for _ in range(q)]
    Zty = [0.0] * q
    for i in range(q):
        Zty[i] = sum(X_others[row][i] * y_col[row] for row in range(n))
        for j in range(q):
            ZtZ[i][j] = sum(
                X_others[row][i] * X_others[row][j] for row in range(n)
            )

    try:
        gamma = _matrix_vector_multiply(_matrix_inverse(ZtZ), Zty)
    except ValueError:
        # Z'Z suy biến nghĩa là các biến giải thích trong hồi quy phụ trợ phụ
        # thuộc tuyến tính hoàn hảo; toàn bộ thiết kế đang có đa cộng tuyến hoàn hảo.
        return 1.0

    y_pred = [
        sum(X_others[row][column] * gamma[column] for column in range(q))
        for row in range(n)
    ]
    ss_res = sum((y_col[row] - y_pred[row]) ** 2 for row in range(n))

    # Sai số số học có thể làm R² hơi vượt khỏi [0, 1], nên chặn về miền hợp lệ.
    return min(1.0, max(0.0, 1.0 - ss_res / ss_tot))


def _matrix_vector_multiply(
    matrix: List[List[float]], vector: List[float]
) -> List[float]:
    """Nhân ma trận với vector bằng Python thuần."""
    if not matrix or len(matrix[0]) != len(vector):
        raise ValueError("matrix and vector dimensions are incompatible")
    return [
        sum(matrix[row][column] * vector[column] for column in range(len(vector)))
        for row in range(len(matrix))
    ]


def _validate_design_matrix(X: List[List[float]]) -> None:
    """Kiểm tra ma trận thiết kế không rỗng, hữu hạn và có dạng chữ nhật."""
    if not X or not X[0]:
        raise ValueError("X must be a non-empty design matrix")
    width = len(X[0])
    if any(len(row) != width for row in X):
        raise ValueError("all rows of X must have the same number of columns")
    if any(not isfinite(value) for row in X for value in row):
        raise ValueError("X must contain only finite values")


def _validate_inference_inputs(
    X: List[List[float]],
    y: List[float],
    beta_hat: List[float],
    sigma2: float,
    alpha: float,
) -> None:
    """Kiểm tra các điều kiện cần trước khi thực hiện suy luận OLS."""
    _validate_design_matrix(X)
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of observations")
    if any(not isfinite(value) for value in y):
        raise ValueError("y must contain only finite values")
    if len(beta_hat) != len(X[0]):
        raise ValueError("beta_hat length must match the number of columns in X")
    if any(not isfinite(value) for value in beta_hat):
        raise ValueError("beta_hat must contain only finite values")
    if sigma2 < 0.0 or not isfinite(sigma2):
        raise ValueError("sigma2 must be a finite non-negative number")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if len(y) - len(beta_hat) <= 0:
        raise ValueError("residual degrees of freedom must be positive")
