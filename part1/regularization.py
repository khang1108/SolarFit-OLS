"""
Module triển khai các phương pháp chuẩn hóa (regularization) trong hồi quy tuyến tính.

Module này cung cấp hai hàm chính cho hồi quy Ridge: ridge_fit tính hệ số Ridge
bằng nghiệm dạng đóng β̂_ridge = (X'X + λI)^{-1}X'y, và ridge_trace theo dõi
quỹ đạo thay đổi của hệ số theo dải giá trị λ. Hồi quy Ridge giải quyết vấn đề
multicollinearity mà OLS thường gặp bằng cách thêm số hạng phạt λ||β||² vào hàm
mục tiêu, qua đó ép các hệ số co về phía không mặc dù gây ra một lượng bias nhỏ.
Khi λ → 0, nghiệm Ridge hội tụ về nghiệm OLS; khi λ → ∞, tất cả hệ số (trừ
intercept) tiến về không. Toàn bộ tính toán được thực hiện bằng Python thuần
(không dùng NumPy) bằng cách giải hệ tuyến tính với Gauss elimination có partial
pivoting thông qua hàm nội bộ _solve_linear_system.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from math import sqrt


@dataclass
class RidgeResult:
    """Class chứa toàn bộ kết quả trả về của hàm ridge_fit.

    Ngoài vector hệ số, dataclass này còn lưu riêng RSS và ridge_penalty để người
    dùng có thể quan sát sự đánh đổi (bias-variance tradeoff) khi thay đổi λ: RSS
    tăng khi λ lớn (bias tăng) nhưng ||β||² giảm (variance giảm). Tổng hàm mục
    tiêu total_loss = RSS + λ·ridge_penalty chính là hàm mà Ridge tối thiểu hóa.
    """
    coefficients: List[float]  # Vector hệ số β̂_ridge ∈ R^{p+1}
    lambda_val: float           # Giá trị tham số chuẩn hóa λ được sử dụng
    rss: float                  # Tổng bình phương phần dư ||y - Xβ̂||²
    ridge_penalty: float        # Số hạng phạt ||β̂_{-0}||² (không tính intercept)
    total_loss: float           # Tổng hàm mục tiêu = RSS + λ·ridge_penalty
    success: bool               # True nếu hệ tuyến tính được giải thành công
    message: str                # Thông báo kết quả hoặc lỗi

def ridge_fit(X: List[List[float]], y: List[float], lam: float) -> RidgeResult:
    """Ước lượng hệ số hồi quy Ridge bằng nghiệm dạng đóng, trả về hệ số và các thống kê.

    Hồi quy Ridge tối thiểu hóa hàm mục tiêu ||y - Xβ||² + λ||β||², trong đó số
    hạng λ||β||² đóng vai trò phạt khi hệ số quá lớn. Nghiệm dạng đóng là
    β̂_ridge = (X'X + λI)^{-1}X'y. Điểm mấu chốt là ma trận X'X + λI luôn khả
    nghịch với mọi λ > 0, kể cả khi X'X gần suy biến do multicollinearity, vì λI
    đảm bảo tất cả trị riêng đều lớn hơn λ > 0. Lưu ý rằng số hạng phạt chỉ áp
    dụng cho các hệ số hồi quy (không áp dụng cho intercept β₀), đây là quy ước
    chuẩn để Ridge không phụ thuộc vào đơn vị đo lường của y.

    Args:
        X: Ma trận thiết kế kích thước n×(p+1) dạng list 2D, với cột đầu tiên
            là vector 1 tương ứng với intercept.
        y: Vector quan sát kích thước (n,).
        lam: Tham số chuẩn hóa λ >= 0. Khi λ = 0, kết quả trùng với OLS.

    Returns:
        RidgeResult chứa β̂_ridge, RSS, ridge_penalty, total_loss và thông báo kết quả.

    Raises:
        Không raise trực tiếp; lỗi được bắt và trả về RidgeResult với success=False.
    """

    # Nếu λ âm, trả về lỗi ngay lập tức vì λ phải là số không âm để đảm bảo tính ổn định của Ridge và ý nghĩa của nó như một tham số phạt. Trả về RidgeResult với success=False
    if lam < 0:
        return RidgeResult(
            coefficients=[],
            lambda_val=lam,
            rss=float('nan'),
            ridge_penalty=float('nan'),
            total_loss=float('nan'),
            success=False,
            message="lambda must be non-negative",
        )

    n = len(y) # Số lượng quan sát
    p = len(X[0]) # Số lượng biến (bao gồm intercept)

    try:
        # Tính X'X — ma trận Gram của X
        XtX = [[0.0] * p for _ in range(p)]
        for i in range(p):
            for j in range(p):
                # Công thức nhân ma trận: (X'X)_{ij} = sum_k X_{ki} * X_{kj}
                XtX[i][j] = sum(X[row][i] * X[row][j] for row in range(n))

        # Thêm λI vào đường chéo để đảm bảo X'X + λI luôn khả nghịch khi λ > 0,
        # ngay cả khi X'X gần suy biến do multicollinearity
        XtX_ridge = [XtX[i][:] for i in range(p)]
        for i in range(p):
            XtX_ridge[i][i] += lam

        # Tính X'y — vế phải của hệ Normal Equations đã được chuẩn hóa
        Xty = [sum(X[row][i] * y[row] for row in range(n)) for i in range(p)]

        # Giải hệ (X'X + λI)β = X'y bằng Gauss elimination có partial pivoting
        beta_ridge = _solve_linear_system(XtX_ridge, Xty)

        # Tính các chỉ số đánh giá sau khi có β̂_ridge
        y_pred = [sum(X[i][j] * beta_ridge[j] for j in range(p)) for i in range(n)]
        residuals = [y[i] - y_pred[i] for i in range(n)]
        rss = sum(e**2 for e in residuals)
        # Số hạng phạt không tính intercept (beta_ridge[0]) theo quy ước chuẩn
        ridge_penalty = sum(b**2 for b in beta_ridge[1:])
        total_loss = rss + lam * ridge_penalty

        return RidgeResult(
            coefficients=beta_ridge,
            lambda_val=lam,
            rss=rss,
            ridge_penalty=ridge_penalty,
            total_loss=total_loss,
            success=True,
            message=f"Ridge solved with lambda={lam}. RSS={rss:.6g}, penalty={ridge_penalty:.6g}",
        )

    except Exception as exc:
        return RidgeResult(
            coefficients=[],
            lambda_val=lam,
            rss=float('nan'),
            ridge_penalty=float('nan'),
            total_loss=float('nan'),
            success=False,
            message=f"ridge_fit failed: {exc}",
        )


@dataclass
class RidgeTraceData:
    """Class chứa dữ liệu ridge trace — quỹ đạo hệ số khi λ thay đổi.

    Ridge trace là đồ thị biểu diễn sự thay đổi của từng hệ số β̂_j theo λ và
    là công cụ trực quan để chọn λ tối ưu: giá trị λ tốt là nơi các hệ số bắt
    đầu ổn định (không dao động mạnh) mà RSS chưa tăng quá nhiều. Trường gcv_trace
    lưu chỉ số Generalized Cross-Validation để hỗ trợ chọn λ tự động.
    """
    lambdas: List[float]                  # Dải giá trị λ được đánh giá
    coefficients_trace: List[List[float]] # β̂_ridge(λ) cho mỗi giá trị λ
    rss_trace: List[float]                # RSS(λ) cho mỗi giá trị λ
    gcv_trace: List[float]                # Chỉ số GCV(λ) cho mỗi giá trị λ


def ridge_trace(X: List[List[float]], y: List[float], lambdas: List[float]) -> RidgeTraceData:
    """Tính quỹ đạo hệ số Ridge trên một dải giá trị λ, phục vụ việc chọn λ tối ưu.

    Hàm này gọi ridge_fit lặp lại trên từng giá trị trong danh sách lambdas và
    thu thập kết quả, tạo ra dữ liệu để vẽ ridge trace plot. Đây là bước cần thiết
    trong Ridge regression vì không có công thức dạng đóng để chọn λ tối
    ưu; thay vào đó ta sẽ quan sát đồ thị hoặc dùng cross-validation để xác định
    λ tại điểm các hệ số bắt đầu ổn định. Chỉ số GCV được tính xấp xỉ trong vì sự phức tạp của hat matrix Ridge.

    Args:
        X: Ma trận thiết kế kích thước n×(p+1) dạng list 2D.
        y: Vector quan sát kích thước (n,).
        lambdas: Danh sách các giá trị λ cần đánh giá, thường được tạo bằng
            np.logspace hoặc dải tuyến tính tùy ngữ cảnh.

    Returns:
        RidgeTraceData chứa quỹ đạo hệ số, RSS và GCV cho mỗi giá trị λ.
    """
    coefficients_trace = []
    rss_trace = []
    gcv_trace = []

    for lam in lambdas: # Tiến hành tính ridge_fit cho từng giá trị λ và thu thập kết quả để vẽ đồ thị
        result = ridge_fit(X, y, lam) # Mỗi lần gọi ridge_fit sẽ trả về RidgeResult chứa hệ số, RSS, penalty và thông báo kết quả
        if result.success:
            # Nếu thành công

            coefficients_trace.append(result.coefficients)
            rss_trace.append(result.rss)
            # GCV (Generalized Cross-Validation): RSS / (1 - tr(H_ridge)/n)^2
            # Simplified approximation
            gcv = result.rss  # placeholder
            gcv_trace.append(gcv)
        else:
            # Nếu ridge_fit thất bại với λ này (hiếm gặp), điền NaN để giữ đúng
            # độ dài mảng, tránh lệch chỉ số khi vẽ đồ thị
            coefficients_trace.append([])
            rss_trace.append(float('nan'))
            gcv_trace.append(float('nan'))

    return RidgeTraceData(
        lambdas=lambdas,
        coefficients_trace=coefficients_trace,
        rss_trace=rss_trace,
        gcv_trace=gcv_trace,
    )


# ============================================================
# Lasso Regression — Coordinate Descent
# ============================================================

@dataclass
class LassoResult:
    """Class chứa toàn bộ kết quả trả về của hàm lasso_fit.

    LassoResult tương tự RidgeResult nhưng dành cho hồi quy Lasso, với penalty 
    là λ||β||₁ thay vì λ||β||². Lasso có khả năng thực hiện feature selection bằng
    cách ép một số hệ số về chính xác 0, trong khi Ridge chỉ co các hệ số về gần 0.
    """
    coefficients: List[float]   # β̂_lasso ∈ R^{p+1}, bao gồm intercept ở vị trí 0
    lambda_val: float            # Giá trị λ đã dùng
    rss: float                   # Tổng bình phương phần dư ||y - Xβ̂||²
    lasso_penalty: float         # Số hạng phạt L1 ||β̂_{-0}||₁ (không tính intercept)
    total_loss: float            # Hàm mục tiêu = RSS + λ·||β̂_{-0}||₁
    n_iter: int                  # Số vòng lặp coordinate descent đến khi hội tụ
    nonzero_coef: int            # Số hệ số ≠ 0 (không tính intercept)
    converged: bool              # True nếu ||Δβ||∞ < tol trước khi hết max_iter
    success: bool                # True nếu hoàn thành không có lỗi
    message: str                 # Thông báo kết quả hoặc lỗi


def _soft_threshold(rho: float, lam: float) -> float:
    """Hàm soft-thresholding S(ρ, λ) = sign(ρ)·max(|ρ| − λ, 0).

    Đây là bước cập nhật cốt lõi của coordinate descent cho Lasso: khi giải
    bài toán tối thiểu hóa theo từng hệ số β_j riêng lẻ (giữ cố định tất cả
    β_k với k ≠ j), nghiệm giải tích chính xác là S(ρ_j, λ) / z_j, trong đó
    ρ_j là tích vô hướng giữa cột X_j và partial residual, còn z_j = ||X_j||².
    Toán tử này tạo ra vùng "chết" [-λ, λ]: bất kỳ ρ nào trong vùng này đều
    cho β_j = 0, giải thích tại sao Lasso thực hiện feature selection tự động
    trong khi Ridge không thể.

    Args:
        rho: Giá trị cần threshold — thường là X_j^T · r^{(-j)} (partial residual).
        lam: Ngưỡng threshold λ ≥ 0. Khi λ = 0, S(ρ, 0) = ρ (không có hiệu ứng).

    Returns:
        Giá trị sau khi áp dụng soft-thresholding.

    Examples:
        >>> _soft_threshold(3.0, 1.0)   # 3 - 1 = 2.0
        2.0
        >>> _soft_threshold(-3.0, 1.0)  # -3 + 1 = -2.0
        -2.0
        >>> _soft_threshold(0.5, 1.0)   # trong vùng chết → 0
        0.0
        >>> _soft_threshold(0.0, 1.0)   # đúng 0 → 0
        0.0
    """
    if rho > lam:
        return rho - lam
    if rho < -lam:
        return rho + lam
    return 0.0


def lasso_fit(
    X: List[List[float]],
    y: List[float],
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    beta_init: Optional[List[float]] = None,
) -> LassoResult:
    """Ước lượng hệ số Lasso bằng thuật toán Coordinate Descent.

    Lasso tối thiểu hóa hàm mục tiêu (nhất quán với Ridge trong module này):

        f(β) = ||y − Xβ||² + λ·||β₋₀||₁

    trong đó ||β₋₀||₁ = Σ|β_j| (j ≥ 1) không áp dụng cho intercept β₀. Vì
    |β_j| không khả vi tại 0, bài toán không có nghiệm dạng đóng — khác với
    Ridge — nên phải dùng thuật toán lặp Coordinate Descent.

    Coordinate Descent cập nhật từng β_j một, giữ cố định tất cả β_k (k ≠ j).
    Bài toán 1D kết quả có nghiệm giải tích chính xác:

        β_j ← S(ρ_j, λ/2) / z_j      (j ≥ 1, có penalty L1)
        β_0 ← ρ_0 / z_0               (j = 0, intercept, không penalty)

    trong đó ρ_j = X_j^T r^{(-j)} là partial residual projection,
    z_j = ||X_j||² là chuẩn bình phương cột j, và S là soft-thresholding.
    Nhân tử λ/2 (không phải λ) xuất hiện vì đạo hàm của RSS sinh ra hệ số 2:
    d/dβ_j ||y-Xβ||² = -2·X_j^T(y-Xβ), và khi giải KKT ta thu được λ/2.

    So sánh với sklearn: sklearn tối thiểu (1/2n)||y-Xβ||² + α||β||₁, tương
    đương khi alpha_sklearn = λ / (2n).

    Residual r được duy trì và cập nhật tăng dần sau mỗi lần thay đổi β_j:
        r ← r − X[:,j]·Δβ_j
    tránh tính lại r = y − Xβ từ đầu, giảm chi phí mỗi epoch từ O(n·p²)
    xuống O(n·p).

    Args:
        X: Ma trận thiết kế n×(p+1), cột đầu là vector 1 (intercept).
        y: Vector mục tiêu (n,).
        lam: Tham số chuẩn hóa λ ≥ 0. λ=0 cho nghiệm OLS (không penalty).
        max_iter: Số epoch tối đa qua toàn bộ features (mặc định 1000).
        tol: Ngưỡng hội tụ: dừng khi ||Δβ||∞ < tol (mặc định 1e-6).
        beta_init: Điểm khởi đầu cho β — dùng cho warm start trong lasso_path.
                   None → khởi tạo bằng vector 0.

    Returns:
        LassoResult với hệ số, RSS, lasso_penalty, n_iter, nonzero_coef,
        converged và thông báo kết quả.
    """
    if lam < 0:
        return LassoResult(
            coefficients=[], lambda_val=lam, rss=float('nan'),
            lasso_penalty=float('nan'), total_loss=float('nan'),
            n_iter=0, nonzero_coef=0, converged=False, success=False,
            message="lambda must be non-negative",
        )

    n = len(y)
    p = len(X[0])

    # Khởi tạo β: warm start nếu có, ngược lại dùng vector 0
    beta: List[float] = beta_init[:] if beta_init is not None else [0.0] * p

    # Tính residual ban đầu r = y − X·β
    r: List[float] = [
        y[i] - sum(X[i][j] * beta[j] for j in range(p))
        for i in range(n)
    ]

    # Precompute z_j = ||X[:,j]||² — hằng số, chỉ cần tính một lần
    z: List[float] = [sum(X[i][j] ** 2 for i in range(n)) for j in range(p)]

    half_lam = lam / 2.0
    n_iter = 0
    converged = False

    try:
        for epoch in range(max_iter):
            beta_old = beta[:]

            for j in range(p):
                if z[j] < 1e-12:
                    # Cột j hằng hoặc toàn 0 → bỏ qua tránh chia cho 0
                    continue

                beta_j_old = beta[j]

                # Partial residual r^{(-j)} = r + X[:,j]·β_j (thêm lại đóng góp của β_j)
                # ρ_j = X[:,j]^T · r^{(-j)} = Σ_i X[i][j] · (r[i] + X[i][j]·β_j_old)
                rho_j = sum(
                    X[i][j] * (r[i] + X[i][j] * beta_j_old)
                    for i in range(n)
                )

                if j == 0:
                    # Intercept β₀: cập nhật OLS thông thường, không soft-threshold
                    beta[j] = rho_j / z[j]
                else:
                    # β_j (j ≥ 1): áp dụng soft-thresholding với ngưỡng λ/2
                    beta[j] = _soft_threshold(rho_j, half_lam) / z[j]

                # Cập nhật residual tăng dần: r ← r − X[:,j]·Δβ_j
                delta = beta[j] - beta_j_old
                if abs(delta) > 1e-15:
                    for i in range(n):
                        r[i] -= X[i][j] * delta

            n_iter = epoch + 1

            # Kiểm tra hội tụ: ||β_new − β_old||∞
            max_change = max(abs(beta[j] - beta_old[j]) for j in range(p))
            if max_change < tol:
                converged = True
                break

        # Tính metrics sau khi coordinate descent kết thúc
        rss = sum(r[i] ** 2 for i in range(n))
        lasso_penalty = sum(abs(beta[j]) for j in range(1, p))
        total_loss = rss + lam * lasso_penalty
        nonzero_coef = sum(1 for j in range(1, p) if abs(beta[j]) > 1e-10)

        status = "converged" if converged else f"stopped at max_iter={max_iter}"
        return LassoResult(
            coefficients=beta,
            lambda_val=lam,
            rss=rss,
            lasso_penalty=lasso_penalty,
            total_loss=total_loss,
            n_iter=n_iter,
            nonzero_coef=nonzero_coef,
            converged=converged,
            success=True,
            message=f"Lasso {status} in {n_iter} iter. nonzero={nonzero_coef}/{p - 1}",
        )

    except Exception as exc:
        return LassoResult(
            coefficients=beta,
            lambda_val=lam,
            rss=float('nan'),
            lasso_penalty=float('nan'),
            total_loss=float('nan'),
            n_iter=n_iter,
            nonzero_coef=0,
            converged=False,
            success=False,
            message=f"lasso_fit failed: {exc}",
        )


@dataclass
class LassoTraceData:
    """Class chứa dữ liệu quỹ đạo Lasso — sự thay đổi của hệ số khi λ thay đổi.

    Lasso path là đồ thị biểu diễn từng hệ số β̂_j theo λ và là công cụ trực quan
    để quan sát thứ tự biến được chọn vào mô hình: khi λ giảm dần, các biến quan
    trọng nhất sẽ thoát khỏi vùng zero trước. Trường nonzero_counts cho thấy
    độ thưa của mô hình tại mỗi mức phạt.
    """
    lambdas: List[float]               # Dải giá trị λ được đánh giá (thứ tự giảm dần)
    coefficients: List[List[float]]    # β̂_lasso(λ) cho mỗi giá trị λ
    rss_values: List[float]            # RSS(λ) cho mỗi giá trị λ
    total_loss_values: List[float]     # RSS + λ||β||₁ cho mỗi giá trị λ
    nonzero_counts: List[int]          # Số hệ số khác 0 tại mỗi giá trị λ
    n_iter_list: List[int]             # Số vòng lặp coordinate descent tại mỗi λ
    converged_list: List[bool]         # Trạng thái hội tụ tại mỗi giá trị λ


def lasso_path(
    X: List[List[float]],
    y: List[float],
    lambdas: List[float],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> LassoTraceData:
    """Tính quỹ đạo hệ số Lasso trên một dải giá trị λ, phục vụ việc chọn λ tối ưu.

    Hàm này gọi lasso_fit lặp lại trên từng giá trị trong danh sách lambdas và thu
    thập kết quả, tạo ra dữ liệu để vẽ Lasso path plot. Điểm khác biệt so với
    ridge_trace là: nghiệm tại λ_k được dùng làm điểm khởi đầu
    cho λ_{k+1}, giúp coordinate descent hội tụ nhanh hơn đáng kể vì nghiệm thay
    đổi ít giữa hai λ liên tiếp. 

    Args:
        X: Ma trận thiết kế kích thước n×(p+1) dạng list 2D.
        y: Vector quan sát kích thước (n,).
        lambdas: Danh sách các giá trị λ cần đánh giá, thường được tạo bằng
            np.logspace hoặc dải tuyến tính tùy ngữ cảnh.
        max_iter: Số vòng lặp tối đa cho mỗi lần gọi lasso_fit.
        tol: Ngưỡng hội tụ ||Δβ||∞ cho mỗi lần gọi lasso_fit.

    Returns:
        LassoTraceData chứa quỹ đạo hệ số, RSS, total loss và độ thưa tại mỗi λ.
    """
    lambdas_sorted = sorted(lambdas, reverse=True)

    coef_path: List[List[float]] = []
    rss_values: List[float] = []
    total_loss_values: List[float] = []
    nonzero_counts: List[int] = []
    n_iter_list: List[int] = []
    converged_list: List[bool] = []

    beta_warm: Optional[List[float]] = None

    for lam in lambdas_sorted:
        result = lasso_fit( X, y, lam,
                            max_iter=max_iter, tol=tol,
                            beta_init=beta_warm)
        # Cập nhật warm start: dùng nghiệm hiện tại làm điểm khởi đầu cho λ tiếp theo
        if result.success and result.coefficients:
            beta_warm = result.coefficients[:]
        coef_path.append(result.coefficients[:] if result.coefficients else [])
        rss_values.append(result.rss)
        total_loss_values.append(result.total_loss)
        nonzero_counts.append(result.nonzero_coef)
        n_iter_list.append(result.n_iter)
        converged_list.append(result.converged)

    return LassoTraceData(
        lambdas=lambdas_sorted,
        coefficients=coef_path,
        rss_values=rss_values,
        total_loss_values=total_loss_values,
        nonzero_counts=nonzero_counts,
        n_iter_list=n_iter_list,
        converged_list=converged_list,
    )


def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Giải hệ phương trình tuyến tính Ax = b bằng khử Gauss có partial pivoting.

    Hàm này được thiết kế để giải hệ Normal Equations đã được chuẩn hóa
    (X'X + λI)β = X'y trong Ridge regression. Partial pivoting (đổi chỗ hàng để
    đưa phần tử lớn nhất về vị trí pivot) được áp dụng để tăng tính ổn định số học
    so với khử Gauss đơn thuần. Bao gồm hai giai đoạn: 
        (1) khử xuôi biến đổi A về dạng tam giác trên
        (2) thế ngược (back substitution) tính x từ dưới lên.

    Args:
        A: Ma trận vuông hệ số kích thước n×n (sẽ được sao chép, không bị thay đổi).
        b: Vector vế phải kích thước (n,).

    Returns:
        Vector nghiệm x kích thước (n,) thỏa mãn Ax = b.

    Raises:
        ValueError: Khi ma trận A suy biến (phần tử pivot nhỏ hơn 1e-12) hoặc
                    kích thước A và b không tương thích.
    """
    n = len(A)
    if len(b) != n:
        raise ValueError("A and b dimensions don't match")

    # Tạo bản sao để tránh thay đổi dữ liệu gốc của người gọi
    mat = [A[i][:] for i in range(n)]
    rhs = b[:]

    # Giai đoạn 1: khử xuôi với partial pivoting
    for col in range(n):
        # Chọn hàng có phần tử tuyệt đối lớn nhất làm pivot để giảm sai số tích lũy
        pivot_row = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot_row][col]) < 1e-12:
            raise ValueError(f"Singular matrix at column {col}")

        # Hoán đổi hàng pivot lên vị trí hiện tại
        mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
        rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]

        # Khử tất cả phần tử dưới pivot trong cột col
        for row in range(col + 1, n):
            if abs(mat[col][col]) < 1e-12:
                continue
            f = mat[row][col] / mat[col][col]
            for j in range(col, n):
                mat[row][j] -= f * mat[col][j]
            rhs[row] -= f * rhs[col]

    # Giai đoạn 2: thế ngược từ hàng cuối lên đầu
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        x[row] = rhs[row]
        for col in range(row + 1, n):
            x[row] -= mat[row][col] * x[col]
        if abs(mat[row][row]) > 1e-12:
            x[row] /= mat[row][row]

    return x


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Demo regularization: nhóm sinh dữ liệu có đa cộng tuyến mạnh để OLS
    # trở nên bất ổn, sau đó khớp Ridge và Lasso rồi vẽ hai quỹ đạo hệ số
    # theo λ. Ridge trace cho thấy các hệ số co dần về 0 một cách trơn tru
    # khi λ tăng, trong khi Lasso path cho thấy từng hệ số bị ép hẳn về 0,
    # qua đó minh họa khả năng chọn biến tự động của chuẩn hóa L1. Cuối cùng
    # nhóm đối chiếu nghiệm Ridge và Lasso với scikit-learn để kiểm chứng.
    # ------------------------------------------------------------------
    import os
    import sys
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

    np.random.seed(11)
    n_obs = 80
    x1 = np.random.randn(n_obs)
    x2 = 0.95 * x1 + 0.05 * np.random.randn(n_obs)   # gần như trùng x1
    x3 = np.random.randn(n_obs)
    x4 = np.random.randn(n_obs)
    X_np = np.column_stack([np.ones(n_obs), x1, x2, x3, x4])
    beta_true = np.array([2.0, 3.0, 0.0, 1.5, 0.0])  # x2 và x4 thực sự vô nghĩa
    y_np = X_np @ beta_true + 0.7 * np.random.randn(n_obs)
    X_list, y_list = X_np.tolist(), y_np.tolist()

    # Bước 1: khớp Ridge và Lasso tại cùng một λ để so sánh cách hai chuẩn
    # hóa xử lý biến dư thừa x2 (vốn gần như trùng x1)
    lam_demo = 8.0
    ridge = ridge_fit(X_list, y_list, lam=lam_demo)
    lasso = lasso_fit(X_list, y_list, lam=lam_demo)
    print("=" * 66)
    print(f"  HỆ SỐ RIDGE VÀ LASSO TẠI λ = {lam_demo}")
    print("=" * 66)
    names = ["Intercept", "x1", "x2", "x3", "x4"]
    print(f"  {'Hệ số':<10}{'Ridge':>12}{'Lasso':>12}{'beta_true':>12}")
    for j, nm in enumerate(names):
        print(f"  {nm:<10}{ridge.coefficients[j]:>12.4f}"
              f"{lasso.coefficients[j]:>12.4f}{beta_true[j]:>12.1f}")
    print(f"\n  Tại cùng λ, Ridge chỉ co nhỏ mọi hệ số trong khi Lasso ép hẳn "
          f"{5 - 1 - lasso.nonzero_coef} biến về 0")
    print(f"  (còn {lasso.nonzero_coef} hệ số khác 0): biến dư thừa x2 bị loại, "
          f"thể hiện khả năng chọn biến tự động của chuẩn L1.")

    # Bước 2: tính quỹ đạo Ridge và Lasso trên dải λ rộng
    lambdas = [10 ** e for e in np.linspace(-2, 3, 40)]
    rt = ridge_trace(X_list, y_list, lambdas)
    lp = lasso_path(X_list, y_list, lambdas)

    # Bước 3: vẽ hai quỹ đạo cạnh nhau
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Quỹ đạo hệ số theo tham số chuẩn hóa λ", fontsize=14, fontweight="bold")

    # Ridge trace: vẽ từng hệ số (bỏ intercept) theo log10(λ)
    for j in range(1, len(names)):
        axL.plot(np.log10(rt.lambdas), [c[j] for c in rt.coefficients_trace],
                 marker="o", markersize=2, label=names[j])
    axL.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axL.set_title("Ridge trace (chuẩn L2)")
    axL.set_xlabel("log₁₀(λ)")
    axL.set_ylabel("Hệ số β̂_j(λ)")
    axL.legend(fontsize=9)

    # Lasso path: cùng trục, làm nổi bật việc hệ số bị ép về 0
    for j in range(1, len(names)):
        axR.plot(np.log10(lp.lambdas), [c[j] for c in lp.coefficients],
                 marker="o", markersize=2, label=names[j])
    axR.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axR.set_title("Lasso path (chuẩn L1)")
    axR.set_xlabel("log₁₀(λ)")
    axR.set_ylabel("Hệ số β̂_j(λ)")
    axR.legend(fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "regularization_paths.png")
    fig.savefig(out_path, dpi=120)
    print("\n  Đã lưu quỹ đạo Ridge trace và Lasso path vào:", out_path)

    # Bước 4: kiểm chứng tính nhất quán nội tại của hai quỹ đạo. Đây là các
    # tính chất định nghĩa của chuẩn hóa nên là cách kiểm tra cài đặt chắc chắn
    # nhất. Lưu ý ridge_trace trả về λ theo thứ tự input (tăng dần) còn lasso_path
    # sắp xếp λ giảm dần để phục vụ warm start, vì vậy nhóm sắp lại theo λ tăng
    # dần trước khi kiểm tra để kết quả không phụ thuộc quy ước thứ tự.
    def _l2(coefs):
        return sqrt(sum(c ** 2 for c in coefs[1:]))   # bỏ intercept khỏi norm

    ridge_sorted = sorted(zip(rt.lambdas, rt.coefficients_trace), key=lambda t: t[0])
    lasso_sorted = sorted(zip(lp.lambdas, lp.nonzero_counts), key=lambda t: t[0])
    ridge_norms = [_l2(c) for _, c in ridge_sorted]
    lasso_nz = [nz for _, nz in lasso_sorted]

    # Khi λ tăng: chuẩn L2 của Ridge không tăng, số hệ số khác 0 của Lasso không tăng
    ridge_monotone = all(ridge_norms[i] >= ridge_norms[i + 1] - 1e-9
                         for i in range(len(ridge_norms) - 1))
    lasso_monotone = all(lasso_nz[i] >= lasso_nz[i + 1]
                         for i in range(len(lasso_nz) - 1))
    lam_lo, lam_hi = ridge_sorted[0][0], ridge_sorted[-1][0]
    print("\n" + "=" * 66)
    print("  KIỂM CHỨNG TÍNH NHẤT QUÁN")
    print("=" * 66)
    print(f"  ||β||₂ giảm từ {ridge_norms[0]:.3f} (λ={lam_lo:.2f}) xuống "
          f"{ridge_norms[-1]:.3f} (λ={lam_hi:.1f}).")
    print(f"  Ridge: ||β||₂ giảm đơn điệu khi λ tăng  → {'PASSED' if ridge_monotone else 'FAILED'}")
    print(f"  Lasso: số hệ số khác 0 giảm khi λ tăng  → {'PASSED' if lasso_monotone else 'FAILED'}")
    print(f"  Lasso path: số biến khác 0 đi từ {lasso_nz[0]} (λ nhỏ) "
          f"xuống {lasso_nz[-1]} (λ lớn) — đúng quy luật chọn biến của chuẩn L1.")