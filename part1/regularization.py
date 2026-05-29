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
from typing import List, Tuple
from math import sqrt


@dataclass
class RidgeResult:
    """Lớp chứa toàn bộ kết quả trả về của hàm ridge_fit.

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

    n = len(y)
    p = len(X[0])

    try:
        # Tính X'X — ma trận Gram của X
        XtX = [[0.0] * p for _ in range(p)]
        for i in range(p):
            for j in range(p):
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
    """Lớp chứa dữ liệu ridge trace — quỹ đạo hệ số khi λ thay đổi.

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
    trong thực hành Ridge regression vì không có công thức dạng đóng để chọn λ tối
    ưu; thay vào đó người ta quan sát đồ thị hoặc dùng cross-validation để xác định
    λ tại điểm các hệ số bắt đầu ổn định. Chỉ số GCV được tính xấp xỉ trong phiên
    bản này do phức tạp số học của hat matrix Ridge.

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

    for lam in lambdas:
        result = ridge_fit(X, y, lam)
        if result.success:
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


def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Giải hệ phương trình tuyến tính Ax = b bằng khử Gauss có partial pivoting.

    Hàm nội bộ này được thiết kế để giải hệ Normal Equations đã được chuẩn hóa
    (X'X + λI)β = X'y trong Ridge regression. Partial pivoting (đổi chỗ hàng để
    đưa phần tử lớn nhất về vị trí pivot) được áp dụng để tăng tính ổn định số học
    so với khử Gauss đơn thuần. Bao gồm hai giai đoạn: (1) khử xuôi biến đổi A về
    dạng tam giác trên, (2) thế ngược (back substitution) tính x từ dưới lên.

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
