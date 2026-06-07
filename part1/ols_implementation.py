"""
File triển khai thuật toán Ordinary Least Squares (OLS) thuần Python.

Triển khai hai hàm chính để ước lượng mô hình hồi quy tuyến tính
theo phương pháp OLS: hàm ``ols_fit`` tính vector hệ số ``beta_hat`` 
thông qua Normal Equations, và hàm hat_matrix tính ma trận chiếu H = X(X'X)^{-1}X'. 
"""

import copy
import sys, dataclasses
import numpy as np

from dataclasses import dataclass
from math import sqrt
from time import perf_counter
from typing import List, Optional, Tuple


# Ngưỡng pivot dùng trong khử Gauss để phát hiện ma trận gần suy biến:
# nếu phần tử pivot nhỏ hơn EPS_PIVOT thì coi như bằng 0 và báo lỗi.
EPS_PIVOT        = 1e-12
# Ngưỡng dung sai Frobenius để kiểm tra tính idempotent của hat matrix:
# ||H^2 - H||_F <= DEFAULT_TOL_IDEM mới coi H là idempotent hợp lệ.
DEFAULT_TOL_IDEM = 1e-10

@dataclass
class OLSResult:
    """Dataclass chứa toàn bộ kết quả trả về của hàm ols_fit.

    Attributes:
        method: Tên phương pháp.
        beta_hat: Vector hệ số ước lượng β̂ ∈ R^{p+1}.
        sigma2_hat: Ước lượng phương sai nhiễu σ² = RSS / (n-p-1).
        y_hat: Giá trị dự báo ŷ = Xβ̂ ∈ R^n.
        residuals: Vector phần dư e = y - ŷ ∈ R^n.
        rss: Tổng bình phương phần dư RSS = ||e||².
        dof: Bậc tự do phần dư = n - p - 1.
        success: True nếu X'X khả nghịch và tính toán thành công, False nếu có lỗi (ví dụ: multicollinearity).
        runtime_sec: Thời gian chạy của hàm (tính bằng giây).
        message: Thông báo kết quả hoặc lỗi chi tiết
    """
    method: str              
    beta_hat: List[float]     
    sigma2_hat: float         
    y_hat: List[float]        
    residuals: List[float]    
    rss: float                
    dof: int                  
    success: bool             
    runtime_sec: float        
    message: str              


@dataclass
class HatMatrixResult:
    """Dataclass chứa toàn bộ kết quả trả về của hàm hat_matrix.

    Attributes:
        method: Tên phương pháp, ở đây là "HatMatrix"
        H: Ma trận chiếu H ∈ R^{n×n}.
        sym_err: ||H - H'||_∞, đo độ lệch đối xứng
        idem_err: ||H² - H||_F, đo độ lệch idempotent
        rank_H: Rank của H, kỳ vọng bằng p+1
        trace_H: Trace của H, kỳ vọng bằng p+1
        success: True nếu idem_err <= tol
        runtime_sec: Thời gian chạy (giây)
        message: Thông báo kết quả hoặc cảnh báo
    """
    method: str                  
    H: List[List[float]]         
    sym_err: float               
    idem_err: float              
    rank_H: int                  
    trace_H: float               
    success: bool                
    runtime_sec: float           
    message: str                 


def _validate_inputs(
    X: List[List[float]], y: Optional[List[float]] = None
) -> Tuple[List[List[float]], Optional[List[float]]]:
    """Kiểm tra tính hợp lệ của ma trận X và vector y, trả về phiên bản chỉnh sửa.

    Hàm này đảm bảo các tiền điều kiện cần thiết trước khi thực hiện đại số tuyến tính:
    X phải là danh sách 2 chiều, tất cả hàng phải cùng độ dài, số hàng n phải không
    nhỏ hơn số cột k (hệ không được thiếu định), và nếu y được cung cấp thì phải có
    đúng n phần tử. Việc tạo bản sao float riêng đảm bảo hàm không làm thay đổi dữ
    liệu đầu vào của người gọi (no side-effects).

    Args:
        X:  Ma trận thiết kế dạng list 2D, kỳ vọng có cột đầu tiên là vector 1
            tương ứng với hệ số tự do, kích thước n×(p+1).
        y:  Vector quan sát tùy chọn, kích thước (n,). Nếu None thì chỉ kiểm tra X.

    Returns:
        (X_copy, y_copy) với các phần tử đã được ép kiểu float. Nếu y là None
        thì phần tử thứ hai của bộ trả về cũng là None.

    Raises:
        ValueError: Khi X không phải list 2D, các hàng không đều nhau, n < k,
                    hoặc len(y) != n.
    """
    if not isinstance(X, list) or not X or not isinstance(X[0], list):
        raise ValueError("X must be a 2-D list of lists.")
    n, k = len(X), len(X[0])
    if any(len(row) != k for row in X):
        raise ValueError("All rows of X must have the same length.")
    if n < k:
        raise ValueError(f"n={n} must be >= k={k} (underdetermined system).")
    X_copy = [[float(v) for v in row] for row in X]
    if y is None:
        return X_copy, None
    if len(y) != n:
        raise ValueError(f"len(y)={len(y)} must equal n={n}.")
    return X_copy, [float(v) for v in y]


def _transpose(A: List[List[float]]) -> List[List[float]]:
    """Tính chuyển vị của ma trận A, trả về A' có kích thước n×m từ A kích thước m×n."""
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Nhân ma trận A (m×k) với ma trận B (k×n), trả về C = AB có kích thước m×n.

    Phép nhân ma trận này được dùng chủ yếu để tính G = X'X và H = X(X'X)^{-1}X'
    trong quá trình giải Normal Equations và xây dựng hat matrix.
    """
    m, k, n = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(n)]
            for i in range(m)]


def _matvec(A: List[List[float]], x: List[float] | None) -> List[float]:
    """Nhân ma trận A với vector cột x, trả về vector Ax.

    Hàm này dùng để tính giá trị dự báo ŷ = Xβ̂ và vector X'y khi xây dựng
    Normal Equations.
    """
    if x is None:
        return [0.0] * len(A)
    
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _vecsub(a: List[float] | None, b: List[float]) -> List[float]:
    """Trừ vector b khỏi vector a, trả về vector hiệu a - b.

    Dùng để tính vector phần dư e = y - ŷ sau khi có ước lượng β̂.
    """
    if a is None:
        return [-b[i] for i in range(len(b))]
    
    return [a[i] - b[i] for i in range(len(a))]


def _dot(x: List[float], y: List[float]) -> float:
    """Tính tích vô hướng (dot product) của hai vector x và y.

    Kết quả x'y được dùng để tính RSS = e'e = ||e||², là hàm mục tiêu mà OLS
    tối thiểu hóa.
    """
    return sum(xi * yi for xi, yi in zip(x, y))


def _mat_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Trừ ma trận B khỏi ma trận A, trả về ma trận hiệu A - B.

    Dùng trong kiểm tra tính đối xứng (H - H') và idempotent (H² - H) của hat matrix.
    """
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _norm_fro(A: List[List[float]]) -> float:
    """Tính chuẩn Frobenius của ma trận A, bằng căn tổng bình phương tất cả phần tử.

    Chuẩn Frobenius ||A||_F được dùng để đo độ lệch idempotent ||H² - H||_F,
    trong đó giá trị càng gần 0 thì H càng thỏa mãn tính chất chiếu.
    """
    return sqrt(sum(A[i][j] ** 2 for i in range(len(A)) for j in range(len(A[0]))))


def _norm_inf_mat(A: List[List[float]]) -> float:
    """Tính chuẩn vô cực (infinity norm) của ma trận A, bằng giá trị tuyệt đối lớn nhất.

    Chuẩn ||A||_∞ = max|a_{ij}| được dùng để đo độ lệch đối xứng ||H - H'||_∞,
    cho biết phần tử nào lệch khỏi đối xứng nhiều nhất.
    """
    return max(abs(A[i][j]) for i in range(len(A)) for j in range(len(A[0])))


def _trace(A: List[List[float]]) -> float:
    """Tính trace (tổng các phần tử trên đường chéo chính) của ma trận vuông A.

    Với hat matrix H, trace(H) = rank(H) = p+1 vì các trị riêng của một phép
    chiếu chỉ có thể là 0 hoặc 1, nên tổng trị riêng bằng số chiều không gian
    con mà H chiếu lên, tức là p+1.
    """
    return sum(A[i][i] for i in range(len(A)))


def _mat_inv(A: List[List[float]]) -> List[List[float]]:
    """Tính nghịch đảo ma trận vuông A bằng phương pháp Gauss-Jordan trên ma trận mở rộng [A | I].

    Phương pháp khử Gauss-Jordan biến đổi ma trận mở rộng [A | I] thành [I | A^{-1}]
    thông qua các phép biến đổi hàng cơ bản, áp dụng partial pivoting để tăng tính
    ổn định số học. Hàm này được gọi trong ols_fit để tính (X'X)^{-1} và trong
    hat_matrix để tính H = X(X'X)^{-1}X'. Nếu phần tử pivot nhỏ hơn EPS_PIVOT,
    ma trận được coi là suy biến và hàm raise ValueError.

    Args:
        A: Ma trận vuông kích thước n×n cần tính nghịch đảo.

    Returns:
        Ma trận nghịch đảo A^{-1} kích thước n×n.

    Raises:
        ValueError: Khi ma trận suy biến (phần tử pivot bằng 0 trong quá trình khử).
    """
    n = len(A)
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < EPS_PIVOT:
            raise ValueError(f"Singular matrix: zero pivot at column {col}.")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot_val = aug[col][col]
        aug[col] = [v / pivot_val for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            f = aug[row][col]
            aug[row] = [aug[row][j] - f * aug[col][j] for j in range(2 * n)]
    return [row[n:] for row in aug]


def _mat_rank(A: List[List[float]], tol: float = EPS_PIVOT) -> int:
    """Tính hạng (rank) của ma trận A bằng phương pháp khử Gauss.

    Rank của ma trận X (hay X'X) là điều kiện đủ để Normal Equations có nghiệm
    duy nhất: hệ y = Xβ + ε có nghiệm OLS duy nhất khi và chỉ khi rank(X) = p+1,
    tức là X có full column rank. Hàm đếm số hàng pivot khác không sau khi khử
    Gauss, dùng tol để phân biệt pivot thực sự với giá trị gần không do sai số
    số học.

    Args:
        A: Ma trận bất kỳ kích thước m×n.
        tol: Ngưỡng để coi phần tử là bằng không, mặc định là EPS_PIVOT = 1e-12.

    Returns:
        Hạng của ma trận A dưới dạng số nguyên.
    """
    m, n = len(A), len(A[0])
    mat = copy.deepcopy(A)
    rank, used_row = 0, 0
    for col in range(n):
        pivot_row = next((r for r in range(used_row, m) if abs(mat[r][col]) > tol), None)
        if pivot_row is None:
            continue
        mat[used_row], mat[pivot_row] = mat[pivot_row], mat[used_row]
        for r in range(used_row + 1, m):
            if abs(mat[used_row][col]) < tol:
                break
            f = mat[r][col] / mat[used_row][col]
            mat[r] = [mat[r][j] - f * mat[used_row][j] for j in range(n)]
        rank += 1
        used_row += 1
    return rank


# --- Hàm ước lượng cốt lõi ---

def ols_fit(X: List[List[float]], y: List[float]) -> OLSResult:
    """Tính ước lượng OLS bằng Normal Equations, trả về hệ số và các thống kê liên quan.

    Hàm này giải bài toán tối thiểu hóa RSS = ||y - Xβ||² bằng nghiệm dạng đóng
    β̂ = (X'X)^{-1}X'y, gọi là Normal Equations. Đây là ước lượng BLUE (Best Linear
    Unbiased Estimator) theo Định lý Gauss-Markov khi các giả thiết GM1-GM4 thỏa
    mãn, tức là khi phần dư có kỳ vọng bằng 0 và phương sai đồng nhất σ²I. Phương
    sai nhiễu được ước lượng không chệch bởi σ̂² = RSS/(n-p-1), trong đó n-p-1 là
    bậc tự do phần dư. Toàn bộ tính toán được thực hiện bằng Python thuần để minh
    họa các bước đại số một cách tường minh.

    Thuật toán gồm 5 bước:
    (1) Tính G = X'X và q = X'y,
    (2) Kiểm tra rank(G) == k để đảm bảo G khả nghịch,
    (3) Tính β̂ = G^{-1}q bằng Gauss-Jordan,
    (4) Tính ŷ = Xβ̂, e = y - ŷ, RSS = e'e,
    (5) Tính σ̂² = RSS/(n-p-1).

    Args:
        X:  Ma trận thiết kế kích thước n×(p+1) dạng list 2D, với cột đầu tiên
            là vector 1 tương ứng với hệ số tự do.
        y:  Vector quan sát kích thước (n,).

    Returns:
        OLSResult chứa β̂, σ̂², ŷ, vector phần dư, RSS, bậc tự do, trạng thái
        thành công và thông báo kết quả.
    """
    start = perf_counter()
    X_mat, y_vec = _validate_inputs(X, y)
    n, k = len(X_mat), len(X_mat[0])
    p = k - 1

    try:
        # Bước 1: xây dựng hệ Normal Equations G = X'X và q = X'y
        Xt = _transpose(X_mat)
        G  = _matmul(Xt, X_mat)
        q  = _matvec(Xt, y_vec)

        # Bước 2: kiểm tra khả nghịch — nếu rank(X'X) < k thì có multicollinearity
        # và Normal Equations không có nghiệm duy nhất
        if _mat_rank(G) < k:
            return OLSResult(
                method="OLS-NormalEquations", beta_hat=[], sigma2_hat=float("nan"),
                y_hat=[], residuals=[], rss=float("nan"), dof=n - p - 1,
                success=False, runtime_sec=perf_counter() - start,
                message="X^T X is not invertible (multicollinearity).",
            )

        # Bước 3: nghiệm dạng đóng β̂ = (X'X)^{-1}X'y
        beta_hat = _matvec(_mat_inv(G), q)

        # Bước 4: tính giá trị dự báo và phần dư
        y_hat     = _matvec(X_mat, beta_hat)
        residuals = _vecsub(y_vec, y_hat)
        rss       = _dot(residuals, residuals)

        # Bước 5: ước lượng không chệch phương sai nhiễu; chia cho (n-p-1) chứ
        # không phải n vì phần dư bị ràng buộc bởi p+1 phương trình pháp tuyến
        dof        = n - p - 1
        sigma2_hat = rss / dof

        return OLSResult(
            method="OLS-NormalEquations", beta_hat=beta_hat, sigma2_hat=sigma2_hat,
            y_hat=y_hat, residuals=residuals, rss=rss, dof=dof, success=True,
            runtime_sec=perf_counter() - start,
            message=f"Solved. RSS={rss:.6g}, sigma2={sigma2_hat:.6g}, dof={dof}.",
        )

    except Exception as exc:
        return OLSResult(
            method="OLS-NormalEquations", beta_hat=[], sigma2_hat=float("nan"),
            y_hat=[], residuals=[], rss=float("nan"), dof=n - p - 1,
            success=False, runtime_sec=perf_counter() - start,
            message=f"ols_fit failed: {exc}",
        )


def hat_matrix(X: List[List[float]], tol: float = DEFAULT_TOL_IDEM) -> HatMatrixResult:
    """Tính hat matrix H = X(X'X)^{-1}X' và kiểm chứng các tính chất đại số của nó.

    Hat matrix H là phép chiếu trực giao lên không gian cột C(X)
    ŷ = Hy. Ma trận phần dư (I - H) chiếu y lên phần bù trực giao của C(X), 
    nên e = (I-H)y. 
    
    Do H là phép chiếu trực giao, nó phải
    thỏa mãn hai tính chất đại số: đối xứng H = H' (vì C(X) là không gian con
    đóng trong R^n) và idempotent H² = H (vì chiếu hai lần bằng chiếu một lần).
    Ngoài ra rank(H) = trace(H) = p+1 do các trị riêng của H chỉ là 0 hoặc 1.
    Hàm này kiểm chứng tất cả các tính chất trên theo chuẩn số và báo cáo sai số.

    Thuật toán:
    (1) Tính G_inv = (X'X)^{-1} bằng Gauss-Jordan,
    (2) Tính H = X G_inv X',
    (3) Kiểm tra sym_err = ||H - H'||_∞, idem_err = ||H² - H||_F, rank và trace.

    Args:
        X: Ma trận thiết kế kích thước n×(p+1) dạng list 2D.
        tol: Ngưỡng dung sai Frobenius để coi H là idempotent hợp lệ, mặc định
            là DEFAULT_TOL_IDEM = 1e-10.

    Returns:
        HatMatrixResult chứa ma trận H, các chỉ số sai số, rank, trace và
        trạng thái thành công.
    """
    start = perf_counter()
    X_mat, _ = _validate_inputs(X)
    n, k = len(X_mat), len(X_mat[0])

    try:
        # Bước 1: tính nghịch đảo (X'X)^{-1} — đây là ma trận phương sai-hiệp phương
        # sai của β̂ nhân với σ², cần thiết để xây dựng H
        Xt    = _transpose(X_mat)
        G_inv = _mat_inv(_matmul(Xt, X_mat))

        # Bước 2: H = X(X'X)^{-1}X' — phép chiếu trực giao lên C(X)
        H = _matmul(_matmul(X_mat, G_inv), Xt)

        # Bước 3: kiểm chứng số học các tính chất lý thuyết của phép chiếu trực giao
        sym_err  = _norm_inf_mat(_mat_sub(H, _transpose(H)))
        idem_err = _norm_fro(_mat_sub(_matmul(H, H), H))
        rank_H   = _mat_rank(H)
        trace_H  = _trace(H)

        ok = idem_err <= tol
        return HatMatrixResult(
            method="HatMatrix", H=H, sym_err=sym_err, idem_err=idem_err,
            rank_H=rank_H, trace_H=trace_H, success=ok,
            runtime_sec=perf_counter() - start,
            message=(
                f"{'[OK]' if ok else '[WARN]'} ||H^2-H||_F={idem_err:.2e}, "
                f"||H-H^T||_inf={sym_err:.2e}, "
                f"rank={rank_H} (exp {k}), tr={trace_H:.4f} (exp {k})."
            ),
        )

    except Exception as exc:
        return HatMatrixResult(
            method="HatMatrix", H=[[]], sym_err=float("nan"), idem_err=float("nan"),
            rank_H=-1, trace_H=float("nan"), success=False,
            runtime_sec=perf_counter() - start,
            message=f"hat_matrix failed: {exc}",
        )


def run_ols_analysis(
    X: List[List[float]], y: List[float], tol_idem: float = DEFAULT_TOL_IDEM
) -> dict:
    """Chạy ``ols_fit`` và ``hat_matrix``, trả về dict chứa cả hai kết quả.

    Hàm tiện ích này gom hai phép tính cốt lõi của OLS vào một lần gọi duy nhất,
    phù hợp cho các script phân tích cần cả ước lượng hệ số lẫn kiểm tra tính
    chất của hat matrix trong cùng một pipeline.

    Args:
        X:          Ma trận thiết kế kích thước n×(p+1).
        y:          Vector quan sát kích thước (n,).
        tol_idem:   Ngưỡng dung sai Frobenius cho kiểm tra idempotent,
                    mặc định là DEFAULT_TOL_IDEM = 1e-10.

    Returns:
        Dict với hai khóa: "ols" chứa OLSResult và "hat" chứa HatMatrixResult.
    """
    return {"ols": ols_fit(X, y), "hat": hat_matrix(X, tol=tol_idem)}
