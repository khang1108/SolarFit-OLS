"""
File minh họa Định lý Gauss-Markov bằng mô phỏng Monte Carlo so sánh OLS với ước lượng thay thế.

Xây dựng một ước lượng tuyến tính không Bias thay thế
β̃_Alt = β̂_OLS + Ay và chứng minh qua mô phỏng rằng Var(β̃_Alt) >= Var(β̂_OLS),
tức là OLS có phương sai nhỏ nhất trong Class ước lượng đó.
Ma trận nhiễu A được chọn thỏa mãn AX = 0 để đảm bảo β̃_Alt không Bias: 
    vì E[β̃_Alt] = E[β̂_OLS] + A·E[y] = β + AXβ = β khi AX = 0. 
    
Cụ thể, A = B(I - H) với B bất kỳ tự động thỏa mãn AX = B(I-H)X = 0 vì (I-H)X = 0 theo định nghĩa của hat matrix. 
"""

import numpy as np
try:
    from .ols_implementation import _matmul, _matvec, hat_matrix, ols_fit
except ImportError:  # Cho phép chạy trực tiếp file
    from ols_implementation import _matmul, _matvec, hat_matrix, ols_fit

def make_perturbation(X, B):
    """Tạo ma trận nhiễu A = B(I - H) thỏa mãn ràng buộc không Bias AX = 0.

    Để ước lượng thay thế β̃_Alt = β̂_OLS + Ay không Bias, cần A thỏa mãn
    AX = 0. Hàm này khai thác tính chất (I - H)X = 0 của ma trận chiếu phần bù:
    vì HX = X (X chiếu lên chính mình qua C(X)), nên (I-H)X = X - HX = 0.

    Args:
        X:  Ma trận thiết kế dạng list 2D, kích thước n×k.
        B:  Ma trận nhiễu ngẫu nhiên dạng list 2D, kích thước k×n. Thường được
            tạo bằng cách nhân một ma trận Gaussian ngẫu nhiên với alt_scale nhỏ
            để ước lượng thay thế không quá lệch so với OLS.

    Returns:
        Ma trận nhiễu A = B(I - H) kích thước k×n thỏa mãn AX = 0.
    """
    H = hat_matrix(X).H
    n = len(X)
    ImH = [[(1.0 if i == j else 0.0) - H[i][j] for j in range(n)] for i in range(n)]
    return _matmul(B, ImH)


def alt_fit(X, y, A):
    """Tính ước lượng tuyến tính không Bias thay thế β̃_Alt = β̂_OLS + Ay.

    Tính không Bias của β̃_Alt được đảm bảo vì E[β̃_Alt | X] = E[β̂_OLS | X] + A·E[y | X] =
    β + AXβ = β + 0 = β. 

    Args:
        X: Ma trận thiết kế dạng list 2D, kích thước n×k.
        y: Vector quan sát dạng list, kích thước (n,).
        A: Ma trận nhiễu dạng list 2D, kích thước k×n, phải thỏa mãn AX = 0.

    Returns:
        Vector hệ số β̃_Alt kích thước (k,) dạng list.
    """
    beta_ols = ols_fit(X, y).beta_hat
    Ay = _matvec(A, y)
    return [b + a for b, a in zip(beta_ols, Ay)]


def run_monte_carlo(n=30, beta_true=(2.0, 1.0, -0.5), sigma=1.0,
                    n_rep=8000, seed=2024, alt_scale=0.06):
    """Chạy mô phỏng Monte Carlo so sánh phương sai OLS với ước lượng thay thế.

    Hàm này là bằng chứng số trung tâm cho tính BLUE của OLS. Trong mỗi trong
    n_rep lần lặp, cùng một ma trận X cố định (thiết kế theo điều kiện của định lý)
    nhận một vector nhiễu mới ε ~ N(0, σ²I), tạo y = Xβ + ε, rồi tính cả β̂_OLS
    lẫn β̃_Alt = β̂_OLS + Ay. 
    
    Sau ``n_rep`` lần, Var(β̂_OLS) và Var(β̃_Alt) được ước
    lượng bằng phương sai mẫu, dự kiến Var(β̃_Alt) > Var(β̂_OLS) với mọi A ≠ 0.
    Ma trận nhiễu A được tạo một lần từ make_perturbation(X, B) và giữ cố định
    trong toàn bộ mô phỏng, đảm bảo so sánh công bằng giữa hai ước lượng. 

    Args:
        n:          Số quan sát trong mỗi lần lặp, mặc định là 30.
        beta_true:  Vector tham số thực dạng tuple (hệ số tự do, slope1, slope2),
                    mặc định là (2.0, 1.0, -0.5).
        sigma:      Độ lệch chuẩn của nhiễu, mặc định là 1.0.
        n_rep:      Số lần lặp Monte Carlo, mặc định là 8000. Giá trị lớn hơn cho
                    ước lượng phương sai chính xác hơn nhưng tốn thời gian hơn.
        seed:       Hạt giống của numpy.random.default_rng để đảm bảo tái lập được.
        alt_scale:  Hệ số tỷ lệ cho ma trận nhiễu B; giá trị nhỏ (0.06) đảm bảo
                    ước lượng thay thế không quá xa OLS để so sánh rõ ràng hơn.

    Returns:
        Dict chứa các trường: "X" (numpy array n×k, ma trận thiết kế cố định),
        "A" (numpy array k×n, ma trận nhiễu), "beta_true", "sigma", "n", "k",
        "n_rep", "beta_ols" (numpy array n_rep×k), "beta_alt" (numpy array n_rep×k).
    """
    rng = np.random.default_rng(seed)
    beta_true = np.asarray(beta_true, float)
    k = len(beta_true)

    # Ma trận thiết kế X được giữ cố định theo quy ước định lý (conditional on X);
    # cột đầu là 1 cho hệ số tự do, các cột sau là biến giải thích Gaussian
    X_np = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    X_list = X_np.tolist()

    # Ma trận nhiễu A = B(I-H) được tạo một lần và cố định trong toàn bộ mô phỏng
    # để hai ước lượng được so sánh trên cùng điều kiện nhiễu hệ thống
    B = (alt_scale * rng.normal(size=(k, n))).tolist()
    A = make_perturbation(X_list, B)
    A_np = np.asarray(A)

    # Phân bổ bộ nhớ trước để tránh append làm chậm vòng lặp lớn
    beta_ols = np.empty((n_rep, k))
    beta_alt = np.empty((n_rep, k))

    # Tính trước phần tín hiệu cố định μ = Xβ để tránh tính lại n_rep lần
    mu = X_np @ beta_true

    # Vòng lặp Monte Carlo: mỗi lần lặp chỉ thay đổi nhiễu ε, X và β không đổi
    for r in range(n_rep):
        eps = rng.normal(scale=sigma, size=n)
        y_vec = (mu + eps).tolist()
        beta_ols[r] = ols_fit(X_list, y_vec).beta_hat
        beta_alt[r] = alt_fit(X_list, y_vec, A)

    return {
        "X": X_np,
        "A": A_np,
        "beta_true": beta_true,
        "sigma": sigma,
        "n": n,
        "k": k,
        "n_rep": n_rep,
        "beta_ols": beta_ols,
        "beta_alt": beta_alt,
    }


def verify_with_numpy_sklearn(X_np, y_one):
    """Kiểm chứng triển khai OLS thuần Python so với NumPy và scikit-learn.

    Hàm này thực hiện kiểm tra tính đúng đắn số học của ols_fit bằng cách so sánh
    hệ số β̂ với kết quả từ hai thư viện chuẩn. Sai số so với numpy.linalg.lstsq
    và sklearn.LinearRegression đều kỳ vọng dưới 1e-8, xác nhận triển khai thuần
    Python không có lỗi số học đáng kể. 

    Args:
        X_np: Ma trận thiết kế dạng numpy array, kích thước n×k, bao gồm cột hệ số tự do.
        y_one: Vector quan sát dạng list hoặc numpy array, kích thước (n,).

    Returns:
        Dict với hai khóa: "error_numpy" là max|β̂_scratch - β̂_numpy|_∞ và
        "error_sklearn" là max|β̂_scratch - β̂_sklearn|_∞.
    """
    from sklearn.linear_model import LinearRegression

    beta_scratch = np.asarray(ols_fit(X_np.tolist(), y_one).beta_hat)
    beta_numpy, *_ = np.linalg.lstsq(X_np, y_one, rcond=None)
    beta_sklearn = LinearRegression(fit_intercept=False).fit(X_np, y_one).coef_

    return {
        "error_numpy": np.max(np.abs(beta_scratch - beta_numpy)),
        "error_sklearn": np.max(np.abs(beta_scratch - beta_sklearn)),
    }
