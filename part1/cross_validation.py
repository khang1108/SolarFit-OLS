"""
File thực hiện ``cross-validation`` để đánh giá khả năng tổng quát hóa của mô hình.

Nó này cung cấp ba hàm cho quy trình cross-validation: kfold_cv thực hiện
k-fold CV cho OLS, kfold_cv_ridge cho Ridge regression với tham số λ cụ thể, và
model_selection_cv so sánh nhiều mô hình để chọn mô hình tốt nhất. 
"""

from dataclasses import dataclass
from typing import List, Callable, Dict
from math import sqrt
import random

@dataclass
class CVResult:
    """Dataclass chứa kết quả k-fold cross-validation cho một mô hình."""

    k: int  # Số fold được dùng trong CV
    model_name: str  # Tên mô hình (ví dụ: "OLS", "Ridge(lambda=0.1)")
    cv_scores: List[float]  # Điểm số trên tập test của từng fold
    mean_cv_score: float  # Trung bình điểm CV qua k fold
    std_cv_score: float  # Độ lệch chuẩn điểm CV, đo tính ổn định
    train_scores: List[float]  # Điểm số trên tập huấn luyện của từng fold
    test_scores: List[float]  # Điểm số trên tập test của từng fold (bằng cv_scores)


def kfold_cv(
    X: List[List[float]], y: List[float], k: int = 5, metric: str = "mse"
) -> CVResult:
    """Thực hiện k-fold cross-validation cho mô hình OLS, đánh giá khả năng tổng quát hóa.

    Args:
        X: Ma trận thiết kế kích thước n×(p+1) dạng list 2D.
        y: Vector quan sát kích thước (n,).
        k: Số fold, phải trong khoảng [2, n], mặc định là 5.
        metric: Chỉ số đánh giá, một trong "mse", "rmse", "mae", "r2".

    Returns:
        CVResult chứa điểm CV cho từng fold, trung bình và độ lệch chuẩn.

    Raises:
        ValueError: Khi k < 2 hoặc k > n.
    """
    n = len(y)
    if k > n or k < 2:
        raise ValueError(f"k must be between 2 and {n}")

    # Split data into k folds
    fold_indices = _stratified_k_fold(n, k)
    cv_scores = []
    train_scores = []
    test_scores = []

    for fold_idx, test_idx in enumerate(fold_indices):
        # Phân tách tập train và test theo chỉ số fold hiện tại
        X_train, y_train = [], []
        X_test, y_test = [], []

        test_set = set(test_idx)
        for i in range(n):
            if i in test_set:
                X_test.append(X[i])
                y_test.append(y[i])
            else:
                X_train.append(X[i])
                y_train.append(y[i])

        # Khớp OLS chỉ trên tập train — tập test phải hoàn toàn "unseen"
        try:
            from ols_implementation import ols_fit

            ols_result = ols_fit(X_train, y_train)
            if not ols_result.success:
                cv_scores.append(float("nan"))
                continue

            # Tính dự báo trên tập test bằng hệ số vừa học từ tập train;
            # ols_result.y_hat là predictions trên TRAIN nên không dùng trực tiếp được
            beta = ols_result.beta_hat
            y_pred_test: List[float] = [
                sum((X_test[i][j] * beta[j] for j in range(len(beta))), 0.0)
                for i in range(len(X_test))
            ]
            score_test = _calculate_metric(y_test, y_pred_test, metric)
            test_scores.append(score_test)

            # Đánh giá trên tập train để quan sát khoảng cách train-test (bias-variance gap)
            score_train = _calculate_metric(y_train, ols_result.y_hat, metric)
            train_scores.append(score_train)

            cv_scores.append(score_test)

        except Exception as e:
            print(f"Fold {fold_idx}: Error - {e}")
            cv_scores.append(float("nan"))

    # Tổng hợp thống kê qua k fold; loại bỏ NaN trước khi tính trung bình
    valid_scores = [s for s in cv_scores if s == s]  # s == s là False khi s là NaN
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")
    if len(valid_scores) > 1:
        var = sum((s - mean_score) ** 2 for s in valid_scores) / (len(valid_scores) - 1)
        std_score = sqrt(var)
    else:
        std_score = float("nan")

    return CVResult(
        k=k,
        model_name="OLS",
        cv_scores=cv_scores,
        mean_cv_score=mean_score,
        std_cv_score=std_score,
        train_scores=train_scores,
        test_scores=test_scores,
    )


def kfold_cv_ridge(
    X: List[List[float]], y: List[float], lam: float, k: int = 5, metric: str = "mse"
) -> CVResult:
    """Thực hiện k-fold cross-validation cho Ridge regression với tham số λ cho trước.

    Args:
        X: Ma trận thiết kế kích thước n×(p+1) dạng list 2D.
        y: Vector quan sát kích thước (n,).
        lam: Tham số chuẩn hóa λ của Ridge, phải >= 0.
        k: Số fold, phải trong khoảng [2, n], mặc định là 5.
        metric: Chỉ số đánh giá, một trong "mse", "rmse", "mae", "r2".

    Returns:
        CVResult chứa điểm CV cho từng fold và thống kê tổng hợp.

    Raises:
        ValueError: Khi k < 2 hoặc k > n.
    """
    n = len(y)
    if k > n or k < 2:
        raise ValueError(f"k must be between 2 and {n}")

    fold_indices = _stratified_k_fold(n, k)
    cv_scores = []

    for fold_idx, test_idx in enumerate(fold_indices):
        X_train, y_train = [], []
        X_test, y_test = [], []

        test_set = set(test_idx)
        for i in range(n):
            if i in test_set:
                X_test.append(X[i])
                y_test.append(y[i])
            else:
                X_train.append(X[i])
                y_train.append(y[i])

        try:
            from regularization import ridge_fit

            ridge_result = ridge_fit(X_train, y_train, lam)
            if not ridge_result.success:
                cv_scores.append(float("nan"))
                continue

            # Tính dự báo trên tập test từ hệ số Ridge vừa học được trên tập train
            y_pred_test: List[float] = [
                sum(
                    (X_test[i][j] * ridge_result.coefficients[j]
                    for j in range(len(ridge_result.coefficients))),
                    0.0,
                )
                for i in range(len(X_test))
            ]

            score = _calculate_metric(y_test, y_pred_test, metric)
            cv_scores.append(score)

        except Exception:
            cv_scores.append(float("nan"))

    valid_scores = [s for s in cv_scores if s == s]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")
    var = (
        sum((s - mean_score) ** 2 for s in valid_scores) / (len(valid_scores) - 1)
        if len(valid_scores) > 1
        else float("nan")
    )
    std_score = sqrt(var) if var == var else float("nan")

    return CVResult(
        k=k,
        model_name=f"Ridge(lambda={lam})",
        cv_scores=cv_scores,
        mean_cv_score=mean_score,
        std_cv_score=std_score,
        train_scores=[],
        test_scores=[],
    )


def _stratified_k_fold(n: int, k: int) -> List[List[int]]:
    """Tạo danh sách chỉ số cho k fold bằng cách chia ngẫu nhiên n quan sát.

    Args:
        n: Tổng số quan sát.
        k: Số fold cần chia.

    Returns:
        Danh sách k phần, mỗi phần là danh sách chỉ số hàng thuộc fold đó.
    """
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        folds.append(indices[start:end])

    return folds


def _calculate_metric(y_true: List[float], y_pred: List[float], metric: str) -> float:
    """Tính chỉ số đánh giá mô hình trên cặp (y_true, y_pred) cho trước.

    Hàm hỗ trợ bốn chỉ số phổ biến: MSE là hàm mục tiêu mà OLS tối thiểu hóa
    trên tập train nhưng cần đánh giá độc lập trên test; RMSE có cùng đơn vị với y
    nên dễ diễn giải hơn; MAE ít nhạy cảm với outlier hơn MSE; R² cho biết tỷ lệ
    phương sai được giải thích, có thể âm khi mô hình kém hơn đường nằm ngang ȳ.

    Args:
        y_true: Vector giá trị quan sát thực tế kích thước (n,).
        y_pred: Vector giá trị dự báo kích thước (n,).
        metric: Tên chỉ số, một trong "mse", "rmse", "mae", "r2".

    Returns:
        Giá trị chỉ số theo kiểu float; trả về NaN khi y_true và y_pred có độ dài khác nhau.

    Raises:
        ValueError: Khi metric không nằm trong danh sách hỗ trợ.
    """
    if len(y_true) != len(y_pred):
        return float("nan")

    n = len(y_true)

    if metric == "mse":
        return sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n

    elif metric == "rmse":
        mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
        return sqrt(mse)

    elif metric == "mae":
        return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

    elif metric == "r2":
        y_mean = sum(y_true) / n
        ss_tot = sum((y_true[i] - y_mean) ** 2 for i in range(n))
        ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
        if ss_tot != 0:
            return 1.0 - (ss_res / ss_tot)
        else:
            return 0.0

    else:
        raise ValueError(f"Unknown metric: {metric}")


@dataclass
class ModelComparisonResult:
    """Lớp chứa kết quả so sánh nhiều mô hình bằng k-fold cross-validation.

    Khi so sánh OLS với Ridge ở nhiều giá trị λ, dataclass này tập hợp CVResult
    của từng mô hình vào một dict và xác định mô hình tốt nhất theo mean_cv_score.
    Lưu ý rằng "tốt nhất" theo CV không nhất thiết là tốt nhất về mặt giải thích
    thống kê: Ridge với λ tối ưu có thể dự báo tốt hơn OLS nhưng hệ số Ridge bị
    chệch nên không thể dùng cho suy luận thống kê chuẩn.
    """

    models: Dict[str, CVResult]  # Dict tên mô hình -> CVResult tương ứng
    best_model: str  # Tên mô hình có mean_cv_score tốt nhất
    best_score: float  # Giá trị mean_cv_score của mô hình tốt nhất


def model_selection_cv(
    X: List[List[float]],
    y: List[float],
    k: int = 5,
    models: Dict[str, Callable] | None = None,
) -> ModelComparisonResult:
    """So sánh nhiều mô hình bằng k-fold cross-validation và xác định mô hình tốt nhất.

    Args:
        X: Ma trận thiết kế kích thước n×(p+1) dạng list 2D.
        y: Vector quan sát kích thước (n,).
        k: Số fold dùng trong CV, mặc định là 5.
        models: Dict ánh xạ tên mô hình (str) sang hàm callable nhận (X, y) và
                trả về CVResult. Nếu None thì dùng bộ mặc định OLS + Ridge.

    Returns:
        ModelComparisonResult chứa CVResult của mỗi mô hình, tên và điểm số của
        mô hình tốt nhất.
    """
    if models is None:
        models = {
            "OLS": lambda X, y: kfold_cv(X, y, k, metric="rmse"),
            "Ridge(lam=0.1)": lambda X, y: kfold_cv_ridge(X, y, 0.1, k, metric="rmse"),
            "Ridge(lam=1.0)": lambda X, y: kfold_cv_ridge(X, y, 1.0, k, metric="rmse"),
        }

    results = {}
    for name, model_func in models.items():
        try:
            results[name] = model_func(X, y)
        except Exception as e:
            print(f"Error with {name}: {e}")

    # Chọn mô hình có mean_cv_score nhỏ nhất (chỉ số lỗi như MSE/RMSE/MAE là tốt hơn khi nhỏ)
    best_model = min(results.keys(), key=lambda m: results[m].mean_cv_score)
    best_score = results[best_model].mean_cv_score

    return ModelComparisonResult(
        models=results,
        best_model=best_model,
        best_score=best_score,
    )


if __name__ == "__main__":
    import sys
    import numpy as np

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

    np.random.seed(2024)
    n_obs = 100
    x1 = np.random.randn(n_obs)
    x2 = 0.9 * x1 + 0.1 * np.random.randn(n_obs)   # tương quan cao với x1
    x3 = np.random.randn(n_obs)
    X_np = np.column_stack([np.ones(n_obs), x1, x2, x3])
    beta_true = np.array([1.0, 2.0, 1.5, -1.0])
    y_np = X_np @ beta_true + 1.5 * np.random.randn(n_obs)
    X_list, y_list = X_np.tolist(), y_np.tolist()

    # Bước 1: 5-fold CV cho OLS, in điểm từng fold
    cv_ols = kfold_cv(X_list, y_list, k=5, metric="mse")
    print("=" * 66)
    print("  5-FOLD CROSS-VALIDATION CHO OLS (chỉ số MSE)")
    print("=" * 66)
    for i, s in enumerate(cv_ols.cv_scores, 1):
        print(f"  Fold {i}: MSE = {s:.4f}")
    print(f"  → Trung bình = {cv_ols.mean_cv_score:.4f} ± {cv_ols.std_cv_score:.4f}")

    # Bước 2: 5-fold CV cho Ridge ở vài giá trị λ để thấy ảnh hưởng chuẩn hóa
    print("\n" + "=" * 66)
    print("  5-FOLD CV CHO RIDGE THEO λ")
    print("=" * 66)
    print(f"  {'λ':>8}   {'MSE trung bình':>16}   {'Độ lệch chuẩn':>14}")
    for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:
        cv_r = kfold_cv_ridge(X_list, y_list, lam=lam, k=5, metric="mse")
        print(f"  {lam:>8.1f}   {cv_r.mean_cv_score:>16.4f}   {cv_r.std_cv_score:>14.4f}")

    # Bước 3: so sánh đồng thời nhiều mô hình và chọn mô hình tốt nhất
    comp = model_selection_cv(X_list, y_list, k=5)
    print("\n" + "=" * 66)
    print("  SO SÁNH MÔ HÌNH (model_selection_cv, chỉ số RMSE)")
    print("=" * 66)
    for name, res in comp.models.items():
        mark = "  ← tốt nhất" if name == comp.best_model else ""
        print(f"  {name:<16} RMSE = {res.mean_cv_score:.4f} ± {res.std_cv_score:.4f}{mark}")
    print(f"\n  Mô hình tốt nhất theo CV: {comp.best_model} (RMSE = {comp.best_score:.4f}).")
    print("  Lưu ý: 'tốt nhất' theo dự báo không đồng nghĩa tốt nhất cho suy luận,")
    print("  vì hệ số Ridge bị chệch nên không dùng trực tiếp cho kiểm định t.")
