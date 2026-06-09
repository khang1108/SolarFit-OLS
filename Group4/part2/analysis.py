"""Pipeline xây dựng, đánh giá và xuất kết quả mô hình cho Phần 2.

Luồng xử lý chính:
    1. Load và Preprocessing dữ liệu bằng DataPipeline.
    2. Huấn luyện target trong không gian log1p để giảm skewness.
    3. Đánh giá các mô hình bằng cross-validation trên đơn vị TZS.
    4. Fit mô hình cuối trên dev_train và Prediction holdout/competition test.
    5. Xuất bảng so sánh, feature importance và submission.

Các mô hình:
    1. OLS
    2. OLS có chọn biến bằng p-value
    3. Ridge
    4. Lasso
    5. ElasticNet
    6. Polynomial bậc 2 + ElasticNet
    7. Kernel Ridge Regression (RBF)
    8. Ensemble trung bình
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from part2.data_pipeline import DataPipeline, PipelineConfig, PipelineResult

from part1.ols_implementation import ols_fit
from part1.regularization import _soft_threshold
from part1.statistical_distributions import student_t_two_sided_pvalue


warnings.filterwarnings("ignore", category=RuntimeWarning)


RANDOM_STATE = 42
TARGET_COL = "total_cost"
CV_FOLDS = 5
INNER_CV_FOLDS = 3  # inner loop của nested CV để chọn hyperparameter per outer fold


@dataclass
class CVResult:
    """Kết quả cross-validation của một mô hình.

    Ta lưu cả từng fold và thống kê mean/std để khi viết báo cáo có thể chỉ ra
    không chỉ mô hình tốt trung bình ra sao, mà còn ổn định tới mức nào giữa
    các phần chia dữ liệu khác nhau.
    """

    model_name: str
    fold_mae: List[float]
    fold_rmse: List[float]
    fold_r2: List[float]
    note: str

    @property
    def mean_mae(self) -> float:
        """Trả MAE trung bình qua tất cả fold."""
        return float(np.mean(self.fold_mae))

    @property
    def std_mae(self) -> float:
        """Trả độ lệch chuẩn MAE để đo độ ổn định giữa các fold."""
        return float(np.std(self.fold_mae))

    @property
    def mean_rmse(self) -> float:
        """Trả RMSE trung bình qua tất cả fold."""
        return float(np.mean(self.fold_rmse))

    @property
    def std_rmse(self) -> float:
        """Trả độ lệch chuẩn RMSE giữa các fold."""
        return float(np.std(self.fold_rmse))

    @property
    def mean_r2(self) -> float:
        """Trả R2 trung bình qua tất cả fold."""
        return float(np.mean(self.fold_r2))

    @property
    def std_r2(self) -> float:
        """Trả độ lệch chuẩn R2 giữa các fold."""
        return float(np.std(self.fold_r2))


@dataclass
class FinalModelResult:
    """Kết quả cuối cùng của một mô hình sau khi fit trên dev_train.

    Attributes:
        name: Tên hiển thị của mô hình.
        slug: Tên ngắn ổn định dùng trong tên cột và tên file.
        train_pred: Prediction trên dev_train trong đơn vị TZS.
        test_pred: Prediction trên test dùng để tạo submission.
        train_mae: MAE trên dev_train.
        train_rmse: RMSE trên dev_train.
        train_r2: R2 trên dev_train.
        feature_count: Số feature hoặc số thành viên ensemble.
        nonzero_coef: Số hệ số khác 0, nếu mô hình có khái niệm hệ số.
        detail: Chuỗi metadata chứa siêu tham số và cách fit.
        coefficients: Hệ số gắn với tên feature, nếu mô hình hỗ trợ.
        holdout_mae: MAE trên holdout 20% chưa dùng trong fit/tuning.
        holdout_rmse: RMSE trên holdout 20% chưa dùng trong fit/tuning.
        holdout_r2: R2 trên holdout 20% chưa dùng trong fit/tuning.
    """
    name: str
    slug: str
    train_pred: np.ndarray
    test_pred: np.ndarray
    train_mae: float
    train_rmse: float
    train_r2: float
    feature_count: int
    nonzero_coef: Optional[int]
    detail: str
    coefficients: Optional[pd.Series] = None
    holdout_mae: float = math.nan
    holdout_rmse: float = math.nan
    holdout_r2: float = math.nan


class KernelRidgeRBF:
    """Kernel Ridge Regression với RBF kernel.

    Mô hình dùng nghiệm dạng đối ngẫu:
        dual_coef = (K + alpha * I)^(-1) y

    Trong đó K là ma trận Gram của RBF kernel và alpha là hệ số điều chuẩn.
    """

    def __init__(self, alpha: float, gamma: float):
        """Khởi tạo hệ số điều chuẩn alpha và độ rộng RBF gamma.

        Args:
            alpha: Hệ số điều chuẩn Ridge, thường là một số dương nhỏ.
            gamma: Độ rộng của RBF kernel, kiểm soát mức độ tương đồng giữa mẫu.
        """
        # Alpha kiểm soát mức phạt Ridge, giúp hệ tuyến tính ổn định hơn.
        self.alpha = alpha

        # Gamma kiểm soát tốc độ giảm của độ tương đồng khi hai mẫu cách xa nhau.
        self.gamma = gamma

        # X_fit lưu tập train vì Prediction kernel cần so sánh mẫu mới với mẫu train.
        self.X_fit: Optional[np.ndarray] = None

        # dual_coef là nghiệm đối ngẫu, mỗi phần tử gắn với một mẫu train.
        self.dual_coef: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y_log: np.ndarray) -> "KernelRidgeRBF":
        """Fit mô hình trên X và target đã biến đổi log1p.

        Args:
            X: Ma trận đặc trưng train, kích thước (n_samples, n_features).
            y_log: Target đã được biến đổi log1p, kích thước (n_samples,).
        """
        # Ép dữ liệu về số thực để các phép nhân ma trận hoạt động nhất quán.
        self.X_fit = X.astype(float, copy=False)

        # Tạo ma trận Gram K, trong đó K[i, j] đo độ tương đồng giữa hai mẫu train.
        kernel = self._rbf_kernel(self.X_fit, self.X_fit)

        # Cộng alpha vào đường chéo tương đương K + alpha*I để điều chuẩn Ridge.
        regularized = kernel + self.alpha * np.eye(kernel.shape[0])

        # Với alpha > 0, K + alpha*I xác định dương và có nghiệm duy nhất.
        self.dual_coef = np.linalg.solve(regularized, y_log)

        # Trả về chính mô hình để hỗ trợ cách gọi KernelRidgeRBF(...).fit(...).
        return self

    def predict_log(self, X: np.ndarray) -> np.ndarray:
        """Prediction target trong không gian log1p cho các mẫu X.

        Args:
            X: Ma trận đặc trưng cần Prediction, kích thước (n_samples, n_features).

        Returns:
            Prediction target đã được biến đổi log1p, kích thước (n_samples,).
        """
        # Không thể Prediction khi chưa có tập train và nghiệm dual từ bước fit.
        if self.X_fit is None or self.dual_coef is None:
            raise RuntimeError("KernelRidgeRBF phải fit trước khi predict.")

        # Tính độ tương đồng giữa từng mẫu cần Prediction và toàn bộ mẫu train.
        kernel = self._rbf_kernel(X.astype(float, copy=False), self.X_fit)

        # Công thức Prediction dạng đối ngẫu: y_hat = K(X, X_train) @ dual_coef.
        return kernel @ self.dual_coef

    def _rbf_kernel(self, X_left: np.ndarray, X_right: np.ndarray) -> np.ndarray:
        """Tính ma trận RBF exp(-gamma * ||x_i - x_j||^2) giữa hai tập mẫu.

        Args:
            X_left: Ma trận đặc trưng của tập mẫu bên trái, kích thước (n_left, n_features).
            X_right: Ma trận đặc trưng của tập mẫu bên phải, kích thước (n_right, n_features).

        Returns:
            Ma trận RBF kernel, kích thước (n_left, n_right), trong đó entry (i, j) là
            exp(-gamma * ||X_left[i] - X_right[j]||^2).
        """
        # Tính ||x_i||^2 cho từng hàng bên trái và chuyển thành vector cột.
        left_norm = np.sum(X_left * X_left, axis=1)[:, None]

        # Tính ||x_j||^2 cho từng hàng bên phải và chuyển thành vector hàng.
        right_norm = np.sum(X_right * X_right, axis=1)[None, :]

        # Dùng ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b; chặn 0 để khử sai số âm nhỏ.
        dist2 = np.maximum(left_norm + right_norm - 2.0 * (X_left @ X_right.T), 0.0)

        # Chuyển khoảng cách bình phương thành độ tương đồng trong khoảng (0, 1].
        return np.exp(-self.gamma * dist2)


def run() -> Dict[str, Path]:
    """Chạy toàn bộ quy trình Preprocessing, huấn luyện, đánh giá và xuất kết quả.

    Mỗi mô hình đi qua ba bước riêng biệt:
        1. Cross-validation trên dev_train 80% để chọn mô hình/siêu tham số.
        2. Fit lại trên dev_train và đánh giá holdout 20% đúng một lượt.
        3. Dùng mô hình đã fit để tạo Prediction competition test.

    Returns:
        Dictionary ánh xạ tên nhóm output sang đường dẫn file hoặc thư mục
        tương ứng trong ``part2/outputs``.
    """
    # Xác định đường dẫn từ vị trí file để không phụ thuộc thư mục đang chạy lệnh.
    part2_dir = Path(__file__).resolve().parent
    data_dir = part2_dir / "data"
    output_dir = part2_dir / "outputs"

    print("=" * 78)
    print("PHÂN TÍCH VÀ XÂY DỰNG CÁC MÔ HÌNH")
    print("=" * 78)
    print("Các mô hình: OLS, OLS+Selection, Ridge, Lasso, ElasticNet, Poly+EN, KRR, Ensemble")

    # Chia train thô trước preprocessing; pipeline chỉ fit trên dev 80% để
    # holdout 20% không ảnh hưởng imputation, vocabulary hay scaler.
    pipe_result, X_holdout, y_holdout = _load_processed_data(data_dir)

    # X_train ở đây là dev_train 80%; X_test vẫn là competition test chính thức.
    X_train = pipe_result.X_train
    X_test = pipe_result.X_test
    y_train = pipe_result.y_train

    # Ghép holdout và competition test để mỗi model chỉ cần fit một lần. Sau
    # khi mọi model đã được chọn/tuning trên dev, Prediction ghép được tách lại.
    X_prediction = np.vstack([X_holdout, X_test])

    # Tên đặc trưng được giữ đúng thứ tự cột để gắn hệ số và xuất importance.
    feature_names = pipe_result.feature_names

    # Đánh dấu phần bắt đầu huấn luyện và đánh giá trong log terminal.
    print("\n" + "-" * 78)
    print("HUẤN LUYỆN VÀ ĐÁNH GIÁ")
    print("-" * 78)

    # results lưu mô hình đã fit trên dev_train và Prediction holdout + test.
    results: List[FinalModelResult] = []

    # cv_results ánh xạ slug mô hình sang kết quả CV; ensemble không có CV riêng.
    cv_results: Dict[str, Optional[CVResult]] = {}

    # Đánh giá OLS bằng 5-fold CV, sau đó fit OLS trên toàn bộ dev_train.
    ols_cv = _cross_validate(
        "OLS",
        X_train,
        y_train,
        lambda X_tr, y_log_tr, X_val: _predict_ols_cost(X_tr, y_log_tr, X_val),
        note="5-fold CV tren dev_train, target log1p, metric TZS",
    )
    ols_result = _fit_ols("OLS", "ols", X_train, y_train, X_prediction, feature_names)
    results.append(ols_result)
    cv_results[ols_result.slug] = ols_cv

    # Với OLS Selected, việc chọn biến bằng p-value được thực hiện lại trong từng fold.
    selected_cv = _cross_validate(
        "OLS_Selected",
        X_train,
        y_train,
        lambda X_tr, y_log_tr, X_val: _predict_ols_selected_cost(
            X_tr, y_log_tr, X_val, feature_names
        ),
        note="5-fold CV, moi fold chon bien bang p-value tren train fold",
    )
    selected_result = _fit_ols_selected(X_train, y_train, X_prediction, feature_names)
    results.append(selected_result)
    cv_results[selected_result.slug] = selected_cv

    # Chọn alpha Ridge bằng CV rồi dùng alpha tốt nhất để fit mô hình cuối cùng.
    # Nested CV (cv_results) dùng để báo cáo hiệu năng không lạc quan; final model
    # vẫn dùng alpha từ _choose_ridge_alpha (fit trên toàn bộ dev_train).
    ridge_alpha, _ = _choose_ridge_alpha(X_train, y_train)
    ridge_result = _fit_ridge(X_train, y_train, X_prediction, feature_names, ridge_alpha)
    results.append(ridge_result)
    cv_results[ridge_result.slug] = _nested_cross_validate_ridge(
        X_train, y_train, alpha_grid=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    )

    # Fit Lasso cuối cùng trên dev_train với alpha chọn bởi _fit_lasso_cv.
    # Nested CV thay _cross_validate(fixed_alpha) để tránh optimistic bias.
    lasso_result = _fit_lasso_cv(X_train, y_train, X_prediction, feature_names)
    results.append(lasso_result)
    cv_results[lasso_result.slug] = _nested_cross_validate_scratch(
        X_train,
        y_train,
        alphas=np.logspace(-4, 0, 9).tolist(),
        l1_ratios=[1.0],
        model_name="Lasso",
    )

    # Fit ElasticNet cuối cùng rồi đánh giá bằng nested CV.
    elastic_result = _fit_elasticnet_cv(X_train, y_train, X_prediction, feature_names)
    results.append(elastic_result)
    cv_results[elastic_result.slug] = _nested_cross_validate_scratch(
        X_train,
        y_train,
        alphas=np.logspace(-4, 0, 7).tolist(),
        l1_ratios=[0.3, 0.5, 0.7, 0.9],
        model_name="ElasticNet",
    )

    # Mở rộng riêng nhóm biến số lên polynomial bậc 2, giữ nguyên biến one-hot.
    X_train_poly, X_prediction_poly, poly_names = _make_polynomial_design(
        X_train, X_prediction, feature_names, pipe_result.feature_types
    )

    # Fit ElasticNet trên không gian đặc trưng polynomial đã mở rộng.
    poly_result = _fit_poly_elasticnet_cv(
        X_train_poly, y_train, X_prediction_poly, poly_names
    )
    results.append(poly_result)
    # Nested CV cho polynomial: polynomial expansion được thực hiện trong từng outer fold.
    cv_results[poly_result.slug] = _nested_cross_validate_poly_elasticnet(
        X_train,
        y_train,
        feature_names=feature_names,
        feature_types=pipe_result.feature_types,
        alphas=np.logspace(-4, 0, 7).tolist(),
        l1_ratios=[0.3, 0.5, 0.7, 0.9],
    )

    # Kernel Ridge tự tạo RBF kernel nên bỏ cột hệ số tự do toàn 1 khỏi đầu vào.
    krr_result, krr_cv = _fit_krr(X_train[:, 1:], y_train, X_prediction[:, 1:])
    results.append(krr_result)
    cv_results[krr_result.slug] = krr_cv

    # Ensemble lấy trung bình Prediction của các mô hình thành viên đã fit phía trên.
    ensemble_result = _fit_ensemble(results, y_train)
    results.append(ensemble_result)

    # Ensemble không có một mô hình đơn để chạy CV trong thiết kế hiện tại.
    cv_results[ensemble_result.slug] = None

    # Holdout chỉ được mở sau khi toàn bộ fit/tuning trên dev đã hoàn tất.
    # Hàm này tính metric một lần rồi giữ lại phần prediction competition test.
    _evaluate_holdout_once(results, y_holdout, X_holdout.shape[0])

    # Ghi bảng so sánh, feature importance, Prediction và từng file submission.
    paths = _write_outputs(output_dir, pipe_result, results, cv_results)

    # In bảng kết quả ngắn ra terminal để kiểm tra nhanh sau khi chạy.
    _print_result_table(results, cv_results)

    # Trả về đường dẫn output để caller có thể sử dụng tiếp theo chương trình.
    return paths


def _load_processed_data(
    data_dir: Path,
) -> Tuple[PipelineResult, np.ndarray, np.ndarray]:
    """Chia raw train 80/20 rồi fit pipeline chỉ trên dev_train."""
    # Dùng median cho biến số và nhãn Unknown cho biến phân loại bị thiếu.
    config = PipelineConfig(data_dir=str(data_dir), missing_method="median")
    train_raw = pd.read_csv(data_dir / config.train_file)
    test_raw = pd.read_csv(data_dir / config.test_file)

    # Random permutation với seed cố định tạo split tái lập được. Sort chỉ số
    # giúp giữ thứ tự gốc bên trong từng tập sau khi chia.
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = rng.permutation(len(train_raw))
    dev_size = int(0.8 * len(train_raw))
    dev_idx = np.sort(shuffled[:dev_size])
    holdout_idx = np.sort(shuffled[dev_size:])
    dev_raw = train_raw.iloc[dev_idx].reset_index(drop=True)
    holdout_raw = train_raw.iloc[holdout_idx].reset_index(drop=True)

    # Mọi preprocessing state chỉ được học từ dev; holdout và competition test
    # chỉ đi qua transform nên không thể ảnh hưởng feature space.
    pipeline = DataPipeline(config).fit(dev_raw)
    dev_result = pipeline.transform(dev_raw)
    holdout_result = pipeline.transform(holdout_raw)
    test_result = pipeline.transform(test_raw)
    pipe_result = pipeline.build_result(dev_result, test_result)

    if holdout_result.y is None:
        raise RuntimeError("Holdout không chứa target.")

    print(
        f"  Raw split: dev_train={len(dev_raw):,} rows, "
        f"holdout={len(holdout_raw):,} rows, seed={RANDOM_STATE}"
    )
    return pipe_result, holdout_result.X, holdout_result.y


def _evaluate_holdout_once(
    results: List[FinalModelResult],
    y_holdout: np.ndarray,
    holdout_rows: int,
) -> None:
    """Tính metric holdout một lượt và tách prediction competition test."""
    for result in results:
        if result.test_pred.shape[0] < holdout_rows:
            raise ValueError(f"Prediction của {result.name} ngắn hơn holdout.")

        holdout_pred = result.test_pred[:holdout_rows]
        result.test_pred = result.test_pred[holdout_rows:]
        (
            result.holdout_mae,
            result.holdout_rmse,
            result.holdout_r2,
        ) = _metrics(y_holdout, holdout_pred)


def _to_log_target(y_cost: np.ndarray) -> np.ndarray:
    """Đưa target từ TZS về log1p để giảm skewness trước khi fit."""
    # Chuyển input về ndarray float và chặn chi phí âm trước khi lấy log1p.
    return np.log1p(np.clip(np.asarray(y_cost, dtype=float), a_min=0.0, a_max=None))


def _to_cost(y_log_pred: np.ndarray) -> np.ndarray:
    """Đưa Prediction log-space về đơn vị TZS và chặn giá trị âm."""
    # Giới hạn log prediction để expm1 không tràn số; log=0 tương ứng chi phí=0.
    clipped_log = np.clip(np.asarray(y_log_pred, dtype=float), a_min=0.0, a_max=25.0)

    # Đảo log1p bằng expm1 và thay mọi giá trị không hữu hạn bằng 0.
    return np.nan_to_num(np.expm1(clipped_log), nan=0.0, posinf=0.0, neginf=0.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Tính MAE, RMSE và R2 trên cùng đơn vị gốc TZS."""
    # Chuẩn hóa kiểu dữ liệu để các phép toán metric luôn dùng số thực.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # MAE đo độ lệch tuyệt đối trung bình, ít nhạy với outlier hơn RMSE.
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # RMSE phạt mạnh các Prediction sai lệch lớn do bình phương phần dư.
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # TSS đo tổng biến thiên của target quanh giá trị trung bình.
    tss = float(np.sum((y_true - np.mean(y_true)) ** 2))

    # RSS đo tổng biến thiên còn lại sau khi dùng Prediction của mô hình.
    rss = float(np.sum((y_true - y_pred) ** 2))

    # R2 = 1 - RSS/TSS; trả 0 khi target là hằng số để tránh chia cho 0.
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    return mae, rmse, float(r2)


def _cross_validate(
    model_name: str,
    X: np.ndarray,
    y_cost: np.ndarray,
    predict_fold: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    note: str,
    k: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> CVResult:
    """Chạy k-fold CV cho một hàm fit/predict bất kỳ.

    ``predict_fold`` nhận train fold, target train đã log1p và validation fold,
    sau đó trả Prediction validation trong đơn vị TZS. Nhờ vậy mọi mô hình được
    đánh giá bằng cùng một quy trình và cùng bộ metric.
    """
    # Lưu metric từng fold để sau đó tính trung bình và độ lệch chuẩn.
    fold_mae: List[float] = []
    fold_rmse: List[float] = []
    fold_r2: List[float] = []

    # Mỗi vòng dùng một fold làm validation và các fold còn lại làm train.
    for train_idx, valid_idx in _kfold_indices(X.shape[0], k, random_state):
        # Tách ma trận đặc trưng theo chỉ số fold scratch vừa tạo.
        X_tr, X_val = X[train_idx], X[valid_idx]

        # Giữ target validation ở TZS để metric cuối phản ánh sai số thực tế.
        y_tr_cost, y_val_cost = y_cost[train_idx], y_cost[valid_idx]

        # Mô hình được fit trên log1p target để giảm ảnh hưởng của độ lệch phải.
        y_tr_log = _to_log_target(y_tr_cost)

        # Hàm callback tự chịu trách nhiệm fit trên fold train và Prediction fold validation.
        y_val_pred = predict_fold(X_tr, y_tr_log, X_val)

        # Tính metric validation trong đơn vị TZS và lưu lại theo fold.
        mae, rmse, r2 = _metrics(y_val_cost, y_val_pred)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)

    # Đóng gói toàn bộ metric để các property tính mean/std khi cần.
    cv = CVResult(model_name, fold_mae, fold_rmse, fold_r2, note)

    # In tóm tắt ngay sau CV để theo dõi tiến trình khi pipeline chạy lâu.
    print(
        f"  CV {model_name:<24} "
        f"RMSE={cv.mean_rmse:,.2f} +/- {cv.std_rmse:,.2f} | "
        f"R2={cv.mean_r2:.4f} +/- {cv.std_r2:.4f}"
    )
    return cv


def _kfold_indices(
    n_rows: int,
    k: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Tạo k-fold indices bằng shuffle có seed, không dùng thư viện CV."""
    if k < 2 or k > n_rows:
        raise ValueError(f"k phải nằm trong [2, {n_rows}].")

    shuffled = np.random.default_rng(random_state).permutation(n_rows)
    fold_sizes = np.full(k, n_rows // k, dtype=int)
    fold_sizes[: n_rows % k] += 1
    folds: List[np.ndarray] = []
    start = 0
    for fold_size in fold_sizes:
        end = start + int(fold_size)
        folds.append(shuffled[start:end])
        start = end

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_index, valid_idx in enumerate(folds):
        train_idx = np.concatenate([
            fold for index, fold in enumerate(folds) if index != fold_index
        ])
        splits.append((train_idx, valid_idx))
    return splits


def _fit_ols_beta(X: np.ndarray, y_log: np.ndarray) -> np.ndarray:
    """Ước lượng OLS bằng ``part1.ols_fit`` triển khai từ scratch.

    Một train fold có thể không chứa category hiếm, làm cột dummy tương ứng
    thành toàn 0. Các cột zero đó được bỏ tạm trước khi fit và nhận hệ số 0 khi
    khôi phục vector đầy đủ.
    """
    nonzero_columns = [
        column
        for column in range(X.shape[1])
        if column == 0 or bool(np.any(np.abs(X[:, column]) > 1e-12))
    ]
    reduced_X = X[:, nonzero_columns]
    result = ols_fit(reduced_X.tolist(), y_log.tolist())
    if not result.success:
        raise RuntimeError(f"Part 1 ols_fit thất bại: {result.message}")

    beta = np.zeros(X.shape[1], dtype=np.float64)
    beta[nonzero_columns] = np.asarray(result.beta_hat, dtype=np.float64)
    return beta


def _predict_ols_cost(X_train: np.ndarray, y_log: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    """Fit OLS trên một train fold và trả Prediction X_pred trong đơn vị TZS."""
    # Ước lượng vector hệ số trên target log1p của train fold.
    beta = _fit_ols_beta(X_train, y_log)

    # Nhân ma trận để Prediction log-space rồi đổi ngược về chi phí TZS.
    return _to_cost(X_pred @ beta)


def _fit_ols(
    name: str,
    slug: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
) -> FinalModelResult:
    """Fit OLS cuối cùng trên toàn bộ dev_train."""
    # Giảm skewness của target trước khi ước lượng hệ số tuyến tính.
    y_log = _to_log_target(y_train)

    # Fit OLS trên toàn bộ dev_train để tạo mô hình dùng cho holdout/submission.
    beta = _fit_ols_beta(X_train, y_log)

    # Tạo Prediction train để tính metric và Prediction test để xuất submission.
    train_pred = _to_cost(X_train @ beta)
    test_pred = _to_cost(X_test @ beta)

    # Đánh giá độ khớp trên train theo đơn vị TZS.
    mae, rmse, r2 = _metrics(y_train, train_pred)

    # Gắn hệ số với đúng tên đặc trưng để xuất feature importance.
    coef = pd.Series(beta, index=feature_names, name=name)

    # Đóng gói Prediction, metric, hệ số và metadata của mô hình.
    return FinalModelResult(
        name=name,
        slug=slug,
        train_pred=train_pred,
        test_pred=test_pred,
        train_mae=mae,
        train_rmse=rmse,
        train_r2=r2,
        feature_count=X_train.shape[1],
        nonzero_coef=int(np.sum(np.abs(beta) > 1e-10)),
        detail="closed_form=normal_equations; solver=part1_ols_fit; target=log1p",
        coefficients=coef,
    )


def _select_by_pvalue(
    X: np.ndarray,
    y_log: np.ndarray,
    p_threshold: float = 0.05,
) -> List[int]:
    """Chọn biến cho OLS dựa trên kiểm định t của từng hệ số."""
    # Fit mô hình đầy đủ để có hệ số và phần dư phục vụ kiểm định.
    beta = _fit_ols_beta(X, y_log)
    residuals = y_log - X @ beta

    # Rank thực của X được dùng để tính đúng bậc tự do khi có đa cộng tuyến.
    rank = int(np.linalg.matrix_rank(X))
    dof = max(X.shape[0] - rank, 1)

    # Ước lượng phương sai nhiễu từ tổng bình phương phần dư.
    sigma2 = float(np.sum(residuals ** 2) / dof)

    # Dùng pseudo-inverse vì X.T @ X có thể suy biến do nhiều biến one-hot.
    cov_beta = sigma2 * np.linalg.pinv(X.T @ X)

    # Sai số chuẩn là căn bậc hai đường chéo ma trận hiệp phương sai hệ số.
    se = np.sqrt(np.maximum(np.diag(cov_beta), 1e-15))

    # Tính t-statistic; errstate tránh cảnh báo khi sai số chuẩn quá nhỏ.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se

    # Kiểm định hai phía bằng phân phối Student-t được triển khai từ đầu.
    # NaN do phép chia không xác định được xem như t = 0, tức p-value = 1.
    p_values = np.array([
        student_t_two_sided_pvalue(0.0 if math.isnan(float(t_stat)) else float(t_stat), dof)
        for t_stat in t_stats
    ])

    # Luôn giữ hệ số tự do ở cột 0 và chỉ giữ feature có p-value dưới ngưỡng.
    kept = [0] + [idx for idx in range(1, X.shape[1]) if p_values[idx] < p_threshold]

    # Nếu không feature nào đạt ngưỡng, giữ feature có p-value nhỏ nhất.
    if len(kept) == 1:
        best_idx = int(np.nanargmin(p_values[1:])) + 1
        kept.append(best_idx)
    return kept


def _predict_ols_selected_cost(
    X_train: np.ndarray,
    y_log: np.ndarray,
    X_pred: np.ndarray,
    feature_names: List[str],
) -> np.ndarray:
    """Chọn biến và Prediction OLS Selected cho một fold cross-validation."""
    # Chọn feature chỉ từ train fold để tránh data leakage sang validation fold.
    kept = _select_by_pvalue(X_train, y_log)

    # Fit lại OLS chỉ trên các cột được giữ.
    beta = _fit_ols_beta(X_train[:, kept], y_log)

    # Prediction bằng cùng tập cột rồi đổi log-space về TZS.
    return _to_cost(X_pred[:, kept] @ beta)


def _fit_ols_selected(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    p_threshold: float = 0.05,
) -> FinalModelResult:
    """Fit OLS sau khi chọn biến bằng p-value trên toàn bộ dev_train."""
    # Chuyển target sang log-space và chọn feature trên toàn bộ dev_train.
    y_log = _to_log_target(y_train)
    kept = _select_by_pvalue(X_train, y_log, p_threshold)

    # Fit OLS cuối cùng chỉ bằng các feature được chọn.
    beta_selected = _fit_ols_beta(X_train[:, kept], y_log)

    # Khôi phục vector hệ số đủ chiều; feature bị loại nhận hệ số bằng 0.
    beta_full = np.zeros(X_train.shape[1])
    beta_full[kept] = beta_selected

    # Prediction train/test bằng đúng các cột đã chọn.
    train_pred = _to_cost(X_train[:, kept] @ beta_selected)
    test_pred = _to_cost(X_test[:, kept] @ beta_selected)

    # Tính metric train và gắn hệ số đầy đủ với tên feature ban đầu.
    mae, rmse, r2 = _metrics(y_train, train_pred)
    coef = pd.Series(beta_full, index=feature_names, name="OLS_Selected")

    # Lưu cả số feature được chọn và ngưỡng p-value để tái tạo thí nghiệm.
    return FinalModelResult(
        name="OLS_Selected",
        slug="ols_selected",
        train_pred=train_pred,
        test_pred=test_pred,
        train_mae=mae,
        train_rmse=rmse,
        train_r2=r2,
        feature_count=len(kept),
        nonzero_coef=int(np.sum(np.abs(beta_selected) > 1e-10)),
        detail=f"p_threshold={p_threshold}; selected={len(kept)-1}/{X_train.shape[1]-1}",
        coefficients=coef,
    )


def _fit_ridge_beta(X: np.ndarray, y_log: np.ndarray, alpha: float) -> np.ndarray:
    """Fit Ridge bằng công thức đóng, không phạt hệ số hệ số tự do."""
    # Ma trận đơn vị tạo penalty L2 cho từng hệ số.
    penalty = np.eye(X.shape[1])

    # Không phạt hệ số tự do để mô hình vẫn tự do điều chỉnh mức nền.
    penalty[0, 0] = 0.0

    # Hệ phương trình chuẩn Ridge: (X.T X + alpha P) beta = X.T y.
    system = X.T @ X + alpha * penalty
    rhs = X.T @ y_log

    # Với alpha > 0, hệ Ridge khả nghịch ngay cả khi X có đa cộng tuyến.
    return np.linalg.solve(system, rhs)


def _predict_ridge_cost(
    X_train: np.ndarray,
    y_log: np.ndarray,
    X_pred: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit Ridge trên một train fold và trả Prediction X_pred trong đơn vị TZS."""
    # Fit hệ số Ridge với alpha cố định.
    beta = _fit_ridge_beta(X_train, y_log, alpha)

    # Prediction log-space rồi đổi ngược về TZS.
    return _to_cost(X_pred @ beta)


def _choose_ridge_alpha(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, CVResult]:
    """Chọn lambda Ridge bằng CV theo RMSE trên đơn vị TZS."""
    # Grid trải nhiều bậc độ lớn để khảo sát từ phạt nhẹ đến phạt mạnh.
    alpha_grid = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_alpha = alpha_grid[0]
    best_cv: Optional[CVResult] = None

    print("\n  Chọn lambda cho Ridge:")

    # Chạy cùng quy trình CV cho từng alpha và chọn RMSE trung bình thấp nhất.
    for alpha in alpha_grid:
        cv = _cross_validate(
            f"Ridge(alpha={alpha:g})",
            X_train,
            y_train,
            lambda X_tr, y_log_tr, X_val, a=alpha: _predict_ridge_cost(
                X_tr, y_log_tr, X_val, a
            ),
            note=f"5-fold CV, alpha={alpha:g}",
        )

        # Cập nhật ứng viên tốt nhất ngay sau khi đánh giá mỗi alpha.
        if best_cv is None or cv.mean_rmse < best_cv.mean_rmse:
            best_alpha = alpha
            best_cv = cv

    # Guard bảo vệ trường hợp grid rỗng hoặc CV không tạo được kết quả.
    if best_cv is None:
        raise RuntimeError("Không chọn được alpha cho Ridge.")

    # Chuẩn hóa tên và ghi alpha thắng cuộc vào metadata trả về.
    best_cv.model_name = "Ridge"
    best_cv.note = f"5-fold CV tren dev_train; alpha={best_alpha:g}"
    return best_alpha, best_cv


def _fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    alpha: float,
) -> FinalModelResult:
    """Fit Ridge cuối cùng trên toàn bộ dev_train bằng alpha đã chọn."""
    # Fit hệ số trên target log1p để giảm tác động của chi phí cực lớn.
    y_log = _to_log_target(y_train)
    beta = _fit_ridge_beta(X_train, y_log, alpha)

    # Tạo Prediction train/test và đánh giá trên đơn vị chi phí gốc.
    train_pred = _to_cost(X_train @ beta)
    test_pred = _to_cost(X_test @ beta)
    mae, rmse, r2 = _metrics(y_train, train_pred)

    # Gắn hệ số Ridge với tên feature để phục vụ giải thích.
    coef = pd.Series(beta, index=feature_names, name="Ridge")

    # Đóng gói mô hình cuối cùng cùng alpha đã sử dụng.
    return FinalModelResult(
        name="Ridge",
        slug="ridge",
        train_pred=train_pred,
        test_pred=test_pred,
        train_mae=mae,
        train_rmse=rmse,
        train_r2=r2,
        feature_count=X_train.shape[1],
        nonzero_coef=int(np.sum(np.abs(beta) > 1e-10)),
        detail=f"alpha={alpha:g}; target=log1p; intercept_not_penalized",
        coefficients=coef,
    )


def _inner_cv_ridge(
    X: np.ndarray,
    y_cost: np.ndarray,
    alpha_grid: List[float],
    k: int,
) -> float:
    """Inner CV của nested CV Ridge: chọn alpha tốt nhất trên outer_train.

    Được gọi bên trong mỗi outer fold của nested CV. X và y_cost ở đây là
    outer_train — tức là phần dữ liệu chưa bị outer fold giữ lại để test.
    Hàm chia tiếp outer_train thành k inner fold để tìm alpha cho RMSE thấp
    nhất, nhằm tránh dùng cùng một tập dữ liệu cho cả tuning lẫn evaluation.
    """
    best_alpha = alpha_grid[0]
    best_rmse = float("inf")
    for alpha in alpha_grid:
        fold_rmse: List[float] = []
        # Chia outer_train thành k inner fold, dùng seed khác outer để
        # tránh fold structure trùng nhau giữa hai vòng lặp.
        for tr_idx, val_idx in _kfold_indices(X.shape[0], k, RANDOM_STATE + 1):
            # Fit trên log1p(y) để khớp với cách model được huấn luyện thật.
            beta = _fit_ridge_beta(X[tr_idx], _to_log_target(y_cost[tr_idx]), alpha)
            # Đánh giá RMSE trên đơn vị TZS gốc (sau expm1) để metric có nghĩa thực tế.
            _, rmse, _ = _metrics(y_cost[val_idx], _to_cost(X[val_idx] @ beta))
            fold_rmse.append(rmse)
        # Chọn alpha cho mean RMSE thấp nhất qua k inner fold.
        if float(np.mean(fold_rmse)) < best_rmse:
            best_rmse = float(np.mean(fold_rmse))
            best_alpha = alpha
    return best_alpha


def _nested_cross_validate_ridge(
    X: np.ndarray,
    y_cost: np.ndarray,
    alpha_grid: List[float],
    k_outer: int = CV_FOLDS,
    k_inner: int = INNER_CV_FOLDS,
) -> CVResult:
    """Nested CV cho Ridge: mỗi outer fold chạy inner CV riêng để chọn alpha.

    Khác với _cross_validate(Ridge, fixed_alpha), hàm này không dùng alpha được
    chọn trên toàn bộ dev_train. Thay vào đó, mỗi outer fold có alpha riêng
    được chọn bằng inner CV trên outer_train, sao cho outer_val không được dùng
    để chọn hyperparameter → ước lượng CV không bị lạc quan.
    """
    fold_mae: List[float] = []
    fold_rmse: List[float] = []
    fold_r2: List[float] = []

    print(f"\n  Nested CV Ridge (outer={k_outer}, inner={k_inner}):")
    for fold_i, (outer_tr_idx, outer_val_idx) in enumerate(
        _kfold_indices(X.shape[0], k_outer, RANDOM_STATE)
    ):
        # Tách outer_train và outer_val cho vòng lặp outer này.
        X_otr, y_otr = X[outer_tr_idx], y_cost[outer_tr_idx]
        X_oval, y_oval = X[outer_val_idx], y_cost[outer_val_idx]

        # Inner CV chỉ nhìn thấy outer_train → alpha được chọn độc lập với outer_val.
        best_alpha = _inner_cv_ridge(X_otr, y_otr, alpha_grid, k_inner)

        # Fit lại trên toàn bộ outer_train với alpha tốt nhất vừa tìm được.
        beta = _fit_ridge_beta(X_otr, _to_log_target(y_otr), best_alpha)
        # Đánh giá trên outer_val — đây là dữ liệu chưa được dùng ở bất kỳ bước nào trên.
        pred = _to_cost(X_oval @ beta)
        mae, rmse, r2 = _metrics(y_oval, pred)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        print(f"    fold {fold_i + 1}: alpha*={best_alpha:g}  RMSE={rmse:,.0f}  R2={r2:.4f}")

    cv = CVResult(
        model_name="Ridge",
        fold_mae=fold_mae,
        fold_rmse=fold_rmse,
        fold_r2=fold_r2,
        note=(
            f"nested {k_outer}-fold outer / {k_inner}-fold inner; "
            f"alpha grid={alpha_grid}; alpha retuned per outer fold"
        ),
    )
    print(
        f"  Nested CV Ridge           "
        f"RMSE={cv.mean_rmse:,.2f} +/- {cv.std_rmse:,.2f} | "
        f"R2={cv.mean_r2:.4f} +/- {cv.std_r2:.4f}"
    )
    return cv


def _inner_cv_scratch(
    X: np.ndarray,
    y_cost: np.ndarray,
    alphas: List[float],
    l1_ratios: List[float],
    k: int,
) -> Tuple[float, float]:
    """Inner CV cho Lasso/ElasticNet scratch: trả (alpha, l1_ratio) tốt nhất.

    Tương tự _inner_cv_ridge nhưng tìm kiếm trên lưới 2 chiều (alpha, l1_ratio).
    l1_ratio=1.0 tương đương Lasso thuần; các giá trị trung gian là ElasticNet.
    X và y_cost là outer_train của vòng outer đang chạy.
    """
    best: Tuple[float, float] = (alphas[0], l1_ratios[0])
    best_rmse = float("inf")
    # Duyệt toàn bộ lưới hyperparameter; độ phức tạp O(|l1_ratios| × |alphas| × k).
    for l1_ratio in l1_ratios:
        for alpha in alphas:
            fold_rmse: List[float] = []
            for tr_idx, val_idx in _kfold_indices(X.shape[0], k, RANDOM_STATE + 1):
                beta = _fit_scratch_regularized_beta(
                    X[tr_idx], _to_log_target(y_cost[tr_idx]), alpha, l1_ratio
                )
                _, rmse, _ = _metrics(y_cost[val_idx], _to_cost(X[val_idx] @ beta))
                fold_rmse.append(rmse)
            mean_rmse = float(np.mean(fold_rmse))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best = (alpha, l1_ratio)
    return best


def _nested_cross_validate_scratch(
    X: np.ndarray,
    y_cost: np.ndarray,
    alphas: List[float],
    l1_ratios: List[float],
    model_name: str,
    k_outer: int = CV_FOLDS,
    k_inner: int = INNER_CV_FOLDS,
) -> CVResult:
    """Nested CV cho Lasso/ElasticNet scratch.

    Inner loop chọn (alpha, l1_ratio) trên outer_train; outer loop đánh giá
    trên outer_val với hyperparameter đó. Không có thông tin nào từ outer_val
    ảnh hưởng đến việc chọn hyperparameter.
    """
    fold_mae: List[float] = []
    fold_rmse: List[float] = []
    fold_r2: List[float] = []

    print(f"\n  Nested CV {model_name} (outer={k_outer}, inner={k_inner}):")
    for fold_i, (outer_tr_idx, outer_val_idx) in enumerate(
        _kfold_indices(X.shape[0], k_outer, RANDOM_STATE)
    ):
        # Tách outer_train và outer_val cho vòng lặp outer này.
        X_otr, y_otr = X[outer_tr_idx], y_cost[outer_tr_idx]
        X_oval, y_oval = X[outer_val_idx], y_cost[outer_val_idx]

        # Inner CV tìm (alpha, l1_ratio) tối ưu chỉ trên outer_train.
        best_alpha, best_l1 = _inner_cv_scratch(X_otr, y_otr, alphas, l1_ratios, k_inner)

        # Fit lại trên toàn bộ outer_train với hyperparameter tốt nhất.
        beta = _fit_scratch_regularized_beta(
            X_otr, _to_log_target(y_otr), best_alpha, best_l1
        )
        # Đánh giá trên outer_val — chưa được nhìn thấy ở bất kỳ bước nào trên.
        pred = _to_cost(X_oval @ beta)
        mae, rmse, r2 = _metrics(y_oval, pred)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        print(
            f"    fold {fold_i + 1}: alpha*={best_alpha:.4g}  "
            f"l1*={best_l1:.2g}  RMSE={rmse:,.0f}  R2={r2:.4f}"
        )

    cv = CVResult(
        model_name=model_name,
        fold_mae=fold_mae,
        fold_rmse=fold_rmse,
        fold_r2=fold_r2,
        note=(
            f"nested {k_outer}-fold outer / {k_inner}-fold inner; "
            f"hyperparams retuned per outer fold"
        ),
    )
    print(
        f"  Nested CV {model_name:<20} "
        f"RMSE={cv.mean_rmse:,.2f} +/- {cv.std_rmse:,.2f} | "
        f"R2={cv.mean_r2:.4f} +/- {cv.std_r2:.4f}"
    )
    return cv


def _nested_cross_validate_poly_elasticnet(
    X: np.ndarray,
    y_cost: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[str, str],
    alphas: List[float],
    l1_ratios: List[float],
    k_outer: int = CV_FOLDS,
    k_inner: int = INNER_CV_FOLDS,
) -> CVResult:
    """Nested CV cho Polynomial+ElasticNet.

    Polynomial expansion được thực hiện bên trong từng outer fold để tránh
    data leakage: nếu expand trên toàn bộ X trước rồi mới CV, thống kê
    scaling của các cross-term sẽ bị rò rỉ từ outer_val vào outer_train.
    """
    fold_mae: List[float] = []
    fold_rmse: List[float] = []
    fold_r2: List[float] = []

    print(f"\n  Nested CV Poly2+ElasticNet (outer={k_outer}, inner={k_inner}):")
    for fold_i, (outer_tr_idx, outer_val_idx) in enumerate(
        _kfold_indices(X.shape[0], k_outer, RANDOM_STATE)
    ):
        # Lấy phần raw (trước polynomial) của outer_train và outer_val.
        X_otr_raw = X[outer_tr_idx]
        X_oval_raw = X[outer_val_idx]
        y_otr = y_cost[outer_tr_idx]
        y_oval = y_cost[outer_val_idx]

        # Expand polynomial chỉ từ outer_train rồi áp transform lên outer_val.
        # outer_val không được nhìn thấy khi quyết định cách expand hay scale.
        X_otr_poly, X_oval_poly, _ = _make_polynomial_design(
            X_otr_raw, X_oval_raw, feature_names, feature_types
        )

        # Inner CV tìm hyperparameter tối ưu trên không gian đã expand.
        best_alpha, best_l1 = _inner_cv_scratch(
            X_otr_poly, y_otr, alphas, l1_ratios, k_inner
        )

        # Fit ElasticNet trên outer_train đã expand với hyperparameter tốt nhất.
        beta = _fit_scratch_regularized_beta(
            X_otr_poly, _to_log_target(y_otr), best_alpha, best_l1
        )
        # Đánh giá trên outer_val — hoàn toàn sạch, chưa tham gia bất kỳ bước nào trên.
        pred = _to_cost(X_oval_poly @ beta)
        mae, rmse, r2 = _metrics(y_oval, pred)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        print(
            f"    fold {fold_i + 1}: alpha*={best_alpha:.4g}  "
            f"l1*={best_l1:.2g}  RMSE={rmse:,.0f}  R2={r2:.4f}"
        )

    cv = CVResult(
        model_name="Polynomial2_ElasticNet",
        fold_mae=fold_mae,
        fold_rmse=fold_rmse,
        fold_r2=fold_r2,
        note=(
            f"nested {k_outer}-fold outer / {k_inner}-fold inner; "
            f"poly expansion + hyperparams retuned per outer fold"
        ),
    )
    print(
        f"  Nested CV Poly2_EN            "
        f"RMSE={cv.mean_rmse:,.2f} +/- {cv.std_rmse:,.2f} | "
        f"R2={cv.mean_r2:.4f} +/- {cv.std_r2:.4f}"
    )
    return cv


def _fit_lasso_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
) -> FinalModelResult:
    """Chọn alpha bằng manual CV rồi fit Lasso scratch của Part 1."""
    y_log = _to_log_target(y_train)
    alpha, _ = _choose_scratch_regularization(
        X_train,
        y_train,
        alphas=np.logspace(-4, 0, 9).tolist(),
        l1_ratios=[1.0],
        model_name="Lasso",
    )
    beta = _fit_scratch_regularized_beta(X_train, y_log, alpha, l1_ratio=1.0)
    train_pred = _to_cost(X_train @ beta)
    test_pred = _to_cost(X_test @ beta)
    mae, rmse, r2 = _metrics(y_train, train_pred)
    coef = pd.Series(beta, index=feature_names, name="Lasso")
    return FinalModelResult(
        name="Lasso",
        slug="lasso",
        train_pred=train_pred,
        test_pred=test_pred,
        train_mae=mae,
        train_rmse=rmse,
        train_r2=r2,
        feature_count=X_train.shape[1],
        nonzero_coef=int(np.sum(np.abs(beta[1:]) > 1e-10)),
        detail=f"alpha={alpha:.12g}; target=log1p; manual_cv=3-fold; solver=scratch_coordinate_descent",
        coefficients=coef,
    )


def _fit_elasticnet_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
) -> FinalModelResult:
    """Chọn alpha/l1_ratio bằng manual CV rồi fit ElasticNet scratch."""
    y_log = _to_log_target(y_train)
    alpha, l1_ratio = _choose_scratch_regularization(
        X_train,
        y_train,
        alphas=np.logspace(-4, 0, 7).tolist(),
        l1_ratios=[0.3, 0.5, 0.7, 0.9],
        model_name="ElasticNet",
    )
    beta = _fit_scratch_regularized_beta(X_train, y_log, alpha, l1_ratio)
    train_pred = _to_cost(X_train @ beta)
    test_pred = _to_cost(X_test @ beta)
    mae, rmse, r2 = _metrics(y_train, train_pred)
    coef = pd.Series(beta, index=feature_names, name="ElasticNet")
    return FinalModelResult(
        name="ElasticNet",
        slug="elasticnet",
        train_pred=train_pred,
        test_pred=test_pred,
        train_mae=mae,
        train_rmse=rmse,
        train_r2=r2,
        feature_count=X_train.shape[1],
        nonzero_coef=int(np.sum(np.abs(beta[1:]) > 1e-10)),
        detail=(
            f"alpha={alpha:.12g}; l1_ratio={l1_ratio:.12g}; "
            "target=log1p; manual_cv=3-fold; solver=scratch_coordinate_descent"
        ),
        coefficients=coef,
    )


def _fit_scratch_regularized_beta(
    X: np.ndarray,
    y_log: np.ndarray,
    alpha: float,
    l1_ratio: float,
) -> np.ndarray:
    """Coordinate descent scratch theo đúng công thức regularization Part 1.

    Các tích vô hướng được vector hóa để dữ liệu Part 2 chạy được trong thời
    gian hợp lý; thuật toán cập nhật và soft-threshold vẫn được tự triển khai,
    không gọi estimator hay optimizer từ thư viện.
    """
    n_rows = X.shape[0]
    lambda_l1 = 2.0 * n_rows * alpha * l1_ratio
    lambda_l2 = n_rows * alpha * (1.0 - l1_ratio)
    beta = np.zeros(X.shape[1], dtype=np.float64)
    residual = np.asarray(y_log, dtype=np.float64).copy()
    squared_norms = np.sum(X * X, axis=0)

    for _ in range(1500):
        largest_change = 0.0
        for column in range(X.shape[1]):
            if squared_norms[column] < 1e-12:
                continue
            old_value = beta[column]
            feature = X[:, column]
            rho = float(feature @ residual + squared_norms[column] * old_value)
            if column == 0:
                new_value = rho / squared_norms[column]
            else:
                new_value = (
                    _soft_threshold(rho, lambda_l1 / 2.0)
                    / (squared_norms[column] + lambda_l2)
                )
            delta = new_value - old_value
            beta[column] = new_value
            largest_change = max(largest_change, abs(delta))
            if abs(delta) > 1e-15:
                residual -= feature * delta
        if largest_change < 1e-5:
            break
    return beta


def _choose_scratch_regularization(
    X: np.ndarray,
    y_cost: np.ndarray,
    alphas: List[float],
    l1_ratios: List[float],
    model_name: str,
    k: int = 3,
) -> Tuple[float, float]:
    """Manual CV chọn alpha/l1_ratio theo RMSE TZS."""
    best_alpha = alphas[0]
    best_l1_ratio = l1_ratios[0]
    best_rmse = float("inf")
    splits = _kfold_indices(X.shape[0], k, RANDOM_STATE)

    print(f"\n  Chọn tham số cho {model_name} bằng manual {k}-fold CV:")
    for l1_ratio in l1_ratios:
        for alpha in alphas:
            fold_rmse: List[float] = []
            for train_idx, valid_idx in splits:
                beta = _fit_scratch_regularized_beta(
                    X[train_idx],
                    _to_log_target(y_cost[train_idx]),
                    alpha,
                    l1_ratio,
                )
                prediction = _to_cost(X[valid_idx] @ beta)
                fold_rmse.append(_metrics(y_cost[valid_idx], prediction)[1])
            mean_rmse = float(np.mean(fold_rmse))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_alpha = alpha
                best_l1_ratio = l1_ratio

    print(
        f"    best alpha={best_alpha:.6g}, l1_ratio={best_l1_ratio:.3g}, "
        f"RMSE={best_rmse:,.2f}"
    )
    return best_alpha, best_l1_ratio


def _make_polynomial_design(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[str, str],
    degree: int = 2,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Tạo feature đa thức cho nhóm biến số, giữ nguyên one-hot category.

    Không mở rộng các biến one-hot vì bình phương biến nhị phân không thêm
    thông tin, còn tương tác giữa hàng trăm dummy dễ làm số chiều tăng mạnh.
    """
    if degree != 2:
        raise ValueError("Polynomial expansion scratch hiện hỗ trợ degree=2.")

    # Xác định vị trí các cột numeric từ metadata của DataPipeline.
    numeric_indices = [
        idx for idx, name in enumerate(feature_names)
        if feature_types.get(name) == "numeric"
    ]

    # Xác định vị trí các cột one-hot categorical để ghép lại nguyên trạng.
    categorical_indices = [
        idx for idx, name in enumerate(feature_names)
        if feature_types.get(name) == "categorical"
    ]

    # Polynomial model không có ý nghĩa nếu pipeline không tìm thấy biến numeric.
    if not numeric_indices:
        raise RuntimeError("Không tìm thấy biến numeric để tạo polynomial features.")

    # Lấy tên theo đúng thứ tự cột để tạo tên polynomial và output cuối.
    numeric_names = [feature_names[idx] for idx in numeric_indices]
    categorical_names = [feature_names[idx] for idx in categorical_indices]

    # Mở rộng theo thứ tự: biến bậc một, bình phương và tương tác.
    train_numeric_poly, poly_names = _polynomial_degree_two(
        X_train[:, numeric_indices], numeric_names
    )
    test_numeric_poly, _ = _polynomial_degree_two(
        X_test[:, numeric_indices], numeric_names
    )

    # Tách khối categorical thành ndarray 2 chiều để contract ghép cột rõ ràng.
    train_categorical: np.ndarray = np.asarray(
        X_train[:, categorical_indices],
        dtype=np.float64,
    )
    test_categorical: np.ndarray = np.asarray(
        X_test[:, categorical_indices],
        dtype=np.float64,
    )

    # Cấp phát trước rồi gán từng khối để giữ contract ndarray 2 chiều rõ ràng.
    n_poly = train_numeric_poly.shape[1]
    n_categorical = train_categorical.shape[1]
    X_train_poly: np.ndarray = np.empty(
        (X_train.shape[0], 1 + n_poly + n_categorical),
        dtype=np.float64,
    )
    X_test_poly: np.ndarray = np.empty(
        (X_test.shape[0], 1 + n_poly + n_categorical),
        dtype=np.float64,
    )

    # hệ số tự do đầu tiên, polynomial tiếp theo, categorical reference-code cuối.
    X_train_poly[:, 0] = 1.0
    X_train_poly[:, 1:1 + n_poly] = train_numeric_poly
    X_train_poly[:, 1 + n_poly:] = train_categorical
    X_test_poly[:, 0] = 1.0
    X_test_poly[:, 1:1 + n_poly] = test_numeric_poly
    X_test_poly[:, 1 + n_poly:] = test_categorical

    # Tên cột phải đi cùng đúng thứ tự ghép ma trận phía trên.
    names = ["hệ số tự do"] + poly_names + categorical_names
    return X_train_poly, X_test_poly, names


def _polynomial_degree_two(
    X_numeric: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Mở rộng đặc trưng bậc hai từ scratch."""
    n_rows, n_features = X_numeric.shape
    output_count = n_features + n_features * (n_features + 1) // 2
    output = np.empty((n_rows, output_count), dtype=np.float64)
    names = feature_names.copy()
    output[:, :n_features] = X_numeric

    column = n_features
    for left in range(n_features):
        for right in range(left, n_features):
            output[:, column] = X_numeric[:, left] * X_numeric[:, right]
            names.append(
                f"{feature_names[left]}^2"
                if left == right
                else f"{feature_names[left]} {feature_names[right]}"
            )
            column += 1
    return output, names


def _fit_poly_elasticnet_cv(
    X_train_poly: np.ndarray,
    y_train: np.ndarray,
    X_test_poly: np.ndarray,
    poly_names: List[str],
) -> FinalModelResult:
    """Manual CV và fit ElasticNet scratch trên polynomial bậc 2."""
    y_log = _to_log_target(y_train)
    alpha, l1_ratio = _choose_scratch_regularization(
        X_train_poly,
        y_train,
        alphas=np.logspace(-4, -0.5, 6).tolist(),
        l1_ratios=[0.5, 0.7, 0.9],
        model_name="Polynomial2_ElasticNet",
    )
    beta = _fit_scratch_regularized_beta(
        X_train_poly, y_log, alpha, l1_ratio
    )
    train_pred = _to_cost(X_train_poly @ beta)
    test_pred = _to_cost(X_test_poly @ beta)
    mae, rmse, r2 = _metrics(y_train, train_pred)
    coef = pd.Series(beta, index=poly_names, name="Polynomial2_ElasticNet")
    return FinalModelResult(
        name="Polynomial2_ElasticNet",
        slug="poly2_elasticnet",
        train_pred=train_pred,
        test_pred=test_pred,
        train_mae=mae,
        train_rmse=rmse,
        train_r2=r2,
        feature_count=X_train_poly.shape[1],
        nonzero_coef=int(np.sum(np.abs(beta[1:]) > 1e-10)),
        detail=(
            f"degree=2; alpha={alpha:.12g}; l1_ratio={l1_ratio:.12g}; "
            "target=log1p; manual_cv=3-fold; solver=scratch_coordinate_descent"
        ),
        coefficients=coef,
    )


def _fit_krr(
    X_train_no_intercept: np.ndarray,
    y_train: np.ndarray,
    X_test_no_intercept: np.ndarray,
) -> Tuple[FinalModelResult, CVResult]:
    """Chọn siêu tham số và fit Kernel Ridge Regression trên mẫu con.

    KRR tạo ma trận kernel kích thước n x n nên chi phí bộ nhớ và thời gian
    tăng theo bình phương số mẫu. Vì vậy CV và fit cuối đều dùng mẫu con cố
    định, nhưng mô hình sau đó vẫn Prediction cho toàn bộ dev_train và test.
    """
    # Giới hạn số hàng để grid search và fit kernel không chiếm quá nhiều bộ nhớ.
    max_cv_rows = 750
    max_fit_rows = 1200

    # Grid nhỏ cho hai siêu tham số quan trọng của Kernel Ridge RBF.
    alpha_grid = [0.1, 1.0, 10.0]
    gamma_grid = [0.0001, 0.001, 0.01]

    # Lấy mẫu con tái lập được để chọn alpha và gamma bằng CV.
    cv_idx = _sample_indices(len(y_train), max_cv_rows, RANDOM_STATE)
    X_cv = X_train_no_intercept[cv_idx]
    y_cv = y_train[cv_idx]

    # Khởi tạo ứng viên tốt nhất bằng phần tử đầu grid.
    best_alpha = alpha_grid[0]
    best_gamma = gamma_grid[0]
    best_cv: Optional[CVResult] = None

    print("\n  Chọn tham số cho KRR trên mẫu con:")

    # Thử mọi tổ hợp alpha/gamma và chọn tổ hợp có CV RMSE thấp nhất.
    for alpha in alpha_grid:
        for gamma in gamma_grid:
            cv = _cross_validate(
                f"KRR(a={alpha:g},g={gamma:g})",
                X_cv,
                y_cv,
                lambda X_tr, y_log_tr, X_val, a=alpha, g=gamma: _predict_krr_cost(
                    X_tr, y_log_tr, X_val, a, g
                ),
                note=(
                    f"3-fold CV tren mau con {len(cv_idx)} rows; "
                    f"alpha={alpha:g}; gamma={gamma:g}"
                ),
                k=3,
                random_state=RANDOM_STATE,
            )

            # Cập nhật siêu tham số tốt nhất sau mỗi tổ hợp.
            if best_cv is None or cv.mean_rmse < best_cv.mean_rmse:
                best_alpha = alpha
                best_gamma = gamma
                best_cv = cv

    # Guard bảo vệ trường hợp grid search không tạo được kết quả.
    if best_cv is None:
        raise RuntimeError("Không chọn được tham số cho KRR.")

    # Lấy mẫu fit cuối lớn hơn mẫu CV để tận dụng thêm dữ liệu trong giới hạn bộ nhớ.
    fit_idx = _sample_indices(len(y_train), max_fit_rows, RANDOM_STATE)

    # KernelRidgeRBF nhận target log1p và fit trên mẫu con đã chọn.
    y_fit_log = _to_log_target(y_train[fit_idx])
    model = KernelRidgeRBF(alpha=best_alpha, gamma=best_gamma)
    model.fit(X_train_no_intercept[fit_idx], y_fit_log)

    # Dùng mô hình kernel đã fit để Prediction toàn bộ dev_train và test.
    train_pred = _to_cost(model.predict_log(X_train_no_intercept))
    test_pred = _to_cost(model.predict_log(X_test_no_intercept))

    # Metric train được tính trên toàn bộ dev_train, không chỉ mẫu con dùng để fit.
    mae, rmse, r2 = _metrics(y_train, train_pred)

    # Chuẩn hóa metadata CV để bảng kết quả chỉ hiển thị một dòng KRR cuối cùng.
    best_cv.model_name = "KernelRidge_RBF"
    best_cv.note = (
        f"3-fold CV tren mau con {len(cv_idx)} rows; "
        f"fit cuoi tren {len(fit_idx)} rows; alpha={best_alpha:g}; gamma={best_gamma:g}"
    )

    # KRR không có coefficient theo feature, nên coefficients được đặt None.
    result = FinalModelResult(
        name="KernelRidge_RBF",
        slug="kernel_ridge_rbf",
        train_pred=train_pred,
        test_pred=test_pred,
        train_mae=mae,
        train_rmse=rmse,
        train_r2=r2,
        feature_count=X_train_no_intercept.shape[1],
        nonzero_coef=len(fit_idx),
        detail=(
            f"alpha={best_alpha:g}; gamma={best_gamma:g}; "
            f"fit_rows={len(fit_idx)}; target=log1p"
        ),
        coefficients=None,
    )
    return result, best_cv


def _predict_krr_cost(
    X_train: np.ndarray,
    y_log: np.ndarray,
    X_pred: np.ndarray,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    """Fit KRR trên một train fold và Prediction X_pred trong đơn vị TZS."""
    # Khởi tạo mô hình bằng cặp alpha/gamma đang được grid search.
    model = KernelRidgeRBF(alpha=alpha, gamma=gamma)

    # Fit trên train fold với target đã được _cross_validate chuyển sang log1p.
    model.fit(X_train, y_log)

    # Prediction validation fold rồi đổi ngược về đơn vị TZS.
    return _to_cost(model.predict_log(X_pred))


def _sample_indices(n_rows: int, max_rows: int, random_state: int) -> np.ndarray:
    """Lấy mẫu con cố định để các mô hình kernel chạy được và tái lập được."""
    # Nếu dữ liệu đã nhỏ hơn giới hạn thì giữ nguyên toàn bộ thứ tự hàng.
    if n_rows <= max_rows:
        return np.arange(n_rows)

    # Generator có seed cố định giúp các lần chạy chọn cùng một tập mẫu.
    rng = np.random.default_rng(random_state)

    # Chọn không hoàn lại và sort chỉ số để giữ thứ tự tương đối của dữ liệu.
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False))


def _fit_ensemble(results: List[FinalModelResult], y_train: np.ndarray) -> FinalModelResult:
    """Tạo ensemble bằng trung bình các mô hình đã fit.

    Ensemble chỉ dùng các mô hình không phải OLS_Selected để tránh một biến thể
    OLS chi phối hai lần. Khi có score website, ta có thể thay trung bình đều
    bằng trọng số theo leaderboard.
    """
    # Chỉ lấy các mô hình đủ đa dạng và có chất lượng cạnh tranh cho ensemble.
    ensemble_members = [
        result for result in results
        if result.slug in {"ridge", "lasso", "elasticnet", "poly2_elasticnet", "kernel_ridge_rbf"}
    ]

    # Trung bình Prediction theo từng hàng giúp giảm phương sai giữa các mô hình.
    train_pred = np.mean([result.train_pred for result in ensemble_members], axis=0)
    test_pred = np.mean([result.test_pred for result in ensemble_members], axis=0)

    # Đánh giá ensemble trên train từ trung bình Prediction của các thành viên.
    mae, rmse, r2 = _metrics(y_train, train_pred)

    # feature_count ở đây biểu thị số mô hình thành viên, không phải số cột dữ liệu.
    return FinalModelResult(
        name="Ensemble_Mean",
        slug="ensemble_mean",
        train_pred=train_pred,
        test_pred=test_pred,
        train_mae=mae,
        train_rmse=rmse,
        train_r2=r2,
        feature_count=len(ensemble_members),
        nonzero_coef=None,
        detail="mean_of=Ridge,Lasso,ElasticNet,Polynomial2_ElasticNet,KernelRidge_RBF",
        coefficients=None,
    )


def _write_outputs(
    output_dir: Path,
    pipe_result: PipelineResult,
    results: List[FinalModelResult],
    cv_results: Dict[str, Optional[CVResult]],
) -> Dict[str, Path]:
    """Ghi bảng so sánh, Prediction tổng hợp và từng file submission riêng."""
    # Tạo thư mục output và submissions nếu chúng chưa tồn tại.
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir = output_dir / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Lấy ID test đúng thứ tự và khởi tạo bảng chứa Prediction mọi mô hình.
    test_ids = _submission_ids(pipe_result)
    prediction_table = pd.DataFrame({"ID": test_ids})

    # Hai danh sách này sẽ được chuyển thành các bảng CSV sau vòng lặp.
    comparison_rows = []
    score_rows = []

    # Mỗi mô hình tạo một cột Prediction tổng hợp và một submission riêng.
    for result in results:
        # Thêm Prediction test vào bảng đối chiếu tất cả mô hình.
        prediction_table[result.slug] = result.test_pred

        # Tên submission dùng slug ổn định, phù hợp để tự động xử lý.
        submission_path = submission_dir / f"submission_{result.slug}.csv"

        # Submission chỉ chứa ID và total_cost không âm theo format Zindi.
        pd.DataFrame(
            {
                "ID": test_ids,
                TARGET_COL: np.clip(result.test_pred, a_min=0.0, a_max=None),
            }
        ).to_csv(submission_path, index=False)

        # Lấy kết quả CV tương ứng để tạo một dòng trong bảng so sánh.
        cv = cv_results.get(result.slug)
        comparison_rows.append(_comparison_row(result, cv, submission_path))

        # Tạo template để người dùng điền score sau khi upload lên website.
        score_rows.append(
            {
                "model": result.name,
                "submission_file": str(submission_path.relative_to(output_dir.parent)),
                "website_score": "",
                "note": "Upload file nay len Zindi roi dien score vao cot website_score.",
            }
        )

    # Ghi bảng metric train/CV và metadata của từng mô hình.
    comparison_path = output_dir / "model_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)

    # Ghi toàn bộ Prediction test cạnh nhau để phân tích hoặc tự tạo ensemble khác.
    predictions_path = output_dir / "all_model_predictions.csv"
    prediction_table.to_csv(predictions_path, index=False)

    # Ghi danh sách submission và cột trống để nhập leaderboard score.
    score_template_path = output_dir / "zindi_score_template.csv"
    pd.DataFrame(score_rows).to_csv(score_template_path, index=False)

    # Ghi top hệ số có độ lớn cao của các mô hình tuyến tính.
    feature_importance_path = output_dir / "feature_importance.csv"
    _write_feature_importance(feature_importance_path, results)

    # Ghi bản tóm tắt dạng text để đọc nhanh mà không cần mở CSV.
    summary_path = output_dir / "analysis_summary.txt"
    _write_summary(summary_path, pipe_result, results, cv_results)

    # Thông báo vị trí output chính sau khi ghi file hoàn tất.
    print("\n" + "-" * 78)
    print("ĐÃ GHI KẾT QUẢ")
    print("-" * 78)
    print(f"  Bảng so sánh       : {comparison_path}")
    print(f"  Prediction tổng hợp   : {predictions_path}")
    print(f"  Mẫu điền điểm số   : {score_template_path}")
    print(f"  Thư mục submission : {submission_dir}")

    # Trả các đường dẫn để code gọi hàm có thể sử dụng tiếp.
    return {
        "comparison": comparison_path,
        "predictions": predictions_path,
        "score_template": score_template_path,
        "submissions": submission_dir,
        "summary": summary_path,
    }


def _submission_ids(pipe_result: PipelineResult) -> np.ndarray:
    """Lấy ID test đúng thứ tự để ghi submission."""
    # Nếu dữ liệu không có ID, tạo ID tuần tự bắt đầu từ 1 làm fallback.
    if pipe_result.test_ids is None:
        return np.arange(1, pipe_result.X_test.shape[0] + 1).astype(str)

    # Đảm bảo ID luôn ở dạng chuỗi để giữ nguyên format khi ghi CSV.
    return pipe_result.test_ids.astype(str)


def _comparison_row(
    result: FinalModelResult,
    cv: Optional[CVResult],
    submission_path: Path,
) -> Dict[str, object]:
    """Tạo một dòng trong bảng so sánh mô hình."""
    # Ensemble không có CV riêng nên các cột CV nhận NaN và note hướng dẫn website.
    return {
        "model": result.name,
        "train_MAE_TZS": result.train_mae,
        "train_RMSE_TZS": result.train_rmse,
        "train_R2": result.train_r2,
        "holdout_MAE_TZS": result.holdout_mae,
        "holdout_RMSE_TZS": result.holdout_rmse,
        "holdout_R2": result.holdout_r2,
        "cv_MAE_TZS_mean": cv.mean_mae if cv else math.nan,
        "cv_MAE_TZS_std": cv.std_mae if cv else math.nan,
        "cv_RMSE_TZS_mean": cv.mean_rmse if cv else math.nan,
        "cv_RMSE_TZS_std": cv.std_rmse if cv else math.nan,
        "cv_R2_mean": cv.mean_r2 if cv else math.nan,
        "cv_R2_std": cv.std_r2 if cv else math.nan,
        "feature_count": result.feature_count,
        "nonzero_coef": result.nonzero_coef,
        "detail": result.detail,
        "cv_note": cv.note if cv else "Ensemble cho score chinh thuc bang website submission.",
        "submission_file": str(submission_path),
        "website_score": "",
    }


def _write_feature_importance(path: Path, results: List[FinalModelResult]) -> None:
    """Ghi top hệ số lớn nhất của các mô hình tuyến tính để phục vụ báo cáo."""
    # Mỗi phần tử rows sẽ trở thành một dòng feature-model trong CSV.
    rows = []

    # Duyệt từng mô hình và bỏ qua KRR/ensemble vì chúng không có coefficient theo feature.
    for result in results:
        if result.coefficients is None:
            continue

        # Chuyển index/value tổng quát của pandas thành cặp (str, float) rõ kiểu.
        ranked_coefficients: List[Tuple[str, float]] = sorted(
            (
                (str(feature), float(coefficient))
                for feature, coefficient in result.coefficients.items()
            ),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:40]

        # Lưu cả hệ số có dấu và trị tuyệt đối để giải thích hướng tác động.
        for feature, coefficient in ranked_coefficients:
            rows.append(
                {
                    "model": result.name,
                    "feature": feature,
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient),
                }
            )

    # Ghi bảng importance ở dạng long-format, thuận tiện lọc theo model.
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_summary(
    path: Path,
    pipe_result: PipelineResult,
    results: List[FinalModelResult],
    cv_results: Dict[str, Optional[CVResult]],
) -> None:
    """Ghi báo cáo txt ngắn để người chạy nắm nhanh output vừa tạo."""
    # Mở file với UTF-8 và ghi đè báo cáo của lần chạy trước.
    with path.open("w", encoding="utf-8") as file:
        # Phần đầu mô tả kích thước dữ liệu và phép biến đổi target.
        file.write("PHAN 2 - THE BEST MODEL\n")
        file.write("=" * 78 + "\n\n")
        file.write(f"Dev train rows: {pipe_result.X_train.shape[0]:,}\n")
        file.write(f"Test rows : {pipe_result.X_test.shape[0]:,}\n")
        file.write(f"Features  : {len(pipe_result.feature_names):,}\n")
        file.write("Target    : log1p(total_cost) khi train, expm1 ve TZS khi submit\n\n")

        # Phần model summary ghi metric train và metric CV quan trọng nhất.
        file.write("MODEL SUMMARY\n")
        file.write("-" * 78 + "\n")

        # Mỗi mô hình chiếm một dòng để báo cáo dễ đối chiếu.
        for result in results:
            # Lấy CV theo slug; ensemble hiện không có CV riêng.
            cv = cv_results.get(result.slug)

            # Chọn nội dung CV hoặc hướng dẫn xem website khi CV không tồn tại.
            cv_text = (
                f"CV R2={cv.mean_r2:.4f}, CV RMSE={cv.mean_rmse:,.2f}"
                if cv else "CV: xem score website"
            )

            # Ghi train metric và CV metric trên cùng một dòng.
            file.write(
                f"{result.name:<26} "
                f"Train R2={result.train_r2:.4f}, "
                f"Train RMSE={result.train_rmse:,.2f}, "
                f"Holdout R2={result.holdout_r2:.4f}, "
                f"Holdout RMSE={result.holdout_rmse:,.2f}, "
                f"{cv_text}\n"
            )

        # Phần cuối nhắc vị trí submission và quy trình nhập score Zindi.
        file.write("\nSUBMISSION\n")
        file.write("-" * 78 + "\n")
        file.write("Moi model da co mot file submission rieng trong outputs/submissions/.\n")
        file.write("Sau khi upload len Zindi, dien score vao outputs/zindi_score_template.csv.\n")


def _print_result_table(
    results: List[FinalModelResult],
    cv_results: Dict[str, Optional[CVResult]],
) -> None:
    """In bảng ngắn ra terminal để kiểm tra nhanh sau khi chạy."""
    print("\n" + "-" * 78)
    print("TÓM TẮT KẾT QUẢ")
    print("-" * 78)
    print(
        f"{'Model':<28} {'Train R2':>10} {'CV R2':>10} "
        f"{'Holdout R2':>12} {'Holdout RMSE':>14}"
    )

    # In một dòng cho mỗi mô hình; ensemble hiển thị website thay cho CV.
    for result in results:
        cv = cv_results.get(result.slug)
        cv_r2 = f"{cv.mean_r2:.4f}" if cv else "website"
        print(
            f"{result.name:<28} {result.train_r2:>10.4f} {cv_r2:>10} "
            f"{result.holdout_r2:>12.4f} {result.holdout_rmse:>14,.2f}"
        )


if __name__ == "__main__":
    run()
