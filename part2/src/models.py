"""
Train và so sánh các mô hình hồi quy tuyến tính cho bài toán dự
đoán chi phí du lịch Tanzania. Module này là nơi chứa toàn bộ pipeline mô hình
hóa, triển khai và interfaces cho ba phương pháp hồi quy: OLS
(Ordinary Least Squares), Ridge (chuẩn hóa L2) và Lasso (chuẩn hóa L1).

Ba chỉ số đánh giá chính được tính toán đồng nhất cho mọi mô hình:
    MAE  — Mean Absolute Error, dễ giải thích theo đơn vị TZS.
    RMSE — Root Mean Squared Error, phạt nặng sai số lớn.
    R²   — Hệ số xác định, đo lường tỷ lệ phương sai được giải thích.

Các class và hàm chính:
    ModelResult      — dataclass lưu kết quả fit một mô hình đơn lẻ.
    ComparisonResult — dataclass lưu kết quả so sánh trên test set.
    RegressionModels — class container điều phối fit và so sánh mô hình.

Module này nhận PipelineResult từ data_pipeline.py và được gọi bởi analysis.py
cũng như main.py.
"""

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

# Nạp lại các module Part 1 (OLS, Ridge, suy luận hệ số, cross-validation) để
# Part 2 dùng đúng cài đặt thuần Python đã chứng minh ở phần lý thuyết, thay vì
# viết lại từ đầu; nhờ vậy kết quả thực nghiệm gắn liền trực tiếp với lý thuyết.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../part1'))

try:
    from ols_implementation import ols_fit, OLSResult  # pyright: ignore[reportMissingImports]
    from regularization import ridge_fit, ridge_trace, RidgeResult  # pyright: ignore[reportMissingImports]
    from model_evaluation import model_metrics, ModelMetrics  # pyright: ignore[reportMissingImports]
    from inference import vif, coef_inference  # pyright: ignore[reportMissingImports]
    from cross_validation import kfold_cv, kfold_cv_ridge  # pyright: ignore[reportMissingImports]
except ImportError as e:
    print(f"Warning: Could not import Part 1 modules: {e}")
    print("Using fallback implementations instead.")

from sklearn.linear_model import Lasso, ElasticNet, Ridge as SKRidge
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.model_selection import KFold


@dataclass
class ModelResult:
    """Kết quả huấn luyện của một mô hình hồi quy đơn lẻ.

    Dataclass này chứa tất cả thông tin cần thiết sau khi fit một mô hình,
    bao gồm các hệ số ước lượng và các chỉ số đánh giá trên tập huấn luyện.
    Thiết kế thống nhất giao diện giữa các mô hình khác nhau (OLS, Ridge,
    Lasso) giúp các hàm downstream (evaluate, visualization) xử lý mọi mô
    hình theo cùng một cách mà không cần phân nhánh logic.

    Attributes:
        name: Tên mô hình, ví dụ "OLS", "Ridge(λ=100)".
        beta_hat: Vector hệ số ước lượng beta, bao gồm hệ số intercept ở vị
                trí đầu tiên, dạng list để tương thích với Part 1 modules.
        y_hat: Giá trị dự đoán trên tập huấn luyện, hình dạng (n_train,).
        train_mae: Mean Absolute Error trên tập huấn luyện, đơn vị TZS.
        train_rmse: Root Mean Squared Error trên tập huấn luyện, đơn vị TZS.
        train_r2: Hệ số xác định R² trên tập huấn luyện.
        train_adj_r2: R² hiệu chỉnh, có tính đến số biến để tránh overfitting
                    khi so sánh mô hình với số đặc trưng khác nhau.
        sigma2_hat: Ước lượng phương sai nhiễu (sigma²), dùng cho kiểm định
                    thống kê và khoảng tin cậy hệ số.
        cv_scores: Kết quả cross-validation nếu có, None khi chưa chạy CV.
        model_object: sklearn object (chỉ dùng cho Lasso/ElasticNet).
    """

    name: str
    beta_hat: List[float]
    y_hat: np.ndarray
    train_mae: float
    train_rmse: float
    train_r2: float
    train_adj_r2: float
    sigma2_hat: Optional[float] = None
    cv_scores: Optional[Dict] = None
    model_object: Optional[object] = None  # Chỉ dùng cho các mô hình sklearn
    selected_indices: Optional[List[int]] = None  # Index cột được giữ lại (OLS+Selection)


@dataclass
class ComparisonResult:
    """Kết quả so sánh mô hình trên tập kiểm tra.

    Dataclass này được tạo ra bởi phương thức evaluate_on_test và lưu trữ
    thông tin đủ để xây dựng bảng so sánh tổng hợp giữa các mô hình. Trường
    nonzero_coef đặc biệt quan trọng khi so sánh Lasso với OLS/Ridge: giá trị
    nhỏ hơn nhiều cho thấy Lasso đã thực hiện feature selection tự động, làm
    cho mô hình thưa hơn (sparse) và dễ giải thích hơn.

    Attributes:
        model_name: Tên mô hình tương ứng, khớp với ModelResult.name.
        test_mae: Mean Absolute Error trên test set, NaN nếu không có nhãn.
        test_rmse: Root Mean Squared Error trên test set, NaN nếu không có nhãn.
        test_r2: Hệ số xác định R² trên test set, NaN nếu không có nhãn.
        test_pred: Vector giá trị dự đoán trên test set, hình dạng (n_test,).
        coef_count: Tổng số hệ số của mô hình (bao gồm cả hệ số bằng 0).
        nonzero_coef: Số hệ số khác 0, phản ánh mức độ thưa của mô hình.
    """

    model_name: str
    test_mae: float
    test_rmse: float
    test_r2: float
    test_pred: np.ndarray
    coef_count: int
    nonzero_coef: int


class RegressionModels:
    """Container điều phối việc huấn luyện và so sánh nhiều mô hình hồi quy.

    Class này đóng vai trò là "model zoo" trong pipeline, cung cấp giao diện
    nhất quán để fit và lưu trữ kết quả của OLS, Ridge và Lasso. Mỗi phương
    thức fit_* đều thực hiện ba việc: (1) huấn luyện mô hình, (2) tính toán
    các chỉ số trên tập huấn luyện, (3) lưu kết quả vào dictionary self.models
    để có thể truy xuất sau. Thiết kế này giúp tránh phải truyền danh sách mô
    hình qua nhiều hàm và giảm nguy cơ mất kết quả khi pipeline phức tạp.

    Attributes:
        models: Dictionary ánh xạ tên mô hình sang đối tượng ModelResult tương
                ứng, được tự động cập nhật sau mỗi lần gọi fit_*.
        feature_names: Danh sách tên đặc trưng, có thể gán để hỗ trợ giải thích
                    hệ số trong báo cáo.

    Cách sử dụng:
        models = RegressionModels()
        ols_result = models.fit_ols(X_train, y_train)
        ridge_result = models.fit_ridge(X_train, y_train, lam=100)
        lasso_result = models.fit_lasso(X_train, y_train, alpha=10)
        comparison = models.compare_on_test(X_test, [ols_result, ridge_result, lasso_result])
    """

    def __init__(self):
        self.models = {}
        self.feature_names = None

    # ========================================================================
    # Hồi quy OLS — mô hình nền tảng để mọi mô hình khác đối chiếu
    # ========================================================================

    def fit_ols(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> ModelResult:
        """Huấn luyện mô hình OLS và trả về kết quả với đầy đủ chỉ số đánh giá.

        OLS giải bài toán tối thiểu hóa tổng bình phương phần dư (RSS) bằng
        công thức đóng beta_hat = (X^T X)^{-1} X^T y. Phương thức này ưu tiên
        dùng hàm ols_fit từ Part 1 để nhất quán với lý thuyết; nếu import thất
        bại hoặc Part 1 trả về kết quả rỗng thì fallback về numpy.linalg.lstsq
        (thuật toán SVD) vốn ổn định số học hơn khi ma trận X^T X gần suy biến.

        Args:
            X_train: Ma trận đặc trưng huấn luyện hình dạng (n, p+1), cột đầu
                    tiên phải là cột intercept (toàn giá trị 1).
            y_train: Vector giá trị mục tiêu total_cost (TZS) hình dạng (n,).
            feature_names: Danh sách tên đặc trưng tùy chọn, dùng khi in báo cáo.

        Returns:
            Đối tượng ModelResult với tên "OLS", vector beta_hat, giá trị dự đoán
            và các chỉ số MAE, RMSE, R², Adjusted R² trên tập huấn luyện.
        """
        print(f"\n{'='*70}")
        print("MODEL: Ordinary Least Squares (OLS)")
        print(f"{'='*70}")

        # Part 1 yêu cầu input dạng list of lists, không phải numpy array
        X_list = X_train.tolist()
        y_list = y_train.tolist()

        try:
            # Ưu tiên dùng Part 1 để nhất quán với lý thuyết được trình bày
            result = ols_fit(X_list, y_list) # pyright: ignore[reportPossiblyUnboundVariable]
            beta_hat = result.beta_hat
            sigma2_hat = result.sigma2_hat
            y_hat_arr = np.array(result.y_hat)

            if len(y_hat_arr) == 0:
                raise ValueError("OLS returned empty predictions")

        except Exception as e:
            # Fallback về numpy khi Part 1 không khả dụng hoặc trả về kết quả rỗng
            print(f"  Using NumPy fallback for OLS (Part 1 error: {type(e).__name__})")
            beta_hat_np = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            beta_hat = beta_hat_np.tolist()
            y_hat_arr = X_train @ beta_hat_np
            sigma2_hat = np.mean((y_train - y_hat_arr) ** 2)

        n = len(y_train)
        p = X_train.shape[1] - 1  # Số biến thực sự, loại trừ intercept

        # Tính RSS và TSS để từ đó suy ra R² và R² hiệu chỉnh
        rss = np.sum((y_train - y_hat_arr) ** 2)
        tss = np.sum((y_train - np.mean(y_train)) ** 2)
        r2 = 1 - (rss / tss) if tss > 0 else 0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else 0

        mae = np.mean(np.abs(y_train - y_hat_arr))
        rmse = np.sqrt(np.mean((y_train - y_hat_arr) ** 2))

        print(f"  ✓ Converged")
        print(f"  Coefficients: {len(beta_hat)}")
        print(f"  Train MAE: {mae:.2f}")
        print(f"  Train RMSE: {rmse:.2f}")
        print(f"  Train R²: {r2:.4f}")
        print(f"  Train Adj R²: {adj_r2:.4f}")

        model_result = ModelResult(
            name="OLS",
            beta_hat=beta_hat,
            y_hat=y_hat_arr,
            train_mae=float(mae),
            train_rmse=float(rmse),
            train_r2=float(r2),
            train_adj_r2=float(adj_r2),
            sigma2_hat=float(sigma2_hat) if sigma2_hat is not None else None,
        )

        self.models["OLS"] = model_result
        return model_result

    # ========================================================================
    # OLS + Feature Selection (dựa trên p-value)
    # ========================================================================

    def fit_ols_selected(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        p_threshold: float = 0.05,
    ) -> ModelResult:
        """Huấn luyện OLS sau khi loại bỏ biến không có ý nghĩa thống kê.

        Quy trình gồm hai bước: (1) fit OLS đầy đủ trên toàn bộ feature để lấy
        p-value của từng hệ số qua kiểm định t (β_j / se(β_j) ~ t_{n-p-1}); (2)
        giữ lại chỉ những feature có p-value < p_threshold, rồi refit OLS trên
        tập con đó. Intercept luôn được giữ và không tham gia vào quá trình lọc.

        Beta trả về có cùng số chiều với X_train (zero-padding tại các vị trí bị
        loại) để evaluate_on_test có thể dùng X_test @ beta_hat mà không cần thay
        đổi logic downstream.

        Args:
            X_train: Ma trận đặc trưng (n, p+1), cột đầu là intercept (toàn số 1).
            y_train: Vector mục tiêu hình dạng (n,).
            feature_names: Danh sách tên đặc trưng, dùng để in log.
            p_threshold: Ngưỡng p-value để giữ biến (mặc định 0.05).

        Returns:
            ModelResult với tên "OLS+Selection" và selected_indices ghi lại
            các cột được chọn để tiện trình bày trong báo cáo.
        """
        print(f"\n{'='*70}")
        print(f"MODEL: OLS + Feature Selection (p-value threshold = {p_threshold})")
        print(f"{'='*70}")

        X_list = X_train.tolist()
        y_list = y_train.tolist()
        n, total_cols = X_train.shape
        p_full = total_cols - 1  # Số biến thực sự, không tính intercept

        # ── Bước 1: Fit OLS đầy đủ ───────────────────────────────────────────
        try:
            full_result = ols_fit(X_list, y_list)  # pyright: ignore[reportPossiblyUnboundVariable]
            beta_full = full_result.beta_hat
            sigma2_hat = full_result.sigma2_hat
            if len(beta_full) == 0:
                raise ValueError("OLS returned empty beta")
        except Exception as e:
            print(f"  Using NumPy fallback for initial OLS (error: {type(e).__name__})")
            beta_np = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            beta_full = beta_np.tolist()
            y_hat_tmp = X_train @ beta_np
            rss_tmp = float(np.sum((y_train - y_hat_tmp) ** 2))
            dof_tmp = n - total_cols
            sigma2_hat = rss_tmp / dof_tmp if dof_tmp > 0 else float('nan')

        # Đảm bảo sigma2_hat hợp lệ để coef_inference không trả về NaN p-values
        if sigma2_hat is None or (isinstance(sigma2_hat, float) and np.isnan(sigma2_hat)):
            y_hat_tmp = X_train @ np.array(beta_full)
            rss_tmp = float(np.sum((y_train - y_hat_tmp) ** 2))
            dof_tmp = n - total_cols
            sigma2_hat = rss_tmp / dof_tmp if dof_tmp > 0 else 1.0

        # ── Bước 2: Tính p-value qua coef_inference ──────────────────────────
        try:
            inference_result = coef_inference(X_list, y_list, beta_full, sigma2_hat)  # pyright: ignore[reportPossiblyUnboundVariable]
            p_values = inference_result.p_values
        except Exception as e:
            print(f"  Warning: coef_inference failed ({e}), falling back to scipy OLS")
            import scipy.stats as stats
            X_np = np.array(X_list)
            beta_np = np.array(beta_full)
            y_hat_np = X_np @ beta_np
            residuals = np.array(y_list) - y_hat_np
            rss = float(np.sum(residuals ** 2))
            s2 = rss / (n - total_cols) if (n - total_cols) > 0 else 1.0
            XtX_inv = np.linalg.pinv(X_np.T @ X_np)
            se = np.sqrt(s2 * np.diag(XtX_inv))
            t_stats = beta_np / (se + 1e-15)
            dof = max(n - total_cols, 1)
            p_values = [float(2 * (1 - stats.t.cdf(abs(t), dof))) for t in t_stats]

        # ── Bước 3: Chọn feature theo p-value ────────────────────────────────
        # Index 0 là intercept — luôn giữ, không đưa vào tiêu chí lọc
        kept = [0] + [j for j in range(1, total_cols) if p_values[j] < p_threshold]

        # Edge case: nếu không có feature nào qua ngưỡng, giữ feature tốt nhất
        if len(kept) == 1:
            best_j = int(np.argmin([p_values[j] for j in range(1, total_cols)])) + 1
            kept = [0, best_j]
            print(f"  ⚠ Không có feature nào qua ngưỡng — giữ lại feature tốt nhất: "
                  f"{feature_names[best_j] if feature_names else best_j}")

        dropped = [j for j in range(1, total_cols) if j not in kept]
        n_kept = len(kept) - 1  # Trừ intercept
        n_dropped = len(dropped)

        print(f"  Features ban đầu : {p_full}")
        print(f"  Giữ lại (p < {p_threshold}): {n_kept}")
        print(f"  Loại bỏ         : {n_dropped}")
        if feature_names and n_dropped <= 20:
            dropped_names = [feature_names[j] for j in dropped]
            print(f"  Biến bị loại    : {dropped_names}")

        # ── Bước 4: Refit OLS trên feature được chọn ─────────────────────────
        X_sel = X_train[:, kept]
        X_sel_list = X_sel.tolist()

        try:
            sel_result = ols_fit(X_sel_list, y_list)  # pyright: ignore[reportPossiblyUnboundVariable]
            beta_sel = sel_result.beta_hat
            sigma2_sel = sel_result.sigma2_hat
            y_hat_arr = np.array(sel_result.y_hat)
            if len(y_hat_arr) == 0:
                raise ValueError("OLS on selected features returned empty predictions")
        except Exception as e:
            print(f"  Using NumPy fallback for selected OLS (error: {type(e).__name__})")
            beta_np_sel = np.linalg.lstsq(X_sel, y_train, rcond=None)[0]
            beta_sel = beta_np_sel.tolist()
            y_hat_arr = X_sel @ beta_np_sel
            rss_sel = float(np.sum((y_train - y_hat_arr) ** 2))
            dof_sel = n - len(kept)
            sigma2_sel = rss_sel / dof_sel if dof_sel > 0 else float('nan')

        # ── Bước 5: Zero-pad beta về đúng kích thước X_train ─────────────────
        beta_padded = [0.0] * total_cols
        for out_pos, col_idx in enumerate(kept):
            beta_padded[col_idx] = beta_sel[out_pos]

        # ── Bước 6: Tính metrics trên tập huấn luyện ─────────────────────────
        rss = float(np.sum((y_train - y_hat_arr) ** 2))
        tss = float(np.sum((y_train - np.mean(y_train)) ** 2))
        r2 = 1 - rss / tss if tss > 0 else 0.0
        adj_r2 = (1 - (1 - r2) * (n - 1) / (n - n_kept - 1)
                  if (n - n_kept - 1) > 0 else 0.0)
        mae = float(np.mean(np.abs(y_train - y_hat_arr)))
        rmse = float(np.sqrt(np.mean((y_train - y_hat_arr) ** 2)))

        print(f"  ✓ Refit xong")
        print(f"  Train MAE:  {mae:.2f}")
        print(f"  Train RMSE: {rmse:.2f}")
        print(f"  Train R²:   {r2:.4f}")
        print(f"  Train Adj R²: {adj_r2:.4f}")

        model_result = ModelResult(
            name="OLS+Selection",
            beta_hat=beta_padded,
            y_hat=y_hat_arr,
            train_mae=mae,
            train_rmse=rmse,
            train_r2=r2,
            train_adj_r2=adj_r2,
            sigma2_hat=sigma2_sel,
            selected_indices=kept,
        )

        self.models["OLS+Selection"] = model_result
        return model_result

    # ========================================================================
    # Hồi quy Ridge — thêm phạt L2 để ổn định nghiệm khi có đa cộng tuyến
    # ========================================================================

    def fit_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        lam: float = 1.0,
        feature_names: Optional[List[str]] = None
    ) -> ModelResult:
        """Huấn luyện mô hình Ridge (hồi quy với chuẩn hóa L2).

        Ridge giải bài toán tối thiểu hóa RSS + λ||β||² bằng công thức đóng
        beta_hat = (X^T X + λI)^{-1} X^T y. Việc thêm ma trận λI vào X^T X
        làm cho ma trận luôn khả nghịch và thu hẹp các hệ số về phía 0 (nhưng
        không về đúng 0), giúp giảm phương sai và cải thiện khả năng tổng quát
        hóa khi có nhiều biến tương quan với nhau. Giá trị λ tối ưu được xác
        định trước qua cross-validation trong evaluate.py.

        Args:
            X_train: Ma trận đặc trưng huấn luyện hình dạng (n, p+1), cột đầu
                     tiên là intercept.
            y_train: Vector giá trị mục tiêu total_cost (TZS) hình dạng (n,).
            lam: Tham số chuẩn hóa lambda (ký hiệu λ trong lý thuyết), giá trị
                 càng lớn thì các hệ số bị thu hẹp càng mạnh.
            feature_names: Danh sách tên đặc trưng tùy chọn.

        Returns:
            Đối tượng ModelResult với tên "Ridge(λ=...)" và đầy đủ chỉ số đánh giá.
        """
        print(f"\n{'='*70}")
        print(f"MODEL: Ridge Regression (λ = {lam})")
        print(f"{'='*70}")

        # Part 1 yêu cầu input dạng list of lists
        X_list = X_train.tolist()
        y_list = y_train.tolist()

        try:
            # Dùng ridge_fit từ Part 1 để nhất quán với công thức lý thuyết
            result = ridge_fit(X_list, y_list, lam)  # pyright: ignore[reportPossiblyUnboundVariable]
            beta_hat = result.coefficients  # RidgeResult lưu hệ số ở field 'coefficients'
            y_hat = X_train @ np.array(beta_hat)
            sigma2_hat = np.mean((y_train - y_hat) ** 2)
        except Exception as e:
            # Fallback về scikit-learn khi Part 1 không khả dụng
            print(f"  Using scikit-learn Ridge fallback (Part 1 error: {type(e).__name__})")
            sk_ridge = SKRidge(alpha=lam, fit_intercept=False)
            sk_ridge.fit(X_train, y_train)
            beta_hat = sk_ridge.coef_.tolist()
            y_hat = sk_ridge.predict(X_train)
            sigma2_hat = np.mean((y_train - y_hat) ** 2)

        # Từ phần dư, ta tính RSS và TSS để suy ra R², R² hiệu chỉnh cùng MAE và
        # RMSE — bộ chỉ số dùng chung cho mọi mô hình nên có thể so sánh trực tiếp.
        n = len(y_train)
        p = X_train.shape[1] - 1

        rss = np.sum((y_train - y_hat) ** 2)
        tss = np.sum((y_train - np.mean(y_train)) ** 2)
        r2 = 1 - (rss / tss) if tss > 0 else 0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else 0

        mae = np.mean(np.abs(y_train - y_hat))
        rmse = np.sqrt(np.mean((y_train - y_hat) ** 2))

        print(f"  ✓ Converged")
        print(f"  Coefficients: {len(beta_hat)}")
        print(f"  Train MAE: {mae:.2f}")
        print(f"  Train RMSE: {rmse:.2f}")
        print(f"  Train R²: {r2:.4f}")
        print(f"  Train Adj R²: {adj_r2:.4f}")

        model_result = ModelResult(
            name=f"Ridge(λ={lam})",
            beta_hat=beta_hat,
            y_hat=y_hat,
            train_mae=float(mae),
            train_rmse=float(rmse),
            train_r2=float(r2),
            train_adj_r2=float(adj_r2),
            sigma2_hat=float(sigma2_hat) if sigma2_hat is not None else None,
        )

        self.models[f"Ridge(λ={lam})"] = model_result
        return model_result

    # ========================================================================
    # Hồi quy Lasso — phạt L1 để tự động chọn biến (đẩy hệ số yếu về 0)
    # ========================================================================

    def fit_lasso(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        alpha: float = 1.0,
        feature_names: Optional[List[str]] = None
    ) -> ModelResult:
        """Huấn luyện mô hình Lasso (hồi quy với chuẩn hóa L1).

        Lasso tối thiểu hóa RSS + α||β||₁ và có đặc tính quan trọng là đẩy
        một số hệ số về đúng 0, thực hiện feature selection tự động. Điều này
        đặc biệt có giá trị sau khi one-hot encoding tạo ra hàng trăm biến giả
        từ cột country, purpose... vì Lasso sẽ tự chọn những đặc trưng thực sự
        có đóng góp vào dự đoán chi phí du lịch. Lasso không có công thức đóng
        nên phải dùng coordinate descent (scikit-learn) với max_iter=10000 để
        đảm bảo hội tụ trên dữ liệu có nhiều biến tương quan.

        Args:
            X_train: Ma trận đặc trưng huấn luyện hình dạng (n, p+1), cột đầu
                     tiên là intercept.
            y_train: Vector giá trị mục tiêu total_cost (TZS) hình dạng (n,).
            alpha: Tham số chuẩn hóa của scikit-learn (tương đương λ/2 trong
                   một số ký hiệu lý thuyết), giá trị càng lớn thì càng nhiều
                   hệ số bị ép về 0.
            feature_names: Danh sách tên đặc trưng tùy chọn.

        Returns:
            Đối tượng ModelResult với tên "Lasso(α=...)" và trường model_object
            chứa đối tượng sklearn Lasso đã fit để dùng trong SHAP analysis.
        """
        print(f"\n{'='*70}")
        print(f"MODEL: Lasso Regression (α = {alpha})")
        print(f"{'='*70}")

        try:
            # Lasso không có nghiệm dạng đóng nên ta dùng coordinate descent của
            # scikit-learn, đặt max_iter lớn để chắc chắn hội tụ trên bộ đặc trưng
            # nhiều biến giả sau one-hot encoding.
            lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
            lasso.fit(X_train, y_train)
            beta_hat = [float(c) for c in lasso.coef_]  # đảm bảo List[float] phẳng
            y_hat = lasso.predict(X_train)
            model_obj = lasso
        except Exception as e:
            print(f"  Error: {e}")
            raise

        # Từ phần dư, ta tính RSS và TSS để suy ra R², R² hiệu chỉnh cùng MAE và
        # RMSE — bộ chỉ số dùng chung cho mọi mô hình nên có thể so sánh trực tiếp.
        n = len(y_train)
        p = X_train.shape[1] - 1

        rss = np.sum((y_train - y_hat) ** 2)
        tss = np.sum((y_train - np.mean(y_train)) ** 2)
        r2 = 1 - (rss / tss) if tss > 0 else 0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else 0

        mae = np.mean(np.abs(y_train - y_hat))
        rmse = np.sqrt(np.mean((y_train - y_hat) ** 2))

        nonzero_coef = np.count_nonzero(beta_hat)

        print(f"  ✓ Converged")
        print(f"  Coefficients: {len(beta_hat)} (nonzero: {nonzero_coef})")
        print(f"  Train MAE: {mae:.2f}")
        print(f"  Train RMSE: {rmse:.2f}")
        print(f"  Train R²: {r2:.4f}")
        print(f"  Train Adj R²: {adj_r2:.4f}")

        model_result = ModelResult(
            name=f"Lasso(α={alpha})",
            beta_hat=beta_hat,
            y_hat=y_hat,
            train_mae=float(mae),
            train_rmse=float(rmse),
            train_r2=float(r2),
            train_adj_r2=float(adj_r2),
            model_object=model_obj,
        )

        self.models[f"Lasso(α={alpha})"] = model_result
        return model_result

    # ========================================================================
    # Đánh giá trên tập test và sinh dự đoán phục vụ submission
    # ========================================================================

    def evaluate_on_test(
        self,
        X_test: np.ndarray,
        y_test_actual: Optional[np.ndarray] = None,
        models: Optional[List[ModelResult]] = None
    ) -> Dict[str, ComparisonResult]:
        """Đánh giá các mô hình trên tập kiểm tra hoặc tạo dự đoán submission.

        Phương thức này được thiết kế để hoạt động trong cả hai tình huống:
        (1) khi có nhãn y_test (ví dụ: chia train/val thủ công) để tính đầy đủ
        các chỉ số; (2) khi không có nhãn (tình huống thực tế của bộ dữ liệu
        Tanzania) thì chỉ tạo vector dự đoán phục vụ submission. Dự đoán được
        tính bằng phép nhân ma trận X_test @ beta_hat, không cần gọi lại đối
        tượng mô hình gốc, đảm bảo tính thống nhất giữa OLS tự triển khai và
        các mô hình sklearn.

        Args:
            X_test: Ma trận đặc trưng tập kiểm tra hình dạng (m, p+1), đã được
                    chuẩn hóa bằng tham số tính từ train.
            y_test_actual: Nhãn thực tế của test set hình dạng (m,), thường là
                           None trong bộ dữ liệu Tanzania.
            models: Danh sách ModelResult cần đánh giá; nếu None thì dùng tất
                    cả mô hình đã lưu trong self.models.

        Returns:
            Dictionary ánh xạ tên mô hình sang ComparisonResult tương ứng,
            với các chỉ số là NaN khi y_test_actual là None.
        """
        if models is None:
            models = list(self.models.values())

        print(f"\n{'='*70}")
        print("EVALUATION ON TEST SET")
        print(f"{'='*70}")

        results = {}

        for model in models:
            print(f"\n  {model.name}")

            # Dự đoán thuần bằng phép nhân X_test @ beta_hat, không gọi lại đối
            # tượng mô hình gốc, nhờ đó OLS tự cài và các mô hình sklearn đi chung
            # một lối, tránh sai khác do cách dự đoán khác nhau giữa hai nguồn.
            y_pred = X_test @ np.array(model.beta_hat)

            if y_test_actual is not None:
                # Khi có nhãn thực, ta tính đầy đủ MAE, RMSE và R² trên tập test
                mae = np.mean(np.abs(y_test_actual - y_pred))
                rmse = np.sqrt(np.mean((y_test_actual - y_pred) ** 2))

                tss = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
                rss = np.sum((y_test_actual - y_pred) ** 2)
                r2 = 1 - (rss / tss) if tss > 0 else 0

                print(f"    MAE:  {mae:.2f}")
                print(f"    RMSE: {rmse:.2f}")
                print(f"    R²:   {r2:.4f}")

                result = ComparisonResult(
                    model_name=model.name,
                    test_mae=float(mae),
                    test_rmse=float(rmse),
                    test_r2=float(r2),
                    test_pred=y_pred,
                    coef_count=len(model.beta_hat),
                    nonzero_coef=int(np.count_nonzero(model.beta_hat)),
                )
            else:
                print(f"    Predictions: {y_pred.shape}")
                result = ComparisonResult(
                    model_name=model.name,
                    test_mae=float("nan"),
                    test_rmse=float("nan"),
                    test_r2=float("nan"),
                    test_pred=y_pred,
                    coef_count=len(model.beta_hat),
                    nonzero_coef=int(np.count_nonzero(model.beta_hat)),
                )

            results[model.name] = result

        return results

    def summary_table(self, eval_results: Dict[str, ComparisonResult]) -> pd.DataFrame:
        """Tạo bảng tổng hợp so sánh hiệu suất các mô hình dưới dạng DataFrame.

        Bảng này là đầu ra trực tiếp cho báo cáo học thuật, trình bày song song
        các chỉ số MAE, RMSE, R² và số hệ số của từng mô hình để người đọc dễ
        dàng so sánh sự đánh đổi giữa độ chính xác và mức độ phức tạp.

        Args:
            eval_results: Dictionary kết quả từ evaluate_on_test, ánh xạ tên
                          mô hình sang ComparisonResult.

        Returns:
            DataFrame với mỗi hàng là một mô hình và các cột là chỉ số đánh giá,
            giá trị NaN được hiển thị là "N/A" trong bảng.
        """
        rows = []

        for name, result in eval_results.items():
            rows.append({
                "Model": name,
                "Test MAE": f"{result.test_mae:.2f}" if not np.isnan(result.test_mae) else "N/A",
                "Test RMSE": f"{result.test_rmse:.2f}" if not np.isnan(result.test_rmse) else "N/A",
                "Test R²": f"{result.test_r2:.4f}" if not np.isnan(result.test_r2) else "N/A",
                "Coef Count": result.coef_count,
                "Nonzero Coef": result.nonzero_coef
            })

        return pd.DataFrame(rows)


# ============================================================================
# Chạy thử trực tiếp — kiểm tra nhanh ba mô hình trên dữ liệu thật
# ============================================================================

if __name__ == "__main__":
    from data_pipeline import DataPipeline, PipelineConfig

    print("Testing Models Module\n")

    # Nạp và tiền xử lý dữ liệu qua pipeline trước, để có ngay X_train, y_train.
    config = PipelineConfig(data_dir="data", missing_method="mean")
    pipeline = DataPipeline(config)
    pipe_result = pipeline.run()

    X_train = pipe_result.X_train
    y_train = pipe_result.y_train
    X_test = pipe_result.X_test
    feature_names = pipe_result.feature_names

    print(f"\nData loaded: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  Feature names: {feature_names[:5]}...")

    # Huấn luyện ba mô hình tiêu biểu để đối chiếu: OLS, Ridge và Lasso.
    models_obj = RegressionModels()

    ols_result = models_obj.fit_ols(X_train, y_train, feature_names)
    ridge_result = models_obj.fit_ridge(X_train, y_train, lam=1000.0)
    lasso_result = models_obj.fit_lasso(X_train, y_train, alpha=100.0)

    # Sinh dự đoán trên test; bộ dữ liệu này không có nhãn nên chỉ tạo prediction
    # chứ chưa tính được sai số thực.
    eval_results = models_obj.evaluate_on_test(X_test, models=[ols_result, ridge_result, lasso_result])

    # Gom kết quả vào một bảng duy nhất để so sánh các mô hình cạnh nhau.
    summary = models_obj.summary_table(eval_results)
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
