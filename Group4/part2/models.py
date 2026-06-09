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

Module này nhận PipelineResult từ data_pipeline.py và cung cấp API mô hình hóa
độc lập cho OLS, Ridge và Lasso.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

from part1.inference import coef_inference
from part1.ols_implementation import ols_fit
from part1.regularization import lasso_fit, ridge_fit


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
        beta_hat: Vector hệ số ước lượng beta, bao gồm hệ số hệ số tự do ở vị
                trí đầu tiên, dạng list để tương thích với Part 1 modules.
        y_hat: Giá trị Prediction trên tập huấn luyện, hình dạng (n_train,).
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
        test_pred: Vector giá trị Prediction trên test set, hình dạng (n_test,).
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

    Class chứa các thông tin về models trong pipeline, cung cấp interface
    để fit và lưu trữ kết quả của OLS, Ridge và Lasso. 
    
    Mỗi cách xây dựng fit_* đều thực hiện ba việc: 
        (1) huấn luyện mô hình, 
        (2) tính toán các chỉ số trên tập huấn luyện, 
        (3) lưu kết quả vào dictionary self.models để có thể truy xuất sau. 

    Attributes:
        models: Dictionary ánh xạ tên mô hình sang  ModelResult tương
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
    # Hồi quy OLS cơ bản 
    # ========================================================================

    def fit_ols(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> ModelResult:
        """Huấn luyện mô hình OLS và trả về kết quả với đầy đủ chỉ số đánh giá.

        OLS giải bài toán tối thiểu hóa tổng bình phương phần dư (RSS) bằng
        công thức đóng beta_hat = (X^T X)^{-1} X^T y. 

        Args:
            X_train: Ma trận đặc trưng huấn luyện hình dạng (n, p+1), cột đầu
                    tiên phải là cột hệ số tự do (toàn giá trị 1).
            y_train: Vector giá trị mục tiêu total_cost (TZS) hình dạng (n,).
            feature_names: Danh sách tên đặc trưng tùy chọn, dùng khi in báo cáo.

        Returns:
            ModelResult với tên "OLS", vector beta_hat, giá trị Prediction
            và các chỉ số MAE, RMSE, R², Adjusted R² trên tập huấn luyện.
        """
        print(f"\n{'='*70}")
        print("MODEL: Ordinary Least Squares (OLS)")
        print(f"{'='*70}")

        X_list = X_train.tolist()
        y_list = y_train.tolist()

        result = ols_fit(X_list, y_list)
        beta_hat = result.beta_hat
        sigma2_hat = result.sigma2_hat
        y_hat_arr = np.array(result.y_hat)

        n = len(y_train)
        p = X_train.shape[1] - 1  # Số biến thực sự, loại trừ hệ số tự do

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

        Quy trình gồm hai bước: 
            (1) fit OLS đầy đủ trên toàn bộ feature để lấy
                ``p-value`` của từng hệ số qua kiểm định t (β_j / se(β_j) ~ t_{n-p-1}); 
        
        (2) giữ lại chỉ những feature có p-value < p_threshold, rồi refit OLS trên
        tập con đó. 

        Beta trả về có cùng số chiều với X_train (zero-padding tại các vị trí bị
        loại) để evaluate_on_test có thể dùng X_test @ beta_hat.

        Args:
            X_train: Ma trận đặc trưng (n, p+1).
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
        p_full = total_cols - 1  # Số biến thực sự, không tính hệ số tự do

        # ── Bước 1: Fit OLS đầy đủ ───────────────────────────────────────────
        full_result = ols_fit(X_list, y_list)
        beta_full = full_result.beta_hat
        sigma2_hat = full_result.sigma2_hat

        # ── Bước 2: Tính p-value qua coef_inference ──────────────────────────
        inference_result = coef_inference(X_list, y_list, beta_full, sigma2_hat)
        p_values = inference_result.p_values

        # ── Bước 3: Chọn feature theo p-value ────────────────────────────────
        # Index 0 là hệ số tự do — luôn giữ, không đưa vào tiêu chí lọc
        kept = [0] + [j for j in range(1, total_cols) if p_values[j] < p_threshold]

        # Edge case: nếu không có feature nào qua ngưỡng, giữ feature tốt nhất
        if len(kept) == 1:
            best_j = int(np.argmin([p_values[j] for j in range(1, total_cols)])) + 1
            kept = [0, best_j]
            print(  f"  ERROR: Không có feature nào qua ngưỡng — giữ lại feature tốt nhất: "
                    f"{feature_names[best_j] if feature_names else best_j}")

        dropped = [j for j in range(1, total_cols) if j not in kept]
        n_kept = len(kept) - 1  # Trừ hệ số tự do
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

        sel_result = ols_fit(X_sel_list, y_list)
        beta_sel = sel_result.beta_hat
        sigma2_sel = sel_result.sigma2_hat
        y_hat_arr = np.array(sel_result.y_hat)

        # ── Bước 5: Zero-pad beta về đúng kích thước X_train ─────────────────
        beta_padded = [0.0] * total_cols
        for out_pos, col_idx in enumerate(kept):
            beta_padded[col_idx] = beta_sel[out_pos]

        # ── Bước 6: Tính metrics trên tập huấn luyện ─────────────────────────
        rss = float(np.sum((y_train - y_hat_arr) ** 2))
        tss = float(np.sum((y_train - np.mean(y_train)) ** 2))
        r2 = 1 - rss / tss if tss > 0 else 0.0
        adj_r2 =    (1 - (1 - r2) * (n - 1) / (n - n_kept - 1)
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
    # Hồi quy Ridge — thêm penalty L2 để ổn định nghiệm khi có đa cộng tuyến
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
            X_train:    Ma trận đặc trưng huấn luyện hình dạng (n, p+1), cột đầu
                        tiên là hệ số tự do.
            y_train:    Vector giá trị mục tiêu total_cost (TZS) hình dạng (n,).
            lam:        Tham số chuẩn hóa lambda (ký hiệu λ trong lý thuyết), giá trị
                        càng lớn thì các hệ số bị thu hẹp càng mạnh.
            feature_names: Danh sách tên đặc trưng tùy chọn.

        Returns:
            ModelResult với tên "Ridge(λ=...)" và đầy đủ chỉ số đánh giá.
        """
        print(f"\n{'='*70}")
        print(f"MODEL: Ridge Regression (λ = {lam})")
        print(f"{'='*70}")

        # Part 1 yêu cầu input dạng list of lists
        X_list = X_train.tolist()
        y_list = y_train.tolist()

        result = ridge_fit(X_list, y_list, lam)
        beta_hat = result.coefficients
        y_hat = X_train @ np.array(beta_hat)
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

        Lasso tối thiểu hóa RSS + α||β₋₀||₁ và có đặc tính quan trọng là đẩy
        một số hệ số về đúng 0, thực hiện feature selection tự động. Điều này
        đặc biệt có giá trị sau khi one-hot encoding tạo ra hàng trăm biến giả
        từ cột country, purpose... vì Lasso sẽ tự chọn những đặc trưng thực sự
        có đóng góp vào Prediction chi phí du lịch. Lasso không có công thức đóng
        nên phải dùng coordinate descent từ scratch (Part 1 lasso_fit).

        Ký hiệu alpha ở đây dùng nhất quán với Part 1: hàm tối thiểu hóa
        RSS + alpha * ||β₋₀||₁ (không phạt intercept β₀), khác với sklearn
        dùng (1/2n)*RSS + alpha*||β||₁.

        Args:
            X_train: Ma trận đặc trưng huấn luyện hình dạng (n, p+1), cột đầu
                        tiên là intercept.
            y_train: Vector giá trị mục tiêu hình dạng (n,).
            alpha:      Tham số chuẩn hóa λ; giá trị càng lớn thì càng nhiều
                        hệ số bị ép về 0.
            feature_names: Danh sách tên đặc trưng tùy chọn.

        Returns:
            ModelResult với tên "Lasso(α=...)".
        """
        print(f"\n{'='*70}")
        print(f"MODEL: Lasso Regression (α = {alpha})")
        print(f"{'='*70}")

        # Chuyển sang List[List[float]] vì Part 1 lasso_fit nhận Python list.
        X_list = X_train.tolist()
        y_list = y_train.tolist()

        # lasso_fit từ Part 1 dùng coordinate descent scratch, không phạt intercept.
        result = lasso_fit(X_list, y_list, lam=alpha)  # pyright: ignore[reportPossiblyUnboundVariable]
        beta_hat = result.coefficients
        y_hat = X_train @ np.array(beta_hat)
        model_obj = None

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
    # Đánh giá trên tập test và Tạo Prediction phục vụ submission
    # ========================================================================

    def evaluate_on_test(
        self,
        X_test: np.ndarray,
        y_test_actual: Optional[np.ndarray] = None,
        models: Optional[List[ModelResult]] = None
    ) -> Dict[str, ComparisonResult]:
        """Đánh giá các mô hình trên tập kiểm tra hoặc tạo Prediction submission.

        Phương thức này được thiết kế để hoạt động trong cả hai tình huống:
            (1) khi có nhãn y_test (ví dụ: chia train/val thủ công) để tính đầy đủ
        các chỉ số; 
            (2) khi không có nhãn (tình huống thực tế của bộ dữ liệu
        Tanzania) thì chỉ tạo vector Prediction phục vụ submission. 
        
        Prediction được tính bằng phép nhân ma trận X_test @ beta_hat, không cần gọi lại đối
        tượng mô hình gốc, đảm bảo tính thống nhất giữa OLS tự triển khai.

        Args:
            X_test: Ma trận đặc trưng tập kiểm tra hình dạng (m, p+1), đã được
                    chuẩn hóa bằng tham số tính từ train.
            y_test_actual:  Nhãn thực tế của test set hình dạng (m,), thường là
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

            # Prediction thuần bằng phép nhân X_test @ beta_hat, không gọi lại đối
            # tượng mô hình gốc, nhờ đó OLS tự cài và các mô hình sklearn đi chung
            # một lối, tránh sai khác do cách Prediction khác nhau giữa hai nguồn.
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

        Trình bày các chỉ số MAE, RMSE, R² và số hệ số của từng mô hình.

        Args:
            eval_results:   Dictionary kết quả từ evaluate_on_test, ánh xạ tên
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
    
if __name__ == "__main__":
    from part2.data_pipeline import DataPipeline, PipelineConfig

    print("Testing Models Module\n")

    # Load và Preprocessing dữ liệu qua pipeline trước, để có ngay X_train, y_train.
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

    # Tạo Prediction trên test; bộ dữ liệu này không có nhãn nên chỉ tạo prediction
    # chứ chưa tính được sai số thực.
    eval_results = models_obj.evaluate_on_test(X_test, models=[ols_result, ridge_result, lasso_result])

    # Gom kết quả vào một bảng duy nhất để so sánh các mô hình cạnh nhau.
    summary = models_obj.summary_table(eval_results)
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
