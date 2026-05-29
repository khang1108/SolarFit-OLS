"""
Module huấn luyện và so sánh các mô hình hồi quy tuyến tính cho bài toán dự
đoán chi phí du lịch Tanzania. Module này là trung tâm của pipeline mô hình
hóa, triển khai và thống nhất giao diện cho ba phương pháp hồi quy: OLS
(Ordinary Least Squares), Ridge (chuẩn hóa L2) và Lasso (chuẩn hóa L1).

Thiết kế quan trọng: module ưu tiên dùng các hàm tự triển khai từ Part 1
(ols_fit, ridge_fit) để đảm bảo tính nhất quán với lý thuyết được trình bày
trong báo cáo. Trong trường hợp import Part 1 thất bại, module tự động
fallback về scikit-learn mà không làm gián đoạn pipeline, đảm bảo khả năng
chạy trên nhiều môi trường khác nhau.

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

# Import Part 1 modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../part1'))

try:
    from ols_implementation import ols_fit, OLSResult
    from regularization import ridge_fit, ridge_trace, RidgeResult
    from model_evaluation import model_metrics, ModelMetrics
    from inference import vif, coef_inference
    from cross_validation import kfold_cv, kfold_cv_ridge
except ImportError as e:
    print(f"Warning: Could not import Part 1 modules: {e}")
    print("Using fallback implementations instead.")

from sklearn.linear_model import Lasso, ElasticNet, Ridge as SKRidge
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.model_selection import KFold


@dataclass
class ModelResult:
    """Kết quả huấn luyện của một mô hình hồi quy đơn lẻ.

    Dataclass này đóng gói tất cả thông tin cần thiết sau khi fit một mô hình,
    bao gồm các hệ số ước lượng và các chỉ số đánh giá trên tập huấn luyện.
    Thiết kế thống nhất giao diện giữa các mô hình khác nhau (OLS, Ridge,
    Lasso) giúp các hàm downstream (evaluate, visualization) xử lý mọi mô
    hình theo cùng một cách mà không cần phân nhánh logic.

    Attributes:
        name: Tên định danh mô hình, ví dụ "OLS", "Ridge(λ=100)".
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
        model_object: Đối tượng mô hình sklearn (chỉ dùng cho Lasso/ElasticNet
                      vốn không có implementation trong Part 1).
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
    # OLS Regression
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
            result = ols_fit(X_list, y_list)
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
            train_mae=mae,
            train_rmse=rmse,
            train_r2=r2,
            train_adj_r2=adj_r2,
            sigma2_hat=sigma2_hat
        )

        self.models["OLS"] = model_result
        return model_result

    # ========================================================================
    # Ridge Regression
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
            result = ridge_fit(X_list, y_list, lam)
            beta_hat = result.beta_hat
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

        # Calculate metrics
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
            train_mae=mae,
            train_rmse=rmse,
            train_r2=r2,
            train_adj_r2=adj_r2,
            sigma2_hat=sigma2_hat
        )

        self.models[f"Ridge(λ={lam})"] = model_result
        return model_result

    # ========================================================================
    # Lasso Regression
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
            # Use scikit-learn Lasso
            lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
            lasso.fit(X_train, y_train)
            beta_hat = lasso.coef_.tolist()
            y_hat = lasso.predict(X_train)
            model_obj = lasso
        except Exception as e:
            print(f"  Error: {e}")
            raise

        # Calculate metrics
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
            train_mae=mae,
            train_rmse=rmse,
            train_r2=r2,
            train_adj_r2=adj_r2,
            model_object=model_obj
        )

        self.models[f"Lasso(α={alpha})"] = model_result
        return model_result

    # ========================================================================
    # Model Evaluation on Test Set
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

            # Make predictions
            y_pred = X_test @ np.array(model.beta_hat)

            if y_test_actual is not None:
                # Calculate metrics
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
                    test_mae=mae,
                    test_rmse=rmse,
                    test_r2=r2,
                    test_pred=y_pred,
                    coef_count=len(model.beta_hat),
                    nonzero_coef=np.count_nonzero(model.beta_hat)
                )
            else:
                print(f"    Predictions: {y_pred.shape}")
                result = ComparisonResult(
                    model_name=model.name,
                    test_mae=np.nan,
                    test_rmse=np.nan,
                    test_r2=np.nan,
                    test_pred=y_pred,
                    coef_count=len(model.beta_hat),
                    nonzero_coef=np.count_nonzero(model.beta_hat)
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
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from data_pipeline import DataPipeline, PipelineConfig

    print("Testing Models Module\n")

    # Load data
    config = PipelineConfig(data_dir="data", missing_method="mean")
    pipeline = DataPipeline(config)
    pipe_result = pipeline.run()

    X_train = pipe_result.X_train
    y_train = pipe_result.y_train
    X_test = pipe_result.X_test
    feature_names = pipe_result.feature_names

    print(f"\nData loaded: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  Feature names: {feature_names[:5]}...")

    # Fit models
    models_obj = RegressionModels()

    ols_result = models_obj.fit_ols(X_train, y_train, feature_names)
    ridge_result = models_obj.fit_ridge(X_train, y_train, lam=1000.0)
    lasso_result = models_obj.fit_lasso(X_train, y_train, alpha=100.0)

    # Evaluate on test set (note: we don't have y_test, so just make predictions)
    eval_results = models_obj.evaluate_on_test(X_test, models=[ols_result, ridge_result, lasso_result])

    # Create summary table
    summary = models_obj.summary_table(eval_results)
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
