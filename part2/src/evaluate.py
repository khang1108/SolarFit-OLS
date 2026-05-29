"""
Module đánh giá mô hình hồi quy thông qua cross-validation và phân tích phần
dư (residual diagnostics) cho bài toán dự đoán chi phí du lịch Tanzania.

Module này giải quyết một vấn đề căn bản trong học máy: làm thế nào để ước
lượng hiệu suất thực sự của mô hình khi không có tập test có nhãn. Giải pháp
là k-fold cross-validation, trong đó tập huấn luyện được chia thành k tập con;
mô hình được huấn luyện trên k-1 tập và đánh giá trên tập còn lại, lặp lại k
lần và lấy trung bình. Quy trình này cũng được dùng để chọn siêu tham số λ tối
ưu cho Ridge và α cho Lasso mà không cần chạm vào test set.

Ngoài cross-validation, module cung cấp phân tích phần dư để kiểm định các giả
thuyết Gauss-Markov: phần dư có phân phối chuẩn (Shapiro-Wilk), không có tự
tương quan (autocorrelation), và phân tán đều (homoscedasticity). Những kiểm
định này là cơ sở để khẳng định giá trị của ước lượng hệ số OLS.

Các class và hàm chính:
  CrossValidationResult    — dataclass lưu kết quả CV theo từng fold.
  FeatureImportanceResult  — dataclass lưu xếp hạng tầm quan trọng đặc trưng.
  ModelEvaluator           — class tĩnh tập hợp các hàm đánh giá.
  _shapiro_test            — hàm kiểm định chuẩn tắc phần dư.

Module này được gọi bởi analysis.py và main.py sau khi huấn luyện mô hình xong.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class CrossValidationResult:
    """Kết quả k-fold cross-validation cho một mô hình hồi quy.

    Dataclass này lưu trữ cả kết quả chi tiết theo từng fold lẫn thống kê tổng
    hợp (mean ± std). Việc lưu riêng từng fold giúp phát hiện các fold bất
    thường (ví dụ: fold có R² âm do phân phối target lệch), còn mean ± std là
    chỉ số cuối cùng để so sánh mô hình trong báo cáo.

    Attributes:
        model_name: Tên định danh mô hình, ví dụ "OLS" hay "Ridge(λ=100)".
        fold_mae: Danh sách MAE của từng fold, độ dài k.
        fold_rmse: Danh sách RMSE của từng fold, độ dài k.
        fold_r2: Danh sách R² của từng fold, độ dài k.
        mean_mae: Trung bình MAE qua k fold, đơn vị TZS.
        std_mae: Độ lệch chuẩn MAE, phản ánh độ ổn định của mô hình.
        mean_rmse: Trung bình RMSE qua k fold, đơn vị TZS.
        std_rmse: Độ lệch chuẩn RMSE.
        mean_r2: Trung bình R² qua k fold.
        std_r2: Độ lệch chuẩn R².
    """

    model_name: str
    fold_mae: List[float]
    fold_rmse: List[float]
    fold_r2: List[float]
    mean_mae: float
    std_mae: float
    mean_rmse: float
    std_rmse: float
    mean_r2: float
    std_r2: float


@dataclass
class FeatureImportanceResult:
    """Kết quả xếp hạng tầm quan trọng đặc trưng dựa trên giá trị tuyệt đối hệ số.

    Xếp hạng theo |β_j| là cách đơn giản và trực quan nhất để giải thích tầm
    quan trọng tương đối của các biến trong mô hình tuyến tính đã được chuẩn
    hóa. Kết quả này bổ sung cho SHAP analysis trong shap_analysis.py, cho phép
    so sánh hai cách tiếp cận khác nhau về feature importance.

    Attributes:
        feature_names: Danh sách tên tất cả đặc trưng, bao gồm intercept.
        coefficients: Vector hệ số gốc (có dấu), hình dạng (p+1,).
        abs_coefficients: Vector giá trị tuyệt đối hệ số, dùng để xếp hạng.
        ranking: Danh sách tuple (tên_đặc_trưng, |hệ_số|) đã sắp xếp giảm dần,
                 giới hạn top_n phần tử theo tham số được truyền vào.
    """

    feature_names: List[str]
    coefficients: np.ndarray
    abs_coefficients: np.ndarray
    ranking: List[Tuple[str, float]]  # (tên_đặc_trưng, |hệ_số|)


class ModelEvaluator:
    """Tập hợp các công cụ đánh giá toàn diện cho mô hình hồi quy tuyến tính.

    Class này sử dụng các phương thức tĩnh (staticmethod) vì không cần lưu trữ
    trạng thái giữa các lần gọi; mỗi hàm độc lập nhận dữ liệu vào và trả về
    kết quả ngay. Thiết kế này tránh sự phụ thuộc thứ tự và dễ dàng kiểm thử
    từng thành phần riêng lẻ.

    Bốn nhóm chức năng chính: (1) chia tập dữ liệu (train_val_split, kfold_split),
    (2) cross-validation đầy đủ (kfold_cv), (3) phân tích tầm quan trọng đặc
    trưng (feature_importance), (4) kiểm định phần dư và chọn siêu tham số
    (residual_diagnostics, hyperparameter_tuning).

    Cách sử dụng:
        cv_result = ModelEvaluator.kfold_cv(X_train, y_train, k=5, fit_func=fit_ols)
        fi_result = ModelEvaluator.feature_importance(feature_names, coef)
    """

    @staticmethod
    def train_val_split(
        X: np.ndarray,
        y: np.ndarray,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Chia dữ liệu thành tập huấn luyện và tập validation ngẫu nhiên.

        Hàm này thực hiện phép chia hold-out đơn giản, hữu ích khi cần kiểm
        tra nhanh hiệu suất mô hình trước khi chạy k-fold cross-validation đầy
        đủ tốn thời gian hơn. Việc xáo trộn chỉ số trước khi chia đảm bảo tập
        validation không bị thiên lệch theo thứ tự thu thập dữ liệu gốc.

        Args:
            X: Ma trận đặc trưng hình dạng (n, p), trong ngữ cảnh Tanzania
               thường là X_train từ PipelineResult.
            y: Vector giá trị mục tiêu hình dạng (n,).
            val_size: Tỷ lệ dữ liệu dành cho tập validation, mặc định 0.2
                      (20% làm validation, 80% làm train).
            random_state: Hạt giống ngẫu nhiên để tái tạo kết quả.

        Returns:
            Bộ bốn ma trận (X_train, X_val, y_train, y_val).
        """
        np.random.seed(random_state)
        n = X.shape[0]
        val_size_n = int(n * val_size)

        indices = np.arange(n)
        np.random.shuffle(indices)

        train_idx = indices[val_size_n:]
        val_idx = indices[:val_size_n]

        return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

    @staticmethod
    def kfold_split(
        n: int,
        k: int = 5,
        random_state: int = 42
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Tạo danh sách k cặp chỉ số (train_idx, test_idx) cho k-fold CV.

        Hàm này tách riêng logic tạo fold khỏi logic huấn luyện để có thể tái
        sử dụng cùng một phép chia dữ liệu cho nhiều mô hình khác nhau, đảm
        bảo so sánh công bằng. Fold cuối cùng (i = k-1) nhận toàn bộ mẫu còn
        lại để xử lý trường hợp n không chia hết cho k.

        Args:
            n: Tổng số mẫu quan sát trong tập dữ liệu.
            k: Số lượng fold, mặc định k=5 theo chuẩn thông thường trong học máy.
            random_state: Hạt giống ngẫu nhiên để tái tạo phép chia.

        Returns:
            Danh sách k tuple (train_indices, test_indices) là các mảng chỉ số
            numpy, mỗi phần tử tương ứng với một fold.
        """
        np.random.seed(random_state)
        fold_size = n // k
        indices = np.arange(n)
        np.random.shuffle(indices)

        folds = []
        for i in range(k):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < k - 1 else n

            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

            folds.append((train_idx, test_idx))

        return folds

    @staticmethod
    def kfold_cv(
        X: np.ndarray,
        y: np.ndarray,
        fit_func: Callable,
        k: int = 5,
        model_name: str = "Model",
        random_state: int = 42
    ) -> CrossValidationResult:
        """Thực hiện k-fold cross-validation và trả về chỉ số đánh giá tổng hợp.

        Hàm nhận vào một hàm fit_func có chữ ký (X_train, y_train) -> beta và
        tự động lặp qua k fold, huấn luyện và đánh giá mô hình trên từng fold.
        Thiết kế dạng higher-order function (nhận hàm làm tham số) giúp tách
        biệt logic cross-validation khỏi chi tiết triển khai mô hình, cho phép
        dùng chung với OLS, Ridge hay bất kỳ hàm hồi quy nào trả về beta.

        Args:
            X: Ma trận đặc trưng hình dạng (n, p+1) bao gồm cột intercept.
            y: Vector giá trị mục tiêu total_cost hình dạng (n,).
            fit_func: Hàm nhận (X_train, y_train) và trả về vector beta dạng
                      numpy array, được dùng để dự đoán qua phép nhân X_test @ beta.
            k: Số lượng fold, mặc định k=5.
            model_name: Tên mô hình dùng trong báo cáo và log.
            random_state: Hạt giống ngẫu nhiên.

        Returns:
            CrossValidationResult với MAE, RMSE, R² theo từng fold và tổng hợp.
        """
        folds = ModelEvaluator.kfold_split(X.shape[0], k, random_state)

        fold_mae = []
        fold_rmse = []
        fold_r2 = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit and predict
            beta = fit_func(X_train, y_train)
            y_pred = X_test @ beta

            # Calculate metrics
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            tss = np.sum((y_test - np.mean(y_test)) ** 2)
            rss = np.sum((y_test - y_pred) ** 2)
            r2 = 1 - (rss / tss) if tss > 0 else 0

            fold_mae.append(mae)
            fold_rmse.append(rmse)
            fold_r2.append(r2)

        return CrossValidationResult(
            model_name=model_name,
            fold_mae=fold_mae,
            fold_rmse=fold_rmse,
            fold_r2=fold_r2,
            mean_mae=np.mean(fold_mae),
            std_mae=np.std(fold_mae),
            mean_rmse=np.mean(fold_rmse),
            std_rmse=np.std(fold_rmse),
            mean_r2=np.mean(fold_r2),
            std_r2=np.std(fold_r2)
        )

    @staticmethod
    def feature_importance(
        feature_names: List[str],
        coefficients: np.ndarray,
        top_n: int = 20
    ) -> FeatureImportanceResult:
        """Xếp hạng đặc trưng theo giá trị tuyệt đối của hệ số hồi quy.

        Phương pháp này chỉ có ý nghĩa khi tất cả các biến đã được chuẩn hóa
        (StandardScaler) trước khi đưa vào mô hình, tức là đơn vị đo lường của
        mọi biến đã về cùng một thang — một điều kiện đã được đảm bảo trong
        DataPipeline. Khi đó |β_j| lớn hơn có nghĩa là biến j có ảnh hưởng lớn
        hơn đến giá trị dự đoán, không phụ thuộc vào đơn vị đo lường gốc.

        Args:
            feature_names: Danh sách tên đặc trưng tương ứng với từng hệ số,
                           bao gồm "intercept" ở vị trí đầu.
            coefficients: Vector hệ số beta_hat hình dạng (p+1,), có thể là
                          list (từ ModelResult) hoặc numpy array.
            top_n: Số lượng đặc trưng quan trọng nhất cần trả về.

        Returns:
            FeatureImportanceResult với danh sách xếp hạng giảm dần theo |β_j|.
        """
        abs_coef = np.abs(coefficients)
        ranking = sorted(
            zip(feature_names, abs_coef),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return FeatureImportanceResult(
            feature_names=feature_names,
            coefficients=coefficients,
            abs_coefficients=abs_coef,
            ranking=ranking
        )

    @staticmethod
    def residual_diagnostics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> Dict:
        """Tính toán các chỉ số chẩn đoán phần dư để kiểm định giả thuyết OLS.

        Phân tích phần dư là bước kiểm định hậu nghiệm không thể bỏ qua: nếu
        phần dư không thỏa mãn các giả thuyết Gauss-Markov (zero mean, phân phối
        chuẩn, không tự tương quan, phương sai đồng nhất) thì ước lượng OLS vẫn
        unbiased nhưng không còn là BLUE (Best Linear Unbiased Estimator) và các
        khoảng tin cậy tính được sẽ không chính xác. Kết quả hàm này trực tiếp
        hỗ trợ phân tích trong báo cáo mục phân tích phần dư.

        Args:
            y_true: Vector giá trị thực tế total_cost hình dạng (n,).
            y_pred: Vector giá trị dự đoán từ mô hình hình dạng (n,).
            X: Ma trận đặc trưng tùy chọn, hiện chưa được sử dụng nhưng giữ
               lại để mở rộng tính VIF hoặc White test sau này.

        Returns:
            Dictionary chứa: residuals (vector phần dư), mean_residual, std_residual,
            min/max_residual, autocorr_lag1 (tự tương quan bậc 1), normality_test
            (kết quả Shapiro-Wilk).
        """
        residuals = y_true - y_pred

        diagnostics = {
            "residuals": residuals,
            "mean_residual": np.mean(residuals),
            "std_residual": np.std(residuals),
            "min_residual": np.min(residuals),
            "max_residual": np.max(residuals),
            "autocorr_lag1": np.corrcoef(residuals[:-1], residuals[1:])[0, 1],
            "normality_test": _shapiro_test(residuals),
        }

        return diagnostics

    @staticmethod
    def hyperparameter_tuning(
        X: np.ndarray,
        y: np.ndarray,
        fit_func: Callable,
        param_name: str,
        param_values: List[float],
        k: int = 5,
        metric: str = "r2"
    ) -> Tuple[float, List[float]]:
        """Tìm siêu tham số tối ưu cho mô hình chuẩn hóa bằng k-fold CV.

        Hàm này triển khai grid search đơn giản: với mỗi giá trị tham số ứng
        viên, thực hiện k-fold CV và tính điểm trung bình. Siêu tham số cho
        điểm tốt nhất (R² cao nhất hoặc MAE/RMSE thấp nhất) được chọn làm giá
        trị tối ưu. Cách tiếp cận này tránh overfitting siêu tham số lên test
        set vì toàn bộ quá trình chỉ sử dụng dữ liệu huấn luyện.

        Args:
            X: Ma trận đặc trưng hình dạng (n, p+1) bao gồm intercept.
            y: Vector giá trị mục tiêu hình dạng (n,).
            fit_func: Hàm nhận (X_train, y_train, param_value) và trả về beta,
                      chữ ký khác với kfold_cv vì có thêm tham số siêu tham số.
            param_name: Tên siêu tham số dùng trong log, ví dụ "lambda", "alpha".
            param_values: Danh sách các giá trị cần thử nghiệm, ví dụ [1, 10, 100].
            k: Số lượng fold cho cross-validation.
            metric: Chỉ số tối ưu hóa: "r2" (tối đa hóa) hoặc "mae"/"rmse"
                    (tối thiểu hóa).

        Returns:
            Tuple (best_param, scores_per_param) trong đó best_param là giá trị
            siêu tham số tốt nhất và scores_per_param là điểm CV cho mỗi ứng viên.
        """
        scores_per_param = []
        # Khởi tạo best_score theo hướng tối ưu của metric: tối đa R², tối thiểu MAE/RMSE
        best_score = -np.inf if metric == "r2" else np.inf
        best_param = param_values[0]

        # Tạo fold một lần và dùng lại cho tất cả giá trị tham số để đảm bảo so sánh công bằng
        folds = ModelEvaluator.kfold_split(X.shape[0], k)

        for param_val in param_values:
            fold_scores = []

            for train_idx, test_idx in folds:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                beta = fit_func(X_train, y_train, param_val)
                y_pred = X_test @ beta

                if metric == "r2":
                    tss = np.sum((y_test - np.mean(y_test)) ** 2)
                    rss = np.sum((y_test - y_pred) ** 2)
                    score = 1 - (rss / tss) if tss > 0 else 0
                elif metric == "rmse":
                    score = np.sqrt(np.mean((y_test - y_pred) ** 2))
                else:  # mae
                    score = np.mean(np.abs(y_test - y_pred))

                fold_scores.append(score)

            mean_score = np.mean(fold_scores)
            scores_per_param.append(mean_score)

            # Cập nhật best theo chiều tối ưu phù hợp với metric đã chọn
            if metric == "r2":
                if mean_score > best_score:
                    best_score = mean_score
                    best_param = param_val
            else:
                if mean_score < best_score:
                    best_score = mean_score
                    best_param = param_val

        return best_param, scores_per_param


def _shapiro_test(residuals: np.ndarray) -> Dict:
    """Kiểm định Shapiro-Wilk để xác định phần dư có tuân theo phân phối chuẩn không.

    Shapiro-Wilk là kiểm định chuẩn tắc mạnh nhất cho cỡ mẫu nhỏ và vừa, phù
    hợp hơn Kolmogorov-Smirnov cho bài toán kiểm định giả thuyết phân phối
    chuẩn của phần dư. Vì scipy.stats.shapiro giới hạn ở 5000 mẫu, hàm này tự
    động cắt bớt khi cần thiết mà không làm mất tính đại diện của mẫu.

    Args:
        residuals: Vector phần dư (y_true - y_pred) hình dạng (n,), có thể chứa
                   hơn 5000 phần tử (sẽ được cắt tự động).

    Returns:
        Dictionary với ba khóa: "statistic" (giá trị thống kê W), "p_value" (xác
        suất quan sát được dưới H0 phân phối chuẩn), "normal" (True nếu p > 0.05
        tức không đủ bằng chứng bác bỏ H0 phân phối chuẩn).
    """
    from scipy import stats

    try:
        # Shapiro-Wilk yêu cầu n <= 5000; dùng sample đầu tiên để kiểm định
        stat, pvalue = stats.shapiro(residuals[:min(5000, len(residuals))])
        return {"statistic": stat, "p_value": pvalue, "normal": pvalue > 0.05}
    except:
        return {"statistic": np.nan, "p_value": np.nan, "normal": None}


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from data_pipeline import DataPipeline, PipelineConfig
    from models import RegressionModels

    print("Testing Evaluator Module\n")

    # Load data
    config = PipelineConfig(data_dir="data", missing_method="mean")
    pipeline = DataPipeline(config)
    pipe_result = pipeline.run()

    X_train = pipe_result.X_train
    y_train = pipe_result.y_train

    # Define fit functions for CV
    def fit_ols(X, y):
        from numpy.linalg import lstsq
        return lstsq(X, y, rcond=None)[0]

    def fit_ridge(X, y, lam):
        from numpy.linalg import inv
        return inv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y

    # Test k-fold CV
    print("=" * 70)
    print("K-FOLD CROSS-VALIDATION (k=5)")
    print("=" * 70)

    cv_ols = ModelEvaluator.kfold_cv(
        X_train, y_train,
        fit_func=fit_ols,
        k=5,
        model_name="OLS"
    )

    print(f"\n{cv_ols.model_name}:")
    print(f"  MAE:  {cv_ols.mean_mae:.2f} ± {cv_ols.std_mae:.2f}")
    print(f"  RMSE: {cv_ols.mean_rmse:.2f} ± {cv_ols.std_rmse:.2f}")
    print(f"  R²:   {cv_ols.mean_r2:.4f} ± {cv_ols.std_r2:.4f}")

    # Test feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Top 20)")
    print("=" * 70)

    beta_ols = fit_ols(X_train, y_train)
    feature_names = pipe_result.feature_names
    fi_result = ModelEvaluator.feature_importance(feature_names, beta_ols, top_n=20)

    print("\nTop 20 features by |coefficient|:")
    for i, (feat, coef) in enumerate(fi_result.ranking, 1):
        print(f"  {i:2d}. {feat:40s}: {coef:12.2f}")

    # Test hyperparameter tuning
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING: Ridge λ selection")
    print("=" * 70)

    lambdas = [1, 10, 100, 1000, 10000]
    best_lam, scores = ModelEvaluator.hyperparameter_tuning(
        X_train, y_train,
        fit_func=fit_ridge,
        param_name="lambda",
        param_values=lambdas,
        k=5,
        metric="r2"
    )

    print(f"\n  Best λ: {best_lam}")
    print(f"  CV Scores (R²):")
    for lam, score in zip(lambdas, scores):
        print(f"    λ={lam:5d}: R²={score:.4f}")
