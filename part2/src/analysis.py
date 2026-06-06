"""
Pipeline mô hình hóa thống nhất cho Phần 2.

File này gom phần mô hình rải rác trước đó về một luồng chạy duy nhất:
nạp dữ liệu qua DataPipeline, huấn luyện các mô hình đã chọn, đánh giá bằng
cross-validation trên tập train, rồi xuất từng file submission riêng cho
Zindi. Vì tập test của cuộc thi không có nhãn, score cuối cùng phải lấy bằng
cách upload từng file submission lên website và điền lại vào bảng so sánh.

Các mô hình được giữ lại:
    1. OLS
    2. OLS có chọn biến bằng p-value
    3. Ridge
    4. Lasso
    5. ElasticNet
    6. Polynomial bậc 2 + ElasticNet
    7. Kernel Ridge Regression (RBF)
    8. Ensemble trung bình

Gradient Boosting và SVR được loại khỏi pipeline này theo yêu cầu hiện tại.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

from data_pipeline import DataPipeline, PipelineConfig, PipelineResult


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


RANDOM_STATE = 42
TARGET_COL = "total_cost"
CV_FOLDS = 5


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
        return float(np.mean(self.fold_mae))

    @property
    def std_mae(self) -> float:
        return float(np.std(self.fold_mae))

    @property
    def mean_rmse(self) -> float:
        return float(np.mean(self.fold_rmse))

    @property
    def std_rmse(self) -> float:
        return float(np.std(self.fold_rmse))

    @property
    def mean_r2(self) -> float:
        return float(np.mean(self.fold_r2))

    @property
    def std_r2(self) -> float:
        return float(np.std(self.fold_r2))


@dataclass
class FinalModelResult:
    """Kết quả cuối cùng của một mô hình sau khi fit trên toàn bộ train set.

    Dự đoán được lưu ở đơn vị gốc TZS, không phải log-space. Nhờ vậy file
    submission có thể ghi trực tiếp ra cột total_cost mà không cần xử lý thêm.
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


class KernelRidgeRBF:
    """Kernel Ridge Regression với RBF kernel.

    Mô hình này dùng công thức nghiệm đối ngẫu:
    alpha = (K + lambda I)^(-1)y, trong đó K là ma trận Gram của RBF kernel.
    Do K có kích thước n x n, pipeline chỉ fit KRR trên một mẫu con cố định để
    script vẫn chạy được ổn định trên laptop khi bấm Ctrl+Alt+N.
    """

    def __init__(self, alpha: float, gamma: float):
        self.alpha = alpha
        self.gamma = gamma
        self.X_fit: Optional[np.ndarray] = None
        self.dual_coef: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y_log: np.ndarray) -> "KernelRidgeRBF":
        self.X_fit = X.astype(float, copy=False)
        kernel = self._rbf_kernel(self.X_fit, self.X_fit)
        regularized = kernel + self.alpha * np.eye(kernel.shape[0])
        try:
            self.dual_coef = np.linalg.solve(regularized, y_log)
        except np.linalg.LinAlgError:
            self.dual_coef = np.linalg.lstsq(regularized, y_log, rcond=None)[0]
        return self

    def predict_log(self, X: np.ndarray) -> np.ndarray:
        if self.X_fit is None or self.dual_coef is None:
            raise RuntimeError("KernelRidgeRBF phải fit trước khi predict.")
        kernel = self._rbf_kernel(X.astype(float, copy=False), self.X_fit)
        return kernel @ self.dual_coef

    def _rbf_kernel(self, X_left: np.ndarray, X_right: np.ndarray) -> np.ndarray:
        left_norm = np.sum(X_left * X_left, axis=1)[:, None]
        right_norm = np.sum(X_right * X_right, axis=1)[None, :]
        dist2 = np.maximum(left_norm + right_norm - 2.0 * (X_left @ X_right.T), 0.0)
        return np.exp(-self.gamma * dist2)


def run() -> Dict[str, Path]:
    """Chạy toàn bộ pipeline hợp nhất và trả về các đường dẫn output chính."""
    part2_dir = Path(__file__).resolve().parents[1]
    data_dir = part2_dir / "data"
    output_dir = part2_dir / "outputs"

    print("=" * 78)
    print("PHAN 2 - PIPELINE MO HINH HOP NHAT")
    print("=" * 78)
    print("Bo model: OLS, OLS+Selection, Ridge, Lasso, ElasticNet, Poly+EN, KRR, Ensemble")
    print("Da loai khoi pipeline: Gradient Boosting, SVR")

    pipe_result = _load_processed_data(data_dir)
    X_train = pipe_result.X_train
    X_test = pipe_result.X_test
    y_train = pipe_result.y_train
    feature_names = pipe_result.feature_names

    print("\n" + "-" * 78)
    print("HUAN LUYEN VA DANH GIA MO HINH")
    print("-" * 78)

    results: List[FinalModelResult] = []
    cv_results: Dict[str, Optional[CVResult]] = {}

    ols_cv = _cross_validate(
        "OLS",
        X_train,
        y_train,
        lambda X_tr, y_log_tr, X_val: _predict_ols_cost(X_tr, y_log_tr, X_val),
        note="5-fold CV tren toan bo train, target log1p, metric TZS",
    )
    ols_result = _fit_ols("OLS", "ols", X_train, y_train, X_test, feature_names)
    results.append(ols_result)
    cv_results[ols_result.slug] = ols_cv

    selected_cv = _cross_validate(
        "OLS_Selected",
        X_train,
        y_train,
        lambda X_tr, y_log_tr, X_val: _predict_ols_selected_cost(
            X_tr, y_log_tr, X_val, feature_names
        ),
        note="5-fold CV, moi fold chon bien bang p-value tren train fold",
    )
    selected_result = _fit_ols_selected(X_train, y_train, X_test, feature_names)
    results.append(selected_result)
    cv_results[selected_result.slug] = selected_cv

    ridge_alpha, ridge_cv = _choose_ridge_alpha(X_train, y_train)
    ridge_result = _fit_ridge(X_train, y_train, X_test, feature_names, ridge_alpha)
    results.append(ridge_result)
    cv_results[ridge_result.slug] = ridge_cv

    lasso_result = _fit_lasso_cv(X_train, y_train, X_test, feature_names)
    lasso_cv = _cross_validate(
        "Lasso",
        X_train,
        y_train,
        lambda X_tr, y_log_tr, X_val: _predict_lasso_cost(
            X_tr, y_log_tr, X_val, alpha=_extract_float(lasso_result.detail, "alpha")
        ),
        note="5-fold CV voi alpha da chon bang LassoCV tren train",
    )
    results.append(lasso_result)
    cv_results[lasso_result.slug] = lasso_cv

    elastic_result = _fit_elasticnet_cv(X_train, y_train, X_test, feature_names)
    elastic_alpha = _extract_float(elastic_result.detail, "alpha")
    elastic_l1 = _extract_float(elastic_result.detail, "l1_ratio")
    elastic_cv = _cross_validate(
        "ElasticNet",
        X_train,
        y_train,
        lambda X_tr, y_log_tr, X_val: _predict_elasticnet_cost(
            X_tr, y_log_tr, X_val, alpha=elastic_alpha, l1_ratio=elastic_l1
        ),
        note="5-fold CV voi alpha/l1_ratio da chon bang ElasticNetCV tren train",
    )
    results.append(elastic_result)
    cv_results[elastic_result.slug] = elastic_cv

    X_train_poly, X_test_poly, poly_names = _make_polynomial_design(
        X_train, X_test, feature_names, pipe_result.feature_types
    )
    poly_result = _fit_poly_elasticnet_cv(X_train_poly, y_train, X_test_poly, poly_names)
    poly_alpha = _extract_float(poly_result.detail, "alpha")
    poly_l1 = _extract_float(poly_result.detail, "l1_ratio")
    poly_cv = _cross_validate(
        "Polynomial2_ElasticNet",
        X_train_poly,
        y_train,
        lambda X_tr, y_log_tr, X_val: _predict_elasticnet_cost(
            X_tr, y_log_tr, X_val, alpha=poly_alpha, l1_ratio=poly_l1
        ),
        note="5-fold CV tren feature da mo rong polynomial bac 2",
    )
    results.append(poly_result)
    cv_results[poly_result.slug] = poly_cv

    krr_result, krr_cv = _fit_krr(X_train[:, 1:], y_train, X_test[:, 1:])
    results.append(krr_result)
    cv_results[krr_result.slug] = krr_cv

    ensemble_result = _fit_ensemble(results, y_train)
    results.append(ensemble_result)
    cv_results[ensemble_result.slug] = None

    paths = _write_outputs(output_dir, pipe_result, results, cv_results)
    _print_result_table(results, cv_results)
    return paths


def _load_processed_data(data_dir: Path) -> PipelineResult:
    """Nạp và tiền xử lý dữ liệu bằng DataPipeline chung của project."""
    config = PipelineConfig(data_dir=str(data_dir), missing_method="median")
    return DataPipeline(config).run()


def _to_log_target(y_cost: np.ndarray) -> np.ndarray:
    """Đưa target từ TZS về log1p để giảm skewness trước khi fit."""
    return np.log1p(np.clip(np.asarray(y_cost, dtype=float), a_min=0.0, a_max=None))


def _to_cost(y_log_pred: np.ndarray) -> np.ndarray:
    """Đưa dự đoán log-space về đơn vị TZS và chặn giá trị âm."""
    clipped_log = np.clip(np.asarray(y_log_pred, dtype=float), a_min=0.0, a_max=25.0)
    return np.nan_to_num(np.expm1(clipped_log), nan=0.0, posinf=0.0, neginf=0.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Tính MAE, RMSE và R2 trên cùng đơn vị gốc TZS."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    tss = float(np.sum((y_true - np.mean(y_true)) ** 2))
    rss = float(np.sum((y_true - y_pred) ** 2))
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

    Hàm predict_fold nhận X_train_fold, y_log_train_fold và X_valid_fold, sau
    đó trả về dự đoán ở đơn vị TZS. Cách viết này giúp mọi mô hình, từ nghiệm
    đóng OLS tới sklearn, đều đi qua cùng một công thức đánh giá.
    """
    splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_mae: List[float] = []
    fold_rmse: List[float] = []
    fold_r2: List[float] = []

    for train_idx, valid_idx in splitter.split(X):
        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr_cost, y_val_cost = y_cost[train_idx], y_cost[valid_idx]
        y_tr_log = _to_log_target(y_tr_cost)
        y_val_pred = predict_fold(X_tr, y_tr_log, X_val)
        mae, rmse, r2 = _metrics(y_val_cost, y_val_pred)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)

    cv = CVResult(model_name, fold_mae, fold_rmse, fold_r2, note)
    print(
        f"  CV {model_name:<24} "
        f"RMSE={cv.mean_rmse:,.2f} +/- {cv.std_rmse:,.2f} | "
        f"R2={cv.mean_r2:.4f} +/- {cv.std_r2:.4f}"
    )
    return cv


def _fit_ols_beta(X: np.ndarray, y_log: np.ndarray) -> np.ndarray:
    """Ước lượng OLS bằng least squares SVD để ổn định khi có nhiều dummy."""
    return np.linalg.lstsq(X, y_log, rcond=None)[0]


def _predict_ols_cost(X_train: np.ndarray, y_log: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    beta = _fit_ols_beta(X_train, y_log)
    return _to_cost(X_pred @ beta)


def _fit_ols(
    name: str,
    slug: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
) -> FinalModelResult:
    """Fit OLS cuối cùng trên toàn bộ train set."""
    y_log = _to_log_target(y_train)
    beta = _fit_ols_beta(X_train, y_log)
    train_pred = _to_cost(X_train @ beta)
    test_pred = _to_cost(X_test @ beta)
    mae, rmse, r2 = _metrics(y_train, train_pred)
    coef = pd.Series(beta, index=feature_names, name=name)
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
        detail="closed_form=least_squares; target=log1p",
        coefficients=coef,
    )


def _select_by_pvalue(
    X: np.ndarray,
    y_log: np.ndarray,
    p_threshold: float = 0.05,
) -> List[int]:
    """Chọn biến cho OLS dựa trên kiểm định t của từng hệ số.

    Intercept luôn được giữ. Với các ma trận one-hot dễ suy biến, ta dùng
    pseudo-inverse thay vì inverse thường để vẫn tính được sai số chuẩn.
    """
    beta = _fit_ols_beta(X, y_log)
    residuals = y_log - X @ beta
    rank = int(np.linalg.matrix_rank(X))
    dof = max(X.shape[0] - rank, 1)
    sigma2 = float(np.sum(residuals ** 2) / dof)
    cov_beta = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov_beta), 1e-15))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(np.nan_to_num(t_stats)), dof))

    kept = [0] + [idx for idx in range(1, X.shape[1]) if p_values[idx] < p_threshold]
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
    kept = _select_by_pvalue(X_train, y_log)
    beta = _fit_ols_beta(X_train[:, kept], y_log)
    return _to_cost(X_pred[:, kept] @ beta)


def _fit_ols_selected(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    p_threshold: float = 0.05,
) -> FinalModelResult:
    """Fit OLS sau khi chọn biến bằng p-value trên toàn bộ train set."""
    y_log = _to_log_target(y_train)
    kept = _select_by_pvalue(X_train, y_log, p_threshold)
    beta_selected = _fit_ols_beta(X_train[:, kept], y_log)

    beta_full = np.zeros(X_train.shape[1])
    beta_full[kept] = beta_selected

    train_pred = _to_cost(X_train[:, kept] @ beta_selected)
    test_pred = _to_cost(X_test[:, kept] @ beta_selected)
    mae, rmse, r2 = _metrics(y_train, train_pred)
    coef = pd.Series(beta_full, index=feature_names, name="OLS_Selected")
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
    """Fit Ridge bằng công thức đóng, không phạt hệ số intercept."""
    penalty = np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    system = X.T @ X + alpha * penalty
    rhs = X.T @ y_log
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0]


def _predict_ridge_cost(
    X_train: np.ndarray,
    y_log: np.ndarray,
    X_pred: np.ndarray,
    alpha: float,
) -> np.ndarray:
    beta = _fit_ridge_beta(X_train, y_log, alpha)
    return _to_cost(X_pred @ beta)


def _choose_ridge_alpha(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, CVResult]:
    """Chọn lambda Ridge bằng CV theo RMSE trên đơn vị TZS."""
    alpha_grid = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_alpha = alpha_grid[0]
    best_cv: Optional[CVResult] = None

    print("\n  Chon lambda cho Ridge:")
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
        if best_cv is None or cv.mean_rmse < best_cv.mean_rmse:
            best_alpha = alpha
            best_cv = cv

    if best_cv is None:
        raise RuntimeError("Không chọn được alpha cho Ridge.")
    best_cv.model_name = "Ridge"
    best_cv.note = f"5-fold CV tren toan bo train; alpha={best_alpha:g}"
    return best_alpha, best_cv


def _fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    alpha: float,
) -> FinalModelResult:
    y_log = _to_log_target(y_train)
    beta = _fit_ridge_beta(X_train, y_log, alpha)
    train_pred = _to_cost(X_train @ beta)
    test_pred = _to_cost(X_test @ beta)
    mae, rmse, r2 = _metrics(y_train, train_pred)
    coef = pd.Series(beta, index=feature_names, name="Ridge")
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


def _fit_lasso_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
) -> FinalModelResult:
    """Fit LassoCV trên train rồi dự đoán test."""
    y_log = _to_log_target(y_train)
    alphas = np.logspace(-4, 1, 30)
    model = LassoCV(
        alphas=alphas,
        cv=CV_FOLDS,
        fit_intercept=True,
        max_iter=50000,
        n_jobs=-1,
    )
    model.fit(X_train[:, 1:], y_log)
    train_pred = _to_cost(model.predict(X_train[:, 1:]))
    test_pred = _to_cost(model.predict(X_test[:, 1:]))
    mae, rmse, r2 = _metrics(y_train, train_pred)

    full_coef = np.r_[model.intercept_, model.coef_]
    coef = pd.Series(full_coef, index=feature_names, name="Lasso")
    return FinalModelResult(
        name="Lasso",
        slug="lasso",
        train_pred=train_pred,
        test_pred=test_pred,
        train_mae=mae,
        train_rmse=rmse,
        train_r2=r2,
        feature_count=X_train.shape[1],
        nonzero_coef=int(np.sum(np.abs(full_coef) > 1e-10)),
        detail=f"alpha={model.alpha_:.12g}; target=log1p; LassoCV={CV_FOLDS}-fold",
        coefficients=coef,
    )


def _predict_lasso_cost(
    X_train: np.ndarray,
    y_log: np.ndarray,
    X_pred: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=50000)
    model.fit(X_train[:, 1:], y_log)
    return _to_cost(model.predict(X_pred[:, 1:]))


def _fit_elasticnet_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
) -> FinalModelResult:
    """Fit ElasticNetCV, tức phối hợp L1 của Lasso và L2 của Ridge."""
    y_log = _to_log_target(y_train)
    alphas = np.logspace(-4, 1, 30)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    model = ElasticNetCV(
        alphas=alphas,
        l1_ratio=l1_ratios,
        cv=CV_FOLDS,
        fit_intercept=True,
        max_iter=50000,
        n_jobs=-1,
    )
    model.fit(X_train[:, 1:], y_log)
    train_pred = _to_cost(model.predict(X_train[:, 1:]))
    test_pred = _to_cost(model.predict(X_test[:, 1:]))
    mae, rmse, r2 = _metrics(y_train, train_pred)

    full_coef = np.r_[model.intercept_, model.coef_]
    coef = pd.Series(full_coef, index=feature_names, name="ElasticNet")
    return FinalModelResult(
        name="ElasticNet",
        slug="elasticnet",
        train_pred=train_pred,
        test_pred=test_pred,
        train_mae=mae,
        train_rmse=rmse,
        train_r2=r2,
        feature_count=X_train.shape[1],
        nonzero_coef=int(np.sum(np.abs(full_coef) > 1e-10)),
        detail=(
            f"alpha={model.alpha_:.12g}; "
            f"l1_ratio={float(model.l1_ratio_):.12g}; "
            f"target=log1p; ElasticNetCV={CV_FOLDS}-fold"
        ),
        coefficients=coef,
    )


def _predict_elasticnet_cost(
    X_train: np.ndarray,
    y_log: np.ndarray,
    X_pred: np.ndarray,
    alpha: float,
    l1_ratio: float,
) -> np.ndarray:
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=50000,
    )
    model.fit(X_train[:, 1:] if _has_intercept(X_train) else X_train, y_log)
    X_eval = X_pred[:, 1:] if _has_intercept(X_pred) else X_pred
    return _to_cost(model.predict(X_eval))


def _has_intercept(X: np.ndarray) -> bool:
    """Nhận diện nhanh ma trận có cột intercept toàn 1 ở vị trí đầu."""
    return X.shape[1] > 0 and np.allclose(X[: min(20, X.shape[0]), 0], 1.0)


def _make_polynomial_design(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[str, str],
    degree: int = 2,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Tạo feature đa thức bậc 2 cho nhóm biến số, giữ nguyên one-hot category.

    Ta không mở rộng polynomial cho biến one-hot vì điều đó tạo ra rất nhiều
    tương tác giả khó diễn giải và dễ overfit. Phần phi tuyến chỉ áp dụng cho
    các biến đếm/số đêm đã log1p + scale.
    """
    numeric_indices = [
        idx for idx, name in enumerate(feature_names)
        if feature_types.get(name) == "numeric"
    ]
    categorical_indices = [
        idx for idx, name in enumerate(feature_names)
        if feature_types.get(name) == "categorical"
    ]

    if not numeric_indices:
        raise RuntimeError("Không tìm thấy biến numeric để tạo polynomial features.")

    numeric_names = [feature_names[idx] for idx in numeric_indices]
    categorical_names = [feature_names[idx] for idx in categorical_indices]

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    train_numeric_poly = poly.fit_transform(X_train[:, numeric_indices])
    test_numeric_poly = poly.transform(X_test[:, numeric_indices])
    poly_names = poly.get_feature_names_out(numeric_names).tolist()

    X_train_poly = np.column_stack([train_numeric_poly, X_train[:, categorical_indices]])
    X_test_poly = np.column_stack([test_numeric_poly, X_test[:, categorical_indices]])
    names = poly_names + categorical_names
    return X_train_poly, X_test_poly, names


def _fit_poly_elasticnet_cv(
    X_train_poly: np.ndarray,
    y_train: np.ndarray,
    X_test_poly: np.ndarray,
    poly_names: List[str],
) -> FinalModelResult:
    """Fit ElasticNet trên feature số đã mở rộng polynomial bậc 2."""
    y_log = _to_log_target(y_train)
    alphas = np.logspace(-4, 1, 25)
    l1_ratios = [0.5, 0.7, 0.9, 0.95, 0.99]
    model = ElasticNetCV(
        alphas=alphas,
        l1_ratio=l1_ratios,
        cv=CV_FOLDS,
        fit_intercept=True,
        max_iter=50000,
        n_jobs=-1,
    )
    model.fit(X_train_poly, y_log)
    train_pred = _to_cost(model.predict(X_train_poly))
    test_pred = _to_cost(model.predict(X_test_poly))
    mae, rmse, r2 = _metrics(y_train, train_pred)

    coef = pd.Series(model.coef_, index=poly_names, name="Polynomial2_ElasticNet")
    return FinalModelResult(
        name="Polynomial2_ElasticNet",
        slug="poly2_elasticnet",
        train_pred=train_pred,
        test_pred=test_pred,
        train_mae=mae,
        train_rmse=rmse,
        train_r2=r2,
        feature_count=X_train_poly.shape[1],
        nonzero_coef=int(np.sum(np.abs(model.coef_) > 1e-10)),
        detail=(
            f"degree=2; alpha={model.alpha_:.12g}; "
            f"l1_ratio={float(model.l1_ratio_):.12g}; target=log1p"
        ),
        coefficients=coef,
    )


def _fit_krr(
    X_train_no_intercept: np.ndarray,
    y_train: np.ndarray,
    X_test_no_intercept: np.ndarray,
) -> Tuple[FinalModelResult, CVResult]:
    """Chọn siêu tham số và fit Kernel Ridge Regression trên mẫu con."""
    max_cv_rows = 750
    max_fit_rows = 1200
    alpha_grid = [0.1, 1.0, 10.0]
    gamma_grid = [0.0001, 0.001, 0.01]

    cv_idx = _sample_indices(len(y_train), max_cv_rows, RANDOM_STATE)
    X_cv = X_train_no_intercept[cv_idx]
    y_cv = y_train[cv_idx]

    best_alpha = alpha_grid[0]
    best_gamma = gamma_grid[0]
    best_cv: Optional[CVResult] = None

    print("\n  Chon tham so cho KRR tren mau con:")
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
            if best_cv is None or cv.mean_rmse < best_cv.mean_rmse:
                best_alpha = alpha
                best_gamma = gamma
                best_cv = cv

    if best_cv is None:
        raise RuntimeError("Không chọn được tham số cho KRR.")

    fit_idx = _sample_indices(len(y_train), max_fit_rows, RANDOM_STATE)
    y_fit_log = _to_log_target(y_train[fit_idx])
    model = KernelRidgeRBF(alpha=best_alpha, gamma=best_gamma)
    model.fit(X_train_no_intercept[fit_idx], y_fit_log)

    train_pred = _to_cost(model.predict_log(X_train_no_intercept))
    test_pred = _to_cost(model.predict_log(X_test_no_intercept))
    mae, rmse, r2 = _metrics(y_train, train_pred)

    best_cv.model_name = "KernelRidge_RBF"
    best_cv.note = (
        f"3-fold CV tren mau con {len(cv_idx)} rows; "
        f"fit cuoi tren {len(fit_idx)} rows; alpha={best_alpha:g}; gamma={best_gamma:g}"
    )

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
    model = KernelRidgeRBF(alpha=alpha, gamma=gamma)
    model.fit(X_train, y_log)
    return _to_cost(model.predict_log(X_pred))


def _sample_indices(n_rows: int, max_rows: int, random_state: int) -> np.ndarray:
    """Lấy mẫu con cố định để các mô hình kernel chạy được và tái lập được."""
    if n_rows <= max_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False))


def _fit_ensemble(results: List[FinalModelResult], y_train: np.ndarray) -> FinalModelResult:
    """Tạo ensemble bằng trung bình các mô hình đã fit.

    Ensemble chỉ dùng các mô hình không phải OLS_Selected để tránh một biến thể
    OLS chi phối hai lần. Khi có score website, ta có thể thay trung bình đều
    bằng trọng số theo leaderboard.
    """
    ensemble_members = [
        result for result in results
        if result.slug in {"ridge", "lasso", "elasticnet", "poly2_elasticnet", "kernel_ridge_rbf"}
    ]
    train_pred = np.mean([result.train_pred for result in ensemble_members], axis=0)
    test_pred = np.mean([result.test_pred for result in ensemble_members], axis=0)
    mae, rmse, r2 = _metrics(y_train, train_pred)
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
    """Ghi bảng so sánh, dự đoán tổng hợp và từng file submission riêng."""
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir = output_dir / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)

    test_ids = _submission_ids(pipe_result)
    prediction_table = pd.DataFrame({"ID": test_ids})
    comparison_rows = []
    score_rows = []

    for result in results:
        prediction_table[result.slug] = result.test_pred
        submission_path = submission_dir / f"submission_{result.slug}.csv"
        pd.DataFrame(
            {
                "ID": test_ids,
                TARGET_COL: np.clip(result.test_pred, a_min=0.0, a_max=None),
            }
        ).to_csv(submission_path, index=False)

        cv = cv_results.get(result.slug)
        comparison_rows.append(_comparison_row(result, cv, submission_path))
        score_rows.append(
            {
                "model": result.name,
                "submission_file": str(submission_path.relative_to(output_dir.parent)),
                "website_score": "",
                "note": "Upload file nay len Zindi roi dien score vao cot website_score.",
            }
        )

    comparison_path = output_dir / "model_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)

    predictions_path = output_dir / "all_model_predictions.csv"
    prediction_table.to_csv(predictions_path, index=False)

    score_template_path = output_dir / "zindi_score_template.csv"
    pd.DataFrame(score_rows).to_csv(score_template_path, index=False)

    feature_importance_path = output_dir / "feature_importance.csv"
    _write_feature_importance(feature_importance_path, results)

    summary_path = output_dir / "analysis_summary.txt"
    _write_summary(summary_path, pipe_result, results, cv_results)

    print("\n" + "-" * 78)
    print("DA GHI OUTPUT")
    print("-" * 78)
    print(f"  Bang so sanh       : {comparison_path}")
    print(f"  Du doan tong hop   : {predictions_path}")
    print(f"  Template dien score: {score_template_path}")
    print(f"  Thu muc submission : {submission_dir}")

    return {
        "comparison": comparison_path,
        "predictions": predictions_path,
        "score_template": score_template_path,
        "submissions": submission_dir,
        "summary": summary_path,
    }


def _submission_ids(pipe_result: PipelineResult) -> np.ndarray:
    """Lấy ID test đúng thứ tự để ghi submission."""
    if pipe_result.test_ids is None:
        return np.arange(1, pipe_result.X_test.shape[0] + 1).astype(str)
    return pipe_result.test_ids.astype(str)


def _comparison_row(
    result: FinalModelResult,
    cv: Optional[CVResult],
    submission_path: Path,
) -> Dict[str, object]:
    """Tạo một dòng trong bảng so sánh mô hình."""
    return {
        "model": result.name,
        "train_MAE_TZS": result.train_mae,
        "train_RMSE_TZS": result.train_rmse,
        "train_R2": result.train_r2,
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
    rows = []
    for result in results:
        if result.coefficients is None:
            continue
        coef_abs = result.coefficients.abs().sort_values(ascending=False).head(40)
        for feature, abs_value in coef_abs.items():
            rows.append(
                {
                    "model": result.name,
                    "feature": feature,
                    "coefficient": float(result.coefficients.loc[feature]),
                    "abs_coefficient": float(abs_value),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_summary(
    path: Path,
    pipe_result: PipelineResult,
    results: List[FinalModelResult],
    cv_results: Dict[str, Optional[CVResult]],
) -> None:
    """Ghi báo cáo txt ngắn để người chạy nắm nhanh output vừa tạo."""
    with path.open("w", encoding="utf-8") as file:
        file.write("PHAN 2 - TOM TAT PIPELINE MO HINH HOP NHAT\n")
        file.write("=" * 78 + "\n\n")
        file.write(f"Train rows: {pipe_result.X_train.shape[0]:,}\n")
        file.write(f"Test rows : {pipe_result.X_test.shape[0]:,}\n")
        file.write(f"Features  : {len(pipe_result.feature_names):,}\n")
        file.write("Target    : log1p(total_cost) khi train, expm1 ve TZS khi submit\n\n")

        file.write("MODEL SUMMARY\n")
        file.write("-" * 78 + "\n")
        for result in results:
            cv = cv_results.get(result.slug)
            cv_text = (
                f"CV R2={cv.mean_r2:.4f}, CV RMSE={cv.mean_rmse:,.2f}"
                if cv else "CV: xem score website"
            )
            file.write(
                f"{result.name:<26} "
                f"Train R2={result.train_r2:.4f}, "
                f"Train RMSE={result.train_rmse:,.2f}, "
                f"{cv_text}\n"
            )

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
    print("TOM TAT KET QUA")
    print("-" * 78)
    print(f"{'Model':<28} {'Train R2':>10} {'CV R2':>10} {'CV RMSE':>14}")
    for result in results:
        cv = cv_results.get(result.slug)
        cv_r2 = f"{cv.mean_r2:.4f}" if cv else "website"
        cv_rmse = f"{cv.mean_rmse:,.2f}" if cv else "website"
        print(f"{result.name:<28} {result.train_r2:>10.4f} {cv_r2:>10} {cv_rmse:>14}")


def _extract_float(detail: str, key: str) -> float:
    """Lấy giá trị float từ chuỗi detail dạng 'alpha=...; l1_ratio=...'."""
    prefix = f"{key}="
    for part in detail.split(";"):
        value = part.strip()
        if value.startswith(prefix):
            return float(value[len(prefix):])
    raise ValueError(f"Không tìm thấy {key} trong detail: {detail}")


def main() -> None:
    """Entrypoint để chạy trực tiếp bằng python hoặc Ctrl+Alt+N."""
    run()


if __name__ == "__main__":
    main()
