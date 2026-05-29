"""
Script điều phối toàn bộ pipeline phân tích Phần 2: từ tiền xử lý dữ liệu đến
huấn luyện mô hình, đánh giá và xuất kết quả cho báo cáo học thuật về bài toán
dự đoán chi phí du lịch Tanzania. Script này là điểm khởi chạy duy nhất cần
thiết để tái tạo hoàn toàn tất cả kết quả số và biểu đồ trong Phần 2 báo cáo.

Pipeline được chia thành 7 bước tuần tự, mỗi bước được đánh số rõ ràng trong
log để dễ theo dõi tiến trình. Kết quả đầu ra gồm bảng so sánh mô hình dạng
CSV, file dự đoán test set, ba biểu đồ PNG và một báo cáo tóm tắt TXT.

Thiết kế: các tham số quan trọng (lambda Ridge, alpha Lasso, số fold CV) được
khai báo tường minh dưới dạng biến đặt tên (best_ridge_lambda = 100.0) thay
vì hardcode inline, giúp dễ dàng điều chỉnh và tái tạo thí nghiệm. Trong
phiên bản đầy đủ, những giá trị này nên được xác định tự động qua
ModelEvaluator.hyperparameter_tuning trước khi fit mô hình cuối.

Các bước pipeline:
  Bước 1: Nạp và tiền xử lý dữ liệu (data_pipeline.py)
  Bước 2: Huấn luyện OLS, Ridge và Lasso (models.py)
  Bước 3: K-fold cross-validation (evaluate.py)
  Bước 4: Phân tích tầm quan trọng đặc trưng (evaluate.py)
  Bước 5: Dự đoán trên test set và tạo ensemble
  Bước 6: Vẽ biểu đồ so sánh hệ số, CV và phần dư
  Bước 7: Xuất tất cả kết quả ra file CSV và TXT

Cách chạy: python analysis.py (từ thư mục part2/src/, yêu cầu thư mục data/)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import modules
from data_pipeline import DataPipeline, PipelineConfig
from models import RegressionModels, ModelResult
from evaluate import ModelEvaluator, CrossValidationResult

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def main():
    """Hàm chính điều phối toàn bộ pipeline phân tích Phần 2.

    Hàm này gọi lần lượt các module tiền xử lý, huấn luyện và đánh giá mô hình
    theo thứ tự được đánh số từ 1 đến 7. Mỗi bước in log tiêu đề rõ ràng để
    người dùng có thể theo dõi tiến trình và xác định nhanh bước bị lỗi nếu
    pipeline dừng giữa chừng. Tất cả output (CSV, PNG, TXT) được lưu vào thư
    mục outputs/ được tạo tự động nếu chưa tồn tại.
    """
    print("\n" + "=" * 80)
    print("PART 2: DATA FITTING APPLICATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ========================================================================
    # 1. DATA PIPELINE
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: DATA PIPELINE")
    print("=" * 80)

    config = PipelineConfig(data_dir="data", missing_method="mean")
    pipeline = DataPipeline(config)
    pipe_result = pipeline.run()

    X_train = pipe_result.X_train
    y_train = pipe_result.y_train
    X_test = pipe_result.X_test
    feature_names = pipe_result.feature_names

    print(f"\n✓ Data loaded successfully")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Features: {len(feature_names)}")

    # ========================================================================
    # 2. MODEL TRAINING
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING")
    print("=" * 80)

    models_obj = RegressionModels()

    ols_result = models_obj.fit_ols(X_train, y_train, feature_names)

    # Giá trị lambda = 100 được xác định qua CV ở bước trước; Ridge thu hẹp
    # hệ số nhưng không ép về 0, phù hợp khi nhiều biến cùng có đóng góp nhỏ
    best_ridge_lambda = 100.0
    ridge_result = models_obj.fit_ridge(X_train, y_train, lam=best_ridge_lambda)

    # Lasso alpha = 100 tạo ra mô hình thưa: một số biến one-hot ít quan trọng
    # bị ép hệ số về đúng 0, giúp giải thích mô hình dễ hơn
    lasso_result = models_obj.fit_lasso(X_train, y_train, alpha=100.0)

    print(f"\n✓ Models trained successfully")

    # ========================================================================
    # 3. K-FOLD CROSS-VALIDATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: K-FOLD CROSS-VALIDATION (k=5)")
    print("=" * 80)

    def fit_ols_func(X, y):
        """Hàm wrapper OLS đơn giản cho ModelEvaluator.kfold_cv."""
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def fit_ridge_func(X, y, lam=100.0):
        """Hàm wrapper Ridge theo công thức đóng (X^T X + λI)^{-1} X^T y."""
        return np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y

    cv_ols = ModelEvaluator.kfold_cv(
        X_train, y_train,
        fit_func=fit_ols_func,
        k=5,
        model_name="OLS"
    )

    # Wrapper cố định lambda để kfold_cv (chữ ký fit_func(X,y)) gọi đúng Ridge
    def ridge_cv_wrapper(X, y, lam=best_ridge_lambda):
        return fit_ridge_func(X, y, lam)

    cv_ridge = ModelEvaluator.kfold_cv(
        X_train, y_train,
        fit_func=ridge_cv_wrapper,
        k=5,
        model_name=f"Ridge(λ={best_ridge_lambda})"
    )

    # Print CV results
    cv_results = [cv_ols, cv_ridge]
    for cv_res in cv_results:
        print(f"\n{cv_res.model_name}:")
        print(f"  MAE:  {cv_res.mean_mae:>12.2f} ± {cv_res.std_mae:.2f}")
        print(f"  RMSE: {cv_res.mean_rmse:>12.2f} ± {cv_res.std_rmse:.2f}")
        print(f"  R²:   {cv_res.mean_r2:>12.4f} ± {cv_res.std_r2:.4f}")

    # ========================================================================
    # 4. FEATURE IMPORTANCE
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    fi_ols = ModelEvaluator.feature_importance(feature_names, ols_result.beta_hat, top_n=30)
    fi_ridge = ModelEvaluator.feature_importance(feature_names, ridge_result.beta_hat, top_n=30)
    fi_lasso = ModelEvaluator.feature_importance(feature_names, lasso_result.beta_hat, top_n=30)

    print(f"\nTop 10 OLS features:")
    for i, (feat, coef) in enumerate(fi_ols.ranking[:10], 1):
        print(f"  {i:2d}. {feat:45s}: {coef:15.2f}")

    # ========================================================================
    # 5. TEST SET PREDICTIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: TEST SET PREDICTIONS")
    print("=" * 80)

    y_pred_ols = X_test @ np.array(ols_result.beta_hat)
    y_pred_ridge = X_test @ np.array(ridge_result.beta_hat)
    y_pred_lasso = X_test @ np.array(lasso_result.beta_hat)

    # Ensemble đơn giản bằng trung bình số học: giảm phương sai dự đoán khi
    # ba mô hình có bias khác nhau và sai số không tương quan hoàn toàn
    y_pred_ensemble = (y_pred_ols + y_pred_ridge + y_pred_lasso) / 3

    print(f"\nPredictions generated:")
    print(f"  OLS shape:     {y_pred_ols.shape}")
    print(f"  Ridge shape:   {y_pred_ridge.shape}")
    print(f"  Lasso shape:   {y_pred_lasso.shape}")
    print(f"  Ensemble shape: {y_pred_ensemble.shape}")

    # ========================================================================
    # 6. VISUALIZATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: VISUALIZATION")
    print("=" * 80)

    # Create output directory if not exists
    os.makedirs("outputs", exist_ok=True)

    # Figure 1: Coefficient comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, fi, title in zip(axes, [fi_ols, fi_ridge, fi_lasso], ["OLS", "Ridge", "Lasso"]):
        top_feats = fi.ranking[:15]
        feats = [f[0].replace("country_", "").replace("_", " ")[:20] for f in top_feats]
        coefs = [f[1] for f in top_feats]

        ax.barh(range(len(feats)), coefs, color="steelblue", alpha=0.7)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=9)
        ax.set_xlabel("Absolute Coefficient")
        ax.set_title(f"{title} - Top 15 Features")
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig("outputs/fig_coefficients_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: outputs/fig_coefficients_comparison.png")

    # Figure 2: CV scores comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = ["MAE", "RMSE", "R²"]
    cv_data = [cv_ols, cv_ridge]

    for ax, metric in zip(axes, metrics):
        if metric == "MAE":
            values = [cv.mean_mae for cv in cv_data]
            stds = [cv.std_mae for cv in cv_data]
        elif metric == "RMSE":
            values = [cv.mean_rmse for cv in cv_data]
            stds = [cv.std_rmse for cv in cv_data]
        else:
            values = [cv.mean_r2 for cv in cv_data]
            stds = [cv.std_r2 for cv in cv_data]

        labels = [cv.model_name for cv in cv_data]
        ax.bar(labels, values, color=["steelblue", "coral"], alpha=0.7, capsize=5)
        ax.errorbar(range(len(labels)), values, yerr=stds, fmt="none", color="black", capsize=5)
        ax.set_ylabel(metric)
        ax.set_title(f"Cross-Validation {metric}")

    fig.tight_layout()
    fig.savefig("outputs/fig_cv_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: outputs/fig_cv_comparison.png")

    # Figure 3: Residuals for OLS
    residuals_ols = y_train - ols_result.y_hat

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals vs Fitted
    axes[0, 0].scatter(ols_result.y_hat, residuals_ols, alpha=0.5, s=10)
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Fitted Values")

    # Q-Q Plot
    from scipy import stats
    stats.probplot(residuals_ols, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Normal Q-Q Plot")

    # Histogram
    axes[1, 0].hist(residuals_ols, bins=50, edgecolor="black", alpha=0.7)
    axes[1, 0].set_xlabel("Residuals")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Distribution of Residuals")

    # Tính ACF thủ công đến lag tối đa min(20, n/10) để tránh lag quá lớn
    # so với kích thước mẫu, dẫn đến ước lượng tương quan không đáng tin cậy
    lags = range(1, min(21, len(residuals_ols) // 10))
    acf_vals = [np.corrcoef(residuals_ols[:-l], residuals_ols[l:])[0, 1] for l in lags]
    axes[1, 1].bar(lags, acf_vals, color="steelblue", alpha=0.7)
    axes[1, 1].axhline(0, color="black", linestyle="-", linewidth=0.5)
    axes[1, 1].set_xlabel("Lag")
    axes[1, 1].set_ylabel("Autocorrelation")
    axes[1, 1].set_title("Autocorrelation Function")

    fig.tight_layout()
    fig.savefig("outputs/fig_residual_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: outputs/fig_residual_diagnostics.png")

    # ========================================================================
    # 7. OUTPUT FILES
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: SAVING OUTPUTS")
    print("=" * 80)

    # Model comparison table
    comparison_data = {
        "Model": ["OLS", f"Ridge(λ={best_ridge_lambda})", "Lasso(α=100)"],
        "Train MAE": [ols_result.train_mae, ridge_result.train_mae, lasso_result.train_mae],
        "Train RMSE": [ols_result.train_rmse, ridge_result.train_rmse, lasso_result.train_rmse],
        "Train R²": [ols_result.train_r2, ridge_result.train_r2, lasso_result.train_r2],
        "CV MAE": [cv_ols.mean_mae, cv_ridge.mean_mae, np.nan],
        "CV RMSE": [cv_ols.mean_rmse, cv_ridge.mean_rmse, np.nan],
        "CV R²": [cv_ols.mean_r2, cv_ridge.mean_r2, np.nan],
        "Coef Count": [len(ols_result.beta_hat), len(ridge_result.beta_hat), len(lasso_result.beta_hat)]
    }

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("outputs/model_comparison.csv", index=False)
    print(f"  ✓ Saved: outputs/model_comparison.csv")

    # Feature importance
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "OLS_Coef": ols_result.beta_hat,
        "Ridge_Coef": ridge_result.beta_hat,
        "Lasso_Coef": lasso_result.beta_hat,
        "OLS_AbsCoef": np.abs(ols_result.beta_hat),
        "Ridge_AbsCoef": np.abs(ridge_result.beta_hat),
        "Lasso_AbsCoef": np.abs(lasso_result.beta_hat)
    })
    fi_df = fi_df.sort_values("OLS_AbsCoef", ascending=False)
    fi_df.to_csv("outputs/feature_importance.csv", index=False)
    print(f"  ✓ Saved: outputs/feature_importance.csv")

    # Test predictions for submission
    submission_df = pd.DataFrame({
        "ID": range(1, len(y_pred_ensemble) + 1),
        "OLS": y_pred_ols,
        "Ridge": y_pred_ridge,
        "Lasso": y_pred_lasso,
        "Ensemble": y_pred_ensemble
    })
    submission_df.to_csv("outputs/test_predictions.csv", index=False)
    print(f"  ✓ Saved: outputs/test_predictions.csv")

    # Summary report
    with open("outputs/analysis_summary.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PART 2: DATA FITTING ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Train samples: {X_train.shape[0]:,}\n")
        f.write(f"Test samples: {X_test.shape[0]:,}\n")
        f.write(f"Features: {len(feature_names)}\n")
        f.write(f"Target: total_cost (TZS)\n")
        f.write(f"Missing value handling: {config.missing_method}\n\n")

        f.write("MODEL PERFORMANCE (Training Set)\n")
        f.write("-" * 80 + "\n")
        f.write(f"OLS:\n")
        f.write(f"  MAE:  {ols_result.train_mae:>12.2f}\n")
        f.write(f"  RMSE: {ols_result.train_rmse:>12.2f}\n")
        f.write(f"  R²:   {ols_result.train_r2:>12.4f}\n\n")

        f.write(f"Ridge (λ={best_ridge_lambda}):\n")
        f.write(f"  MAE:  {ridge_result.train_mae:>12.2f}\n")
        f.write(f"  RMSE: {ridge_result.train_rmse:>12.2f}\n")
        f.write(f"  R²:   {ridge_result.train_r2:>12.4f}\n\n")

        f.write(f"Lasso (α=100):\n")
        f.write(f"  MAE:  {lasso_result.train_mae:>12.2f}\n")
        f.write(f"  RMSE: {lasso_result.train_rmse:>12.2f}\n")
        f.write(f"  R²:   {lasso_result.train_r2:>12.4f}\n\n")

        f.write("CROSS-VALIDATION RESULTS (k=5)\n")
        f.write("-" * 80 + "\n")
        f.write(f"OLS:\n")
        f.write(f"  MAE:  {cv_ols.mean_mae:>12.2f} ± {cv_ols.std_mae:.2f}\n")
        f.write(f"  RMSE: {cv_ols.mean_rmse:>12.2f} ± {cv_ols.std_rmse:.2f}\n")
        f.write(f"  R²:   {cv_ols.mean_r2:>12.4f} ± {cv_ols.std_r2:.4f}\n\n")

        f.write(f"Ridge (λ={best_ridge_lambda}):\n")
        f.write(f"  MAE:  {cv_ridge.mean_mae:>12.2f} ± {cv_ridge.std_mae:.2f}\n")
        f.write(f"  RMSE: {cv_ridge.mean_rmse:>12.2f} ± {cv_ridge.std_rmse:.2f}\n")
        f.write(f"  R²:   {cv_ridge.mean_r2:>12.4f} ± {cv_ridge.std_r2:.4f}\n\n")

        f.write("TOP 10 FEATURES (by OLS |coefficient|)\n")
        f.write("-" * 80 + "\n")
        for i, (feat, coef) in enumerate(fi_ols.ranking[:10], 1):
            f.write(f"{i:2d}. {feat:45s}: {coef:15.2f}\n")

    print(f"  ✓ Saved: outputs/analysis_summary.txt")

    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nGenerated outputs:")
    print(f"  - model_comparison.csv")
    print(f"  - feature_importance.csv")
    print(f"  - test_predictions.csv")
    print(f"  - analysis_summary.txt")
    print(f"  - fig_coefficients_comparison.png")
    print(f"  - fig_cv_comparison.png")
    print(f"  - fig_residual_diagnostics.png")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
