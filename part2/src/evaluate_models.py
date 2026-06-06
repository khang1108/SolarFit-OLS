# -*- coding: utf-8 -*-
"""
HỆ THỐNG HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH HỒI QUY BỨC XẠ MẶT TRỜI
Tác giả: Nhóm Nghiên cứu TUDTK
Mục tiêu:
1. Đọc dữ liệu (giả lập TAHMO hoặc tệp thực tế) và áp dụng DataPipeline tiền xử lý.
2. Huấn luyện đồng thời 4 mô hình/chiến lược:
   - OLS Raw Baseline (Hồi quy tuyến tính trên dữ liệu thô, không xử lý ngoại lai).
   - OLS Proposed (Hồi quy tuyến tính trên dữ liệu đã chuẩn hóa, Winsorize P99 và Log-Target).
   - Gradient Boosting Regressor (Mô hình cây quyết định học tuần tự nâng cao).
   - SVR - Support Vector Regression (Mô hình hồi quy vector hỗ trợ phi tuyến mạnh mẽ).
3. Đánh giá khách quan bằng kiểm thử chéo phân tầng Stratified 5-Fold Cross Validation.
4. Trực quan hóa sai số RMSE và xuất biểu đồ so sánh tự động để chèn vào báo cáo.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Sử dụng Agg backend không tương tác để lưu ảnh không bị treo trên Terminal/Server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa đặc trưng bắt buộc cho SVR
import os
import sys

# Đảm bảo in tiếng Việt có dấu chuẩn xác trên console Windows
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import lớp tiền xử lý DataPipeline đã hoàn thiện từ file data_pipeline.py
try:
    from data_pipeline import DataPipeline
except ImportError:
    print("[LỖI]: Không tìm thấy file 'data_pipeline.py' trong cùng thư mục!")
    print("Vui lòng đảm bảo file 'data_pipeline.py' và 'evaluate_models.py' nằm cạnh nhau.")
    sys.exit(1)

# =====================================================================
# 1. ĐỊNH NGHĨA THUẬT TOÁN HỒI QUY OLS TOÁN HỌC (TỰ CÀI ĐẶT BẰNG MA TRẬN)
# =====================================================================
def fit_ols(X, y):
    """
    Huấn luyện mô hình Hồi quy Tuyến tính OLS bằng công thức ma trận thuần túy:
    Beta = (X^T * X)^-1 * X^T * Y
    
    Tham số:
      - X: Ma trận đặc trưng đầu vào kích thước (N, M), đã được chèn cột Intercept (cột toàn số 1).
      - y: Vector biến mục tiêu thực tế kích thước (N,).
    Trả về:
      - Vector trọng số Beta tối ưu kích thước (M,).
    """
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

def predict_ols(X, beta):
    """
    Dự đoán giá trị đầu ra dựa trên ma trận đặc trưng X và vector trọng số Beta:
    Y_pred = X * Beta
    """
    return np.dot(X, beta)


# =====================================================================
# 2. NẠP DỮ LIỆU DỰ ÁN (HỖ TRỢ CẢ GIẢ LẬP LẪN DỮ LIỆU THỰC TẾ TAHMO)
# =====================================================================
TARGET_COL = 'shortwave_radiation'
NUMERIC_FEATURES = ['temperature']

print("="*70)
print("                 KHỞI TẠO TIỀN XỬ LÝ & NẠP DỮ LIỆU TAHMO")
print("="*70)

# Kiểm tra sự tồn tại của file dữ liệu thực tế
data_path = 'data/tahmo_radiation.csv'
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"[THÀNH CÔNG]: Đã nạp dữ liệu thực tế từ tệp '{data_path}'")
else:
    # Nếu không có dữ liệu thực tế, tự động chạy chế độ giả lập phân phối dữ liệu TAHMO chuẩn
    print(f"[CẢNH BÁO]: Không tìm thấy '{data_path}'. Đang giả lập dữ liệu TAHMO để chạy thử nghiệm...")
    np.random.seed(42)
    df = pd.DataFrame({
        'shortwave_radiation': np.concatenate([
            np.zeros(4000),                              # Khoảng thời gian ban đêm (bức xạ = 0)
            np.random.normal(500, 150, 5950),            # Ban ngày bình thường
            np.random.uniform(1500, 3500, 50)            # Ngoại lai nhiễu vật lý do hỏng cảm biến (sensor drift)
        ]),
        'temperature': np.random.normal(25, 5, 10000)    # Đặc trưng nhiệt độ môi trường
    })
    # Giả lập khuyết thiếu dữ liệu (Missing Values) để kiểm tra tính năng Imputation của Pipeline
    df.loc[np.random.choice(df.index, 300), 'shortwave_radiation'] = np.nan
    df.loc[np.random.choice(df.index, 200), 'temperature'] = np.nan
    print(f"[THÀNH CÔNG]: Khởi tạo dữ liệu giả lập thành công! Kích thước: {df.shape}")

# Đảm bảo giá trị bức xạ mặt trời không âm về mặt vật lý
df[TARGET_COL] = np.where(df[TARGET_COL] < 0, 0, df[TARGET_COL])

# Loại bỏ các dòng bị khuyết Target trước khi đưa vào huấn luyện mô hình
df_model = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
print(f"-> Số lượng mẫu dữ liệu sau khi lọc bỏ missing Target: {len(df_model)} dòng")


# =====================================================================
# 3. THIẾT LẬP CHIẾN LƯỢC KIỂM THỬ PHÂN TẦNG (STRATIFIED K-FOLD)
# =====================================================================
# Sử dụng Stratified K-Fold (k=5) phân tầng theo các phân vị Target (Target Quantiles)
# Chia dữ liệu Target thành 10 bins phân vị để đảm bảo phân phối ngày/đêm 
# và các khoảng cường độ bức xạ luôn đồng đều trên cả 5 folds.
df_model['target_bin'] = pd.qcut(df_model[TARGET_COL].rank(method='first'), q=10, labels=False)
bins = df_model['target_bin']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# =====================================================================
# 4. CHẠY VÒNG LẶP HUẤN LUYỆN & KIỂM THỬ CHÉO 5 FOLDS
# =====================================================================
# Khởi tạo từ điển lưu trữ sai số RMSE cho từng mô hình qua các fold
results = {
    'OLS_Raw': [],
    'OLS_Proposed': [],
    'Gradient_Boosting': [],
    'SVR': []
}

print("\n" + "="*70)
print("      BẮT ĐẦU ĐÁNH GIÁ 5-FOLD CROSS VALIDATION CHO CẢ 4 MÔ HÌNH")
print("="*70)

for fold, (train_idx, val_idx) in enumerate(skf.split(df_model, bins), 1):
    print(f"\n>>> Đang xử lý Fold {fold}/5...")
    
    # Phân chia tập dữ liệu Train/Validation độc lập cho fold hiện tại
    df_train = df_model.iloc[train_idx].copy()
    df_val = df_model.iloc[val_idx].copy()
    
    # Lưu nhãn thực tế nguyên bản chưa xử lý của tập Validation làm hệ quy chiếu đối chứng
    y_val_actual = df_val[TARGET_COL].values
    
    # -----------------------------------------------------------------
    # MÔ HÌNH 1: OLS RAW BASELINE (Học trực tiếp trên dữ liệu thô)
    # -----------------------------------------------------------------
    # Điền khuyết cơ bản bằng trung vị tập Train để tránh Data Leakage
    temp_median = df_train['temperature'].median()
    X_train_raw = df_train['temperature'].fillna(temp_median).values
    X_val_raw = df_val['temperature'].fillna(temp_median).values
    
    # Chèn cột Intercept (toàn số 1) vào ma trận đặc trưng
    X_tr_raw = np.c_[np.ones(X_train_raw.shape[0]), X_train_raw]
    X_va_raw = np.c_[np.ones(X_val_raw.shape[0]), X_val_raw]
    y_tr_raw = df_train[TARGET_COL].values
    
    # Huấn luyện OLS trên dữ liệu thô bằng công thức ma trận tự cài đặt
    beta_raw = fit_ols(X_tr_raw, y_tr_raw)
    y_pred_raw = predict_ols(X_va_raw, beta_raw)
    
    rmse_raw = np.sqrt(np.mean((y_val_actual - y_pred_raw)**2))
    results['OLS_Raw'].append(rmse_raw)
    print(f"  * OLS Raw Baseline RMSE : {rmse_raw:.4f}")
    
    # -----------------------------------------------------------------
    # TIỀN XỬ LÝ DỮ LIỆU NÂNG CAO BẰNG DATAPIPELINE ĐỀ XUẤT
    # -----------------------------------------------------------------
    # Khởi tạo pipeline (Winsorize P99 + Log1p Target + Điền khuyết Median)
    pipe = DataPipeline(target_col=TARGET_COL, numeric_features=NUMERIC_FEATURES)
    pipe.fit(df_train)
    
    train_trans = pipe.transform(df_train)
    val_trans = pipe.transform(df_val)
    
    # Trích xuất ma trận đặc trưng và biến mục tiêu (đã xử lý winsorize & log1p)
    X_train_clean = train_trans[NUMERIC_FEATURES].values
    X_val_clean = val_trans[NUMERIC_FEATURES].values
    y_train_clean_log = train_trans[f'{TARGET_COL}_log1p'].values
    
    # Chuẩn hóa đặc trưng số (Standardization) bằng StandardScaler
    # Cực kỳ quan trọng để SVR hoạt động chính xác và giúp thuật toán OLS ổn định số học
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_val_scaled = scaler.transform(X_val_clean)
    
    # -----------------------------------------------------------------
    # MÔ HÌNH 2: OLS PROPOSED (Dữ liệu đã qua Pipeline & Chuẩn hóa)
    # -----------------------------------------------------------------
    # Chèn Intercept
    X_tr_proposed = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
    X_va_proposed = np.c_[np.ones(X_val_scaled.shape[0]), X_val_scaled]
    
    # Huấn luyện OLS Proposed
    beta_proposed = fit_ols(X_tr_proposed, y_train_clean_log)
    y_pred_proposed_log = predict_ols(X_va_proposed, beta_proposed)
    # Đổi ngược dự đoán từ dạng Log về dạng đơn vị gốc bằng expm1 để so sánh công bằng
    y_pred_proposed = np.expm1(y_pred_proposed_log)
    
    rmse_proposed = np.sqrt(np.mean((y_val_actual - y_pred_proposed)**2))
    results['OLS_Proposed'].append(rmse_proposed)
    print(f"  * OLS Proposed RMSE     : {rmse_proposed:.4f}")
    
    # -----------------------------------------------------------------
    # MÔ HÌNH 3: GRADIENT BOOSTING REGRESSOR (Dữ liệu đã qua Pipeline & Chuẩn hóa)
    # -----------------------------------------------------------------
    # Khởi tạo mô hình Gradient Boosting của Scikit-Learn
    gb_model = GradientBoostingRegressor(
        n_estimators=100,      # Số lượng cây quyết định tuần tự
        learning_rate=0.1,     # Tốc độ học (co hẹp trọng số cây)
        max_depth=4,           # Độ sâu tối đa để kiểm soát quá khớp (overfitting)
        random_state=42
    )
    # Huấn luyện trên nhãn Log-Target đã được lọc sạch nhiễu
    gb_model.fit(X_train_scaled, y_train_clean_log)
    y_pred_gb_log = gb_model.predict(X_val_scaled)
    # Đưa dự đoán log ngược về đơn vị ban đầu
    y_pred_gb = np.expm1(y_pred_gb_log)
    
    rmse_gb = np.sqrt(np.mean((y_val_actual - y_pred_gb)**2))
    results['Gradient_Boosting'].append(rmse_gb)
    print(f"  * Gradient Boosting RMSE: {rmse_gb:.4f}")
    
    # -----------------------------------------------------------------
    # MÔ HÌNH 4: SVR - SUPPORT VECTOR REGRESSION (Dữ liệu đã qua Pipeline & Chuẩn hóa)
    # -----------------------------------------------------------------
    # Khởi tạo mô hình SVR với hàm nhân phi tuyến RBF mạnh mẽ
    svr_model = SVR(
        kernel='rbf',          # Sử dụng hàm nhân phi tuyến Radial Basis Function
        C=10.0,                # Tham số phạt lỗi điều hòa (Regularization)
        epsilon=0.1            # Biên không phạt lỗi
    )
    # Huấn luyện trên dữ liệu đặc trưng đã scaling và nhãn đã xử lý ngoại lai
    svr_model.fit(X_train_scaled, y_train_clean_log)
    y_pred_svr_log = svr_model.predict(X_val_scaled)
    # Đưa dự đoán log ngược về đơn vị ban đầu
    y_pred_svr = np.expm1(y_pred_svr_log)
    
    rmse_svr = np.sqrt(np.mean((y_val_actual - y_pred_svr)**2))
    results['SVR'].append(rmse_svr)
    print(f"  * SVR Model RMSE        : {rmse_svr:.4f}")


# =====================================================================
# 5. TỔNG HỢP KẾT QUẢ & IN BÁO CÁO ĐỐI CHỨNG CHUYÊN SÂU
# =====================================================================
print("\n" + "="*70)
print("             BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (RMSE)")
print("="*70)
print(f"{'Mô hình / Chiến lược':<35} | {'RMSE Trung bình':<18} | {'Độ lệch chuẩn (Std)':<10}")
print("-"*70)

summary_results = {}
for model_name, rmse_list in results.items():
    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    summary_results[model_name] = mean_rmse
    print(f"{model_name:<35} | {mean_rmse:<18.4f} | {std_rmse:<10.4f}")

print("="*70)

# Tìm mô hình tối ưu nhất
best_model = min(summary_results, key=summary_results.get)
print(f"\n=> KẾT LUẬN CUỐI CÙNG: Mô hình tối ưu nhất về mặt thực nghiệm là '{best_model}'")
print(f"   với RMSE trung bình thấp nhất qua 5 fold kiểm thử chéo là {summary_results[best_model]:.4f}")
print("="*70)


# =====================================================================
# 6. TỰ ĐỘNG XUẤT ĐỒ THỊ SO SÁNH RMSE CHẤT LƯỢNG CAO
# =====================================================================
try:
    plt.figure(figsize=(11, 6.5))
    model_names_display = [
        "OLS Raw (Baseline)", 
        "OLS Proposed (P99 Win + Log)", 
        "Gradient Boosting (Proposed)", 
        "SVR (Proposed)"
    ]
    mean_rmses = [summary_results['OLS_Raw'], summary_results['OLS_Proposed'], 
                  summary_results['Gradient_Boosting'], summary_results['SVR']]
    
    # Thiết lập màu sắc chuyên nghiệp
    colors = ['#d9534f', '#0275d8', '#f0ad4e', '#5cb85c'] # Đỏ nhạt, Xanh dương, Cam nhạt, Xanh lá nhạt
    
    bars = plt.bar(model_names_display, mean_rmses, color=colors, alpha=0.85, edgecolor='black', width=0.55)
    plt.ylabel('RMSE Trung bình trên tập Validation (W/m²)', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('So Sánh Sai Số RMSE Giữa Các Mô Hình Trên 5-Fold Validation', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # In điểm số chính xác trên đầu mỗi cột đồ thị
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(mean_rmses) * 0.01),
                 f"{height:.2f} W/m²", ha='center', va='bottom', fontweight='bold', fontsize=10.5)
        
    plt.tight_layout()
    
    # Lưu tệp đồ thị dạng PNG phân giải cao ngay cạnh file code hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, 'rmse_comparison.png')
    
    plt.savefig(plot_path, dpi=300)
    print(f"\n[THÀNH CÔNG]: Biểu đồ so sánh chất lượng cao đã được lưu tại: '{plot_path}'")
except Exception as e:
    print(f"\n[LƯU Ý]: Không thể xuất đồ thị đồ họa. Chi tiết lỗi: {e}")
    print("Mặc dù vậy, toàn bộ kết quả dạng bảng văn bản đã được hiển thị đầy đủ và chuẩn xác ở trên!")
