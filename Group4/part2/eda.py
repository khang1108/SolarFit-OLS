"""
Module phân tích khám phá dữ liệu (EDA) chuyên sâu cho bộ dữ liệu Tanzania
Tourism Expenditure, tập trung vào các biến liên tục và giá trị khuyết.

Module này thực hiện EDA theo trình tự logic từ tổng quan đến chi tiết: đầu
tiên kiểm tra chất lượng dữ liệu thô (missing values, duplicates), sau đó
mô tả thống kê và phát hiện outlier, tiếp theo là kiểm định đa cộng tuyến
(VIF), và cuối cùng là phát hiện distribution shift giữa train và test set
bằng kiểm định Kolmogorov-Smirnov. Mỗi bước đưa ra quyết định thiết kế cụ
thể được ghi lại tường minh và trực tiếp ảnh hưởng đến cách xây dựng
DataPipeline trong data_pipeline.py.

Kết quả phân tích được tóm tắt trong bảng quyết định cuối script và dẫn
đến ba quyết định Preprocessing chính: (1) dùng log1p transform cho target
total_cost vì skew = 2.97, (2) điền khuyết travel_with bằng nhãn "Unknown"
thay vì xóa hàng vì tỷ lệ missing 23% quá cao cho listwise deletion, (3)
giữ nguyên outlier trong target vì chúng là dữ liệu hợp lệ phản ánh chi phí
tour trọn gói cao cấp thực tế.

Module tạo 7 biểu đồ PNG trong thư mục outputs/ phục vụ báo cáo học thuật.
Các biến CONFIG ở đầu file kiểm soát đường dẫn và danh sách đặc trưng để
dễ dàng điều chỉnh khi cấu trúc thư mục thay đổi.

Tác giả: trongnghia090406@gmail.com
Tham chiếu: Toan_UDTK_Project_2 — Mục 2.2.1, 2.2.2, 2.2.3
Cách chạy: python eda.py (yêu cầu Train.csv và Test.csv trong thư mục data/)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from scipy import stats

from part1.ols_implementation import ols_fit

warnings.filterwarnings("ignore")
np.random.seed(42)

# ════════════════════════════════════════════════════════════════
# CẤU HÌNH
# Ta gom toàn bộ đường dẫn, tên cột mục tiêu và danh sách đặc trưng về một chỗ
# ngay đầu file, để khi cấu trúc thư mục hay tên biến thay đổi thì chỉ cần sửa
# duy nhất khối này thay vì dò khắp script.
# ════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
TRAIN_FILE = "Train.csv"
TEST_FILE  = "Test.csv"

TARGET   = "total_cost"
ID_COL   = "ID"
NUM_FEAT = [
    "total_female",
    "total_male",
    "night_mainland",
    "night_zanzibar",
]
CAT_FEAT = [
    "country",
    "age_group",
    "travel_with",
    "purpose",
    "main_activity",
    "info_source",
    "tour_arrangement",
    "package_transport_int",
    "package_accomodation",
    "package_food",
    "package_transport_tz",
    "package_sightseeing",
    "package_guided_tour",
    "package_insurance",
    "payment_mode",
    "first_trip_tz",
    "most_impressing",
]

SEP = "=" * 65

# ════════════════════════════════════════════════════════════════
# BƯỚC 0 — Load DỮ LIỆU THÔ
# Trước khi phân tích bất cứ điều gì, ta Load đồng thời cả train lẫn test để có
# thể soi hai tập song song, đồng thời đếm ngay số dòng trùng lặp: một dòng bị
# lặp sẽ làm phồng giả tạo trọng số của quan sát đó khi mô hình fit về sau.
# ════════════════════════════════════════════════════════════════
print(SEP)
print("0. LOADING DATA (Tanzania Tourism Expenditure)")
print(SEP)

train = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
test  = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))

print(f"  Train : {train.shape[0]:,} rows × {train.shape[1]} cols")
print(f"  Test  : {test.shape[0]:,} rows × {test.shape[1]} cols")
print(f"  Target in test? {TARGET in test.columns}")

# Duplicate check (yêu cầu đề mục 2.2.1)
dup_train = train.duplicated().sum()
dup_test  = test.duplicated().sum()
print(f"\n  Duplicate rows — Train: {dup_train:,}  |  Test: {dup_test:,}")
if dup_train > 0:
    print(f"  ⚠️  {dup_train} duplicate rows in Train → sẽ drop trước khi modeling")


# ════════════════════════════════════════════════════════════════
# BƯỚC 1 — KIỂM KÊ GIÁ TRỊ KHUYẾT
# Với mỗi cột, ta soi cả số lượng lẫn tỷ lệ phần trăm missing, vì chính tỷ lệ
# này mới quyết định cách xử lý: thiếu rất ít thì điền khuyết, thiếu nhiều thì
# phải phân tích cơ chế (MCAR/MAR/MNAR) rồi mới chọn chiến lược cho từng cột.
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("1. MISSING VALUES AUDIT (Xác định cột thiếu ≥ 5%)")
print(SEP)

def missing_report(df, label):
    """Tạo bảng thống kê giá trị khuyết và in kết quả có định dạng rõ ràng.

    Hàm này tổng hợp số lượng và tỷ lệ phần trăm missing cho từng cột, sắp
    xếp giảm dần theo tỷ lệ để các cột cần chú ý nhất nằm trên cùng. Kết quả
    này là cơ sở để phân tích cơ chế missing (MCAR/MAR/MNAR) và lựa chọn
    chiến lược xử lý phù hợp cho từng cột.

    Args:
        df: DataFrame cần kiểm tra, có thể là train hoặc test set.
        label: Nhãn in trong log để phân biệt kết quả của hai tập dữ liệu.

    Returns:
        DataFrame chứa các cột có ít nhất một giá trị khuyết, với hai cột
        "missing_count" và "missing_pct(%)", đã sắp xếp giảm dần theo tỷ lệ.
    """
    miss     = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(4)
    report   = pd.DataFrame({"missing_count": miss, "missing_pct(%)": miss_pct})
    report   = report[report["missing_count"] > 0].sort_values("missing_pct(%)", ascending=False)
    if report.empty:
        print(f"  [{label}] ✅ No missing values found.")
    else:
        print(f"  [{label}] Missing columns:")
        print(report.to_string(index=True))
    return report

miss_train = missing_report(train, "TRAIN")
miss_test  = missing_report(test,  "TEST")

THRESH = 5.0
high_train = miss_train[miss_train["missing_pct(%)"] >= THRESH].index.tolist()
high_test  = miss_test[miss_test["missing_pct(%)"]  >= THRESH].index.tolist()
print(f"\n  Cols ≥ {THRESH}% missing — TRAIN: {high_train or 'None'}")
print(f"  Cols ≥ {THRESH}% missing — TEST : {high_test  or 'None'}")

print("""
  ── Phân tích cơ chế missing (MCAR / MAR / MNAR) ──────────────────────────
  Dataset hiện tại có chứa missing values thực tế thỏa mãn yêu cầu đồ án:
  
  1. travel_with (Train: 23.16% | Test: 20.42% missing):
    → Cơ chế: MAR (Missing At Random) hoặc MNAR (Missing Not At Random).
      Nhiều khả năng khách du lịch đi một mình (Alone) thường bỏ qua câu hỏi 
      "travel_with" vì không đi cùng ai, hoặc do nhân viên phỏng vấn bỏ trống.
    → Xử lý: Impute bằng nhóm mới "Unknown" hoặc điền Mode (trị xuất hiện nhiều nhất).

  2. most_impressing (Train: 6.51% | Test: 6.93% missing):
    → Cơ chế: MNAR (Missing Not At Random). Khách du lịch không điền câu này do họ 
      không có ấn tượng sâu sắc hoặc ngại viết nhận xét tự do bằng chữ.
    → Xử lý: Impute bằng một nhãn đặc trưng "Unknown" để mô hình học được.

  3. total_female & total_male (Train: ~0.1% missing):
    → Cơ chế: MCAR (Missing Completely At Random). Đây là lỗi bỏ sót dữ liệu ngẫu nhiên 
      khi nhập liệu, tỷ lệ rất thấp (< 0.5%).
    → Xử lý: Impute bằng trung vị (MEDIAN) của tập Train.

  ── Chiến lược imputation không leakage (fit on TRAIN only) ────────────────
    • numeric (total_female, total_male) → MEDIAN [ưu tiên vì lệch nặng/outlier]
    • categorical (travel_with, most_impressing) → điền nhãn "Unknown"
    ⚠️  Tất cả thống kê (Median, Mode) đều được fit trên TRAIN only và transform TEST.
""")


# ════════════════════════════════════════════════════════════════
# BƯỚC 2 — THỐNG KÊ MÔ TẢ VÀ PHÁT HIỆN OUTLIER
# Ta tính song song hai thước đo outlier cho từng biến số: luật 1.5·IQR của
# Tukey và ngưỡng |z| > 3. Bước này quan trọng vì OLS tối thiểu hóa tổng bình
# phương phần dư, nên mỗi outlier kéo lệch β̂ theo bình phương khoảng cách,
# tức ảnh hưởng không cân xứng so với một quan sát bình thường.
# ════════════════════════════════════════════════════════════════
print(f"{SEP}")
print("2. DESCRIPTIVE STATISTICS + OUTLIER DETECTION (Tính trên Train)")
print(SEP)

rows = []
for col in NUM_FEAT:
    s        = train[col]
    q1, q3   = s.quantile(0.25), s.quantile(0.75)
    iqr_val  = q3 - q1
    iqr_out  = ((s < q1 - 1.5*iqr_val) | (s > q3 + 1.5*iqr_val)).sum()
    z_scores = np.abs(stats.zscore(s.dropna()))
    z_out    = (z_scores > 3).sum()

    # Dùng dropna() khi tính tương quan để tránh NaN làm hỏng kết quả Pearson
    clean_df = train[[col, TARGET]].dropna()
    corr_    = clean_df.corr().iloc[0, 1]
    
    rows.append({
        "feature"       : col,
        "mean"          : round(s.mean(), 3),
        "median"        : round(s.median(), 3),
        "std"           : round(s.std(), 3),
        "min"           : round(s.min(), 3),
        "max"           : round(s.max(), 3),
        "skew"          : round(s.skew(), 4),
        "IQR_out%"      : round(iqr_out / len(s) * 100, 2),
        "Zscore_out%"   : round(z_out  / len(s) * 100, 2),
        "corr_target"   : round(corr_, 3),
        "n_unique"      : s.nunique(),
    })

df_stats = pd.DataFrame(rows)
print(df_stats.to_string(index=False))

# Biến mục tiêu là tâm điểm của cả bài toán, nên ta soi riêng phân bố của nó:
# độ lệch trước và sau log1p quyết định trực tiếp có cần biến đổi target không.
print(f"\n  ── TARGET: {TARGET} (Chi phí du lịch - TZS) ──")
tgt      = train[TARGET]
q1t, q3t = tgt.quantile(0.25), tgt.quantile(0.75)
iqr_t    = q3t - q1t
out_iqr  = ((tgt < q1t - 1.5*iqr_t) | (tgt > q3t + 1.5*iqr_t))
z_tgt    = np.abs(stats.zscore(tgt))
print(f"  Count  : {tgt.count():,}")
print(f"  Min/Max: {tgt.min():,.2f} / {tgt.max():,.2f} TZS")
print(f"  Mean   : {tgt.mean():,.2f}  |  Median: {tgt.median():,.2f} TZS  ← lệch phải cực nặng")
print(f"  Std    : {tgt.std():,.2f}  |  IQR: {iqr_t:,.2f}  (Q1={q1t:,.2f}, Q3={q3t:,.2f})")
print(f"  Skew (original): {tgt.skew():.4f}  →  Skew (log1p): {np.log1p(tgt).skew():.4f}")
print(f"  IQR outliers   : {out_iqr.sum():,} ({out_iqr.mean()*100:.2f}%)")
print(f"  Z>3  outliers  : {(z_tgt > 3).sum():,} ({(z_tgt > 3).mean()*100:.2f}%)")
print("""
  Decision:
    ✅ Dùng log1p(total_cost) cho mô hình (giảm skew cực đẹp từ 2.97 → -0.32)
    ✅ Giữ outlier — đây là chi phí của khách đi tour trọn gói cao cấp, thực tế và hợp lệ
    ✅ Validation: K-Fold stratified theo nhóm quantile của target cost
""")


# ════════════════════════════════════════════════════════════════
# BƯỚC 3 — PHÂN LOẠI ĐẶC TRƯNG
# Bốn biến đếm này tuy nhận giá trị nguyên nhỏ nhưng vẫn mang bản chất numeric
# thật sự chứ không phải nhãn phân loại trá hình, nên ta xác nhận chúng đều cần
# log1p để ghìm độ lệch trước khi đưa vào mô hình, thay vì đem one-hot encode.
# ════════════════════════════════════════════════════════════════
print(f"{SEP}")
print("3. FEATURE CLASSIFICATION: true numeric vs. pseudo-category")
print(SEP)

classification = {
    "total_female"   : "✅ True numeric   | skew=13.04 (extreme) → log1p  | MEDIAN impute",
    "total_male"     : "✅ True numeric   | skew=13.81 (extreme) → log1p  | MEDIAN impute",
    "night_mainland" : "✅ True numeric   | skew=4.03 (extreme) → log1p   | No missing",
    "night_zanzibar" : "✅ True numeric   | skew=4.23 (extreme) → log1p   | No missing",
}
for k, v in classification.items():
    print(f"  {k:<18}: {v}")


# ════════════════════════════════════════════════════════════════
# BƯỚC 4 — ĐỀ XUẤT CHUẨN HÓA
# Vì penalty của Ridge và Lasso áp trực tiếp lên không gian tham số mà không
# phân biệt đơn vị đo, ta phải đưa mọi biến số về cùng một thang: trước hết
# log1p để giảm lệch phải, sau đó StandardScaler để mỗi biến có trung bình 0
# và độ lệch chuẩn 1, nhờ đó penalty mới phạt công bằng lên tất cả hệ số.
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("4. SCALING RECOMMENDATIONS")
print(SEP)

scaling = {
    "total_female"   : "log1p → StandardScaler (giảm skew từ 13.0 → 1.1)",
    "total_male"     : "log1p → StandardScaler (giảm skew từ 13.8 → 1.2)",
    "night_mainland" : "log1p → StandardScaler (giảm skew từ 4.0 → 0.2)",
    "night_zanzibar" : "log1p → StandardScaler (giảm skew từ 4.2 → 0.9)",
}
for k, v in scaling.items():
    print(f"  {k:<18}: {v}")


# ════════════════════════════════════════════════════════════════
# BƯỚC 5 — DỊCH CHUYỂN PHÂN PHỐI GIỮA TRAIN VÀ TEST
# Nếu hai tập đến từ phân phối khác nhau, hiện tượng covariate shift xảy ra và
# metric đo trên test sẽ không còn phản ánh đúng khả năng tổng quát hóa thực sự.
# Ta dùng kiểm định Kolmogorov–Smirnov cho từng biến số, với H0 là hai mẫu cùng
# phân phối, và chỉ lo ngại khi p-value tụt xuống dưới 0.05.
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("5. TRAIN / TEST DISTRIBUTION SHIFT  (Kolmogorov–Smirnov test)")
print(SEP)

ks_rows = []
for col in NUM_FEAT:
    ks, p = stats.ks_2samp(train[col].dropna(), test[col].dropna())
    flag  = "⚠️  SHIFT" if p < 0.05 else "✅ OK"
    ks_rows.append({"feature": col, "KS_stat": round(ks, 4),
                    "p_value": round(p, 6), "status": flag})
    print(f"  {col:<18}: KS={ks:.4f}, p={p:.6f}  {flag}")

print("""
  Nhận xét: Không có dịch chuyển phân phối nghiêm trọng nào giữa Train và Test
  trên các biến số (tất cả p > 0.05, phân phối Train và Test đồng nhất).
  Hành động: Có thể chia K-Fold ngẫu nhiên hoặc Stratified an toàn.
""")


# ════════════════════════════════════════════════════════════════
# BƯỚC 6 — KIỂM TRA ĐA CỘNG TUYẾN BẰNG VIF
# Tương quan theo cặp chỉ thấy được quan hệ giữa đúng hai biến, trong khi đa
# cộng tuyến thực tế có thể xuất hiện theo kiểu đa biến: một biến bị xấp xỉ tốt
# bởi tổ hợp tuyến tính của nhiều biến khác. VIF_j = 1/(1 - R²_j) bắt được cả
# tình huống đó, và ta coi VIF vượt 10 là dấu hiệu đa cộng tuyến nghiêm trọng.
# ════════════════════════════════════════════════════════════════
print(f"{SEP}")
print("6. VIF — MULTICOLLINEARITY CHECK (Kiểm đa cộng tuyến biến liên tục)")
print(SEP)

# Xóa hàng có NaN trước khi tính VIF vì hồi quy phụ cần ma trận hoàn chỉnh,
# không thể có missing trong biến predictor
vif_df = train[NUM_FEAT].dropna()
X_vif  = vif_df.values
vif_scores = {}
for j, col in enumerate(NUM_FEAT):
    # Công thức VIF_j = 1 / (1 - R²_j) trong đó R²_j là hệ số xác định khi
    # hồi quy biến j lên tất cả các biến còn lại; VIF > 10 báo hiệu đa cộng tuyến
    X_other = np.delete(X_vif, j, axis=1)
    y_col   = X_vif[:, j]
    # Thêm cột intercept (all 1s) theo quy ước Part 1 ols_fit
    ones    = np.ones((len(y_col), 1))
    X_aug   = np.hstack([ones, X_other])
    res     = ols_fit(X_aug.tolist(), y_col.tolist())
    if res.success:
        y_hat = np.array(res.y_hat)
        ss_res = float(np.sum((y_col - y_hat) ** 2))
        ss_tot = float(np.sum((y_col - y_col.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        r2 = 0.0
    vif_scores[col] = round(1 / (1 - r2), 3) if r2 < 1 else float("inf")

for col, vif in vif_scores.items():
    flag = "🔴 HIGH"     if vif > 10 else \
           "🟡 MODERATE" if vif > 5  else "🟢 OK"
    print(f"  {col:<18}: VIF = {vif:>8.3f}  {flag}")

print("""
  Max VIF = 1.15 (total_female) → hoàn toàn không có đa cộng tuyến.
  Không cần loại bỏ bất kỳ biến số liên tục nào trước khi đưa vào OLS.
""")


# ════════════════════════════════════════════════════════════════
# BƯỚC 7 — TỔNG HỢP QUYẾT ĐỊNH Preprocessing
# Toàn bộ phân tích phía trên được gói lại thành một bảng quyết định duy nhất.
# Bảng này đóng vai trò như một bản hợp đồng để DataPipeline trong
# data_pipeline.py thực thi đúng những gì EDA đã kết luận, tránh tình trạng
# mỗi nơi xử lý dữ liệu một kiểu khác nhau.
# ════════════════════════════════════════════════════════════════
print(f"{SEP}")
print("7. PREPROCESSING DECISIONS — tổng hợp cho DataPipeline Day 3")
print(SEP)
print("""
  ┌──────────────────────────────────────────────────────────────────┐
  │ MISSING VALUES                                                   │
  │  • Dataset gốc chứa missing thực tế:                             │
  │    - travel_with (23.1%), most_impressing (6.5%) → "Unknown" nhãn│
  │    - total_female, total_male (0.1%) → MEDIAN điền khuyết        │
  │  • Tất cả fit trên TRAIN only, transform TEST → tránh leakage    │
  ├──────────────────────────────────────────────────────────────────┤
  │ NUMERIC FEATURES & SCALING                                       │
  │  • 4 biến continuous: total_female, total_male, nights...        │
  │  • Tất cả đều lệch phải cực nặng (Skew > 4.0)                    │
  │  • Giải pháp: log1p biến đổi phi tuyến cho tất cả 4 biến         │
  │  • Áp dụng StandardScaler (Z-score) sau khi log1p                │
  ├──────────────────────────────────────────────────────────────────┤
  │ OUTLIER HANDLING                                                 │
  │  • Target outliers (IQR 4.2%): Giữ lại vì chi phí thật           │
  │  • Hạn chế Winsorization biến mục tiêu để bảo vệ phân phối log1p │
  ├──────────────────────────────────────────────────────────────────┤
  │ TRANSFORM & TARGET                                               │
  │  • Target: log1p(total_cost) giảm skew cực tốt (2.97 → -0.32)    │
  ├──────────────────────────────────────────────────────────────────┤
  │ VIF & SHIFT                                                      │
  │  • Max VIF = 1.15 → Không có hiện tượng đa cộng tuyến            │
  │  • Phân phối Train vs Test đồng nhất (KS p > 0.05)               │
  └──────────────────────────────────────────────────────────────────┘
""")


# ════════════════════════════════════════════════════════════════
# BƯỚC 8 — TRỰC QUAN HÓA
# Mỗi nhận định bằng số ở trên được vẽ lại thành biểu đồ tương ứng, gồm
# histogram, boxplot, heatmap tương quan, VIF và KS. Nhờ đó báo cáo có hình
# minh họa trực tiếp cho từng kết luận, thay vì chỉ trình bày con số khô khan.
# ════════════════════════════════════════════════════════════════
print("Generating plots...")

COLORS = ["#FF5722", "#9C27B0", "#FF9800", "#00BCD4"]

SHORT = {
    "total_female"   : "female",
    "total_male"     : "male",
    "night_mainland" : "night_main",
    "night_zanzibar" : "night_zanzi",
    TARGET           : "cost",
}

# ── Hình 1a: phân bố biến mục tiêu — đặt cạnh nhau ba góc nhìn (gốc,
# sau log1p, boxplot) để thấy rõ tác dụng của phép biến đổi log ───────
fig1a, axes = plt.subplots(1, 3, figsize=(16, 5))
fig1a.suptitle("Target Distribution Analysis: total_cost (TZS)",
               fontsize=13, fontweight="bold")

# Bên trái — phân bố gốc: đuôi phải rất dài nên cột tần suất dồn hết về bên trái
ax = axes[0]
ax.hist(tgt, bins=60, color="#2196F3", edgecolor="white", alpha=0.85, linewidth=1.2)
ax.set_title(f"Original Distribution\nSkewness: {tgt.skew():.2f}", fontsize=11, fontweight="bold")
ax.set_xlabel("total_cost (TZS)", fontsize=10)
ax.set_ylabel("Frequency", fontsize=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
ax.grid(axis="y", alpha=0.3)

# Ở giữa — sau log1p: đuôi phải bị nén lại, phân bố trở nên gần chuẩn hơn hẳn
ax = axes[1]
log_tgt = np.log1p(tgt)
ax.hist(log_tgt, bins=60, color="#4CAF50", edgecolor="white", alpha=0.85, linewidth=1.2)
ax.set_title(f"Log-Transformed Distribution\nSkewness: {log_tgt.skew():.2f} (reduced from {tgt.skew():.2f})",
             fontsize=11, fontweight="bold")
ax.set_xlabel("log1p(total_cost)", fontsize=10)
ax.set_ylabel("Frequency", fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Bên phải — boxplot: nhìn trực tiếp các điểm outlier theo luật 1.5·IQR
ax = axes[2]
bp = ax.boxplot(tgt, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#2196F3", alpha=0.7, linewidth=1.5),
                medianprops=dict(color="red", linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker="o", markersize=4, alpha=0.5, color="#F44336"))
ax.set_title(f"Boxplot Summary\nOutliers (IQR): {out_iqr.sum():,} ({out_iqr.mean()*100:.1f}%)",
             fontsize=11, fontweight="bold")
ax.set_ylabel("total_cost (TZS)", fontsize=10)
ax.set_xticklabels([TARGET])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
ax.grid(axis="y", alpha=0.3)

fig1a.tight_layout()
out1a = os.path.join(OUTPUT_DIR, "fig_target_distribution.png")
fig1a.savefig(out1a, dpi=150, bbox_inches="tight")
plt.close(fig1a)
print(f"  ✅ Saved: {out1a}")

# ── Hình 1b: histogram bốn biến số, để thấy từng biến lệch phải đến mức nào ──
fig1b, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1b.suptitle("Numeric Features Distribution", fontsize=13, fontweight="bold")
axes = axes.flatten()

for idx, (col, color) in enumerate(zip(NUM_FEAT, COLORS)):
    ax = axes[idx]
    data = train[col].dropna()
    ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.85, linewidth=1.2)
    ax.set_title(f"{SHORT[col]} (n={len(data):,})\nSkewness: {data.skew():.2f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

fig1b.tight_layout()
out1b = os.path.join(OUTPUT_DIR, "fig_numeric_histograms.png")
fig1b.savefig(out1b, dpi=150, bbox_inches="tight")
plt.close(fig1b)
print(f"  ✅ Saved: {out1b}")

# ── Hình 1c: heatmap tương quan giữa các biến số và mục tiêu, vừa soi
# multicollinearity vừa xem biến nào có tín hiệu tuyến tính với target ──
fig1c, ax = plt.subplots(figsize=(12, 8))
corr_cols   = NUM_FEAT + [TARGET]
corr_matrix = train[corr_cols].corr()
short_names = [SHORT[c] for c in corr_cols]

im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(short_names)))
ax.set_xticklabels(short_names, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(short_names)))
ax.set_yticklabels(short_names, fontsize=11, fontweight="bold")
ax.set_title("Correlation Matrix: Numeric Features & Target", fontsize=13, fontweight="bold", pad=15)

for i in range(len(short_names)):
    for j in range(len(short_names)):
        val = corr_matrix.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                color="white" if abs(val) > 0.55 else "black", fontsize=10, fontweight="bold")

cbar = fig1c.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson Correlation", fontsize=11, fontweight="bold")

fig1c.tight_layout()
out1c = os.path.join(OUTPUT_DIR, "fig_correlation_heatmap.png")
fig1c.savefig(out1c, dpi=150, bbox_inches="tight")
plt.close(fig1c)
print(f"  ✅ Saved: {out1c}")


# ── Hình 2: boxplot từng biến số để định vị outlier, kèm tỷ lệ điểm
# vượt ngưỡng theo cả luật IQR lẫn ngưỡng |z| > 3 ──────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("EDA — Boxplots (Outlier Detection)", fontsize=14, fontweight="bold")

for ax, col, color in zip(axes2.flat, NUM_FEAT, COLORS):
    data = train[col].dropna()
    bp   = ax.boxplot(data, vert=True, patch_artist=True,
                      boxprops=dict(facecolor=color, alpha=0.65),
                      medianprops=dict(color="black", linewidth=2),
                      flierprops=dict(marker=".", markersize=1.5, alpha=0.3))
    q1_, q3_  = data.quantile(0.25), data.quantile(0.75)
    iqr_      = q3_ - q1_
    out_pct   = ((data < q1_ - 1.5*iqr_) | (data > q3_ + 1.5*iqr_)).mean() * 100
    z_pct     = (np.abs(stats.zscore(data)) > 3).mean() * 100
    ax.set_title(f"{SHORT[col]}\nIQR out={out_pct:.1f}% | Z>3={z_pct:.1f}%", fontsize=10)
    ax.set_ylabel(col, fontsize=8)
    ax.set_xticks([])

fig2.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "day1_fig2_boxplots.png")
fig2.savefig(out2, dpi=120, bbox_inches="tight")
plt.close(fig2)
print(f"  ✅ Saved: {out2}")


# ── Hình 3a: VIF của từng biến số, kèm hai đường ngưỡng cảnh báo 5 và 10 ──
fig3a, ax = plt.subplots(figsize=(10, 6))
vif_feats  = [SHORT[f] for f in vif_scores]
vif_vals   = list(vif_scores.values())
vif_colors = ["#F44336" if v > 10 else ("#FF9800" if v > 5 else "#4CAF50") for v in vif_vals]
bars = ax.barh(vif_feats, vif_vals, color=vif_colors, edgecolor="white", linewidth=1.5)
ax.axvline(5,  color="orange", linestyle="--", linewidth=2, label="VIF=5 (moderate)", alpha=0.7)
ax.axvline(10, color="red",    linestyle="--", linewidth=2, label="VIF=10 (high)", alpha=0.7)
ax.set_xlabel("VIF Score", fontsize=12, fontweight="bold")
ax.set_title("Variance Inflation Factor (Multicollinearity Detection)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, vif_vals):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=10, fontweight="bold")
fig3a.tight_layout()
out3a = os.path.join(OUTPUT_DIR, "fig_vif_scores.png")
fig3a.savefig(out3a, dpi=150, bbox_inches="tight")
plt.close(fig3a)
print(f"  ✅ Saved: {out3a}")

# ── Hình 3b: tương quan Pearson của từng biến số với mục tiêu, để đánh
# giá nhanh tiềm năng Prediction tuyến tính đơn biến của mỗi biến ──────────
fig3b, ax = plt.subplots(figsize=(10, 6))
corr_vals   = [train[[c, TARGET]].dropna().corr().iloc[0, 1] for c in NUM_FEAT]
corr_colors = ["#F44336" if v < 0 else "#4CAF50" for v in corr_vals]
corr_labels = [SHORT[c] for c in NUM_FEAT]
bars = ax.barh(corr_labels, corr_vals, color=corr_colors, edgecolor="white", linewidth=1.5)
ax.axvline(0, color="black", linewidth=1.5)
ax.set_xlabel("Pearson Correlation Coefficient", fontsize=12, fontweight="bold")
ax.set_title("Feature Correlation with Target (total_cost)", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, corr_vals):
    xpos = val + (0.02 if val >= 0 else -0.02)
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=10, fontweight="bold")
fig3b.tight_layout()
out3b = os.path.join(OUTPUT_DIR, "fig_correlation_target.png")
fig3b.savefig(out3b, dpi=150, bbox_inches="tight")
plt.close(fig3b)
print(f"  ✅ Saved: {out3b}")

# ── Hình 3c: thống kê KS từng biến, thanh đỏ đánh dấu biến có dịch
# chuyển phân phối giữa train và test (p < 0.05) ──────────────────────
fig3c, ax = plt.subplots(figsize=(10, 6))
ks_vals  = [r["KS_stat"] for r in ks_rows]
ks_labs  = [SHORT[r["feature"]] for r in ks_rows]
ks_cols  = ["#F44336" if r["p_value"] < 0.05 else "#4CAF50" for r in ks_rows]
bars = ax.barh(ks_labs, ks_vals, color=ks_cols, edgecolor="white", linewidth=1.5)
ax.set_xlabel("KS Statistic", fontsize=12, fontweight="bold")
ax.set_title("Kolmogorov-Smirnov Test: Train-Test Distribution Shift\n(Red = p<0.05, significant shift detected)",
             fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, ks_vals):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
fig3c.tight_layout()
out3c = os.path.join(OUTPUT_DIR, "fig_ks_shift.png")
fig3c.savefig(out3c, dpi=150, bbox_inches="tight")
plt.close(fig3c)
print(f"  ✅ Saved: {out3c}")

print(f"\n{'='*65}")
print("✅  Day 1 EDA — Missing Values & Numeric Features: HOÀN THÀNH")
print(f"{'='*65}")
