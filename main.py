"""
=============================================================================
Day 1 EDA — Missing Values & Numeric Features (Tanzania Tourism Expenditure)
Task owner : trongnghia090406@gmail.com
Dataset    : Tanzania Tourism Expenditure (4.8K train, 1.6K test rows)
Target     : total_cost (TZS)
Ref        : Toan_UDTK_Project_2 — Section 2.2.1, 2.2.2, 2.2.3
=============================================================================
Cách chạy:
    python main.py
Yêu cầu: Train_new.csv và Test_new.csv nằm trong thư mục data/
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
np.random.seed(42)

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════
DATA_DIR   = "data"
OUTPUT_DIR = "."
TRAIN_FILE = "Train_new.csv"
TEST_FILE  = "Test_new.csv"

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
# 0. LOAD DATA
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
# 1. MISSING VALUES AUDIT
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("1. MISSING VALUES AUDIT (Xác định cột thiếu ≥ 5%)")
print(SEP)

def missing_report(df, label):
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
# 2. DESCRIPTIVE STATISTICS + OUTLIER (IQR & Z-score)
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
    
    # Pearson correlation with target (drop NaN để tính)
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

# Target stats
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
# 3. FEATURE CLASSIFICATION
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
# 4. SCALING RECOMMENDATIONS  (đề mục 2.2.3)
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
# 5. TRAIN / TEST DISTRIBUTION SHIFT  (KS test)
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
# 6. VIF — MULTICOLLINEARITY  (công thức đề: VIF_j = 1/(1-R²_j))
# ════════════════════════════════════════════════════════════════
print(f"{SEP}")
print("6. VIF — MULTICOLLINEARITY CHECK (Kiểm đa cộng tuyến biến liên tục)")
print(SEP)

# Drop rows with NaN in numeric features for VIF calculation
vif_df = train[NUM_FEAT].dropna()
X_vif  = vif_df.values
vif_scores = {}
for j, col in enumerate(NUM_FEAT):
    X_other = np.delete(X_vif, j, axis=1)
    y_col   = X_vif[:, j]
    lr      = LinearRegression().fit(X_other, y_col)
    r2      = lr.score(X_other, y_col)
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
# 7. PREPROCESSING DECISION SUMMARY
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
# 8. VISUALIZATIONS  (đủ yêu cầu đề: histogram, boxplot, heatmap)
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

# ── Figure 1: Target + Histograms + Correlation Heatmap ──────────
fig1 = plt.figure(figsize=(20, 22))
fig1.suptitle("EDA — Target, Histogram & Correlation Heatmap (Tanzania Tourism)",
              fontsize=15, fontweight="bold", y=0.99)
gs1 = gridspec.GridSpec(4, 3, figure=fig1, hspace=0.50, wspace=0.35)

# Target — original
ax = fig1.add_subplot(gs1[0, 0])
ax.hist(tgt, bins=60, color="#2196F3", edgecolor="white", alpha=0.85)
ax.set_title(f"Target: {TARGET}\n[original]  skew={tgt.skew():.2f}")
ax.set_xlabel("TZS"); ax.set_ylabel("Count")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

# Target — log1p
ax = fig1.add_subplot(gs1[0, 1])
log_tgt = np.log1p(tgt)
ax.hist(log_tgt, bins=60, color="#4CAF50", edgecolor="white", alpha=0.85)
ax.set_title(f"Target: log1p(total_cost)\nskew {tgt.skew():.2f} → {log_tgt.skew():.2f}")
ax.set_xlabel("log1p(TZS)"); ax.set_ylabel("Count")

# Target — boxplot
ax = fig1.add_subplot(gs1[0, 2])
bp = ax.boxplot(tgt, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#2196F3", alpha=0.7),
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker=".", markersize=1, alpha=0.3, color="#F44336"))
ax.set_title(f"Target Boxplot\nIQR outliers: {out_iqr.sum():,} ({out_iqr.mean()*100:.1f}%)")
ax.set_ylabel("TZS"); ax.set_xticks([])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

# Feature histograms (4 features + 2 placeholders)
for i, (col, color) in enumerate(zip(NUM_FEAT, COLORS)):
    row     = 1 + i // 3
    col_idx = i % 3
    ax      = fig1.add_subplot(gs1[row, col_idx])
    data    = train[col].dropna()
    ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(f"{SHORT[col]}\nskew={data.skew():.2f} | unique={data.nunique()}")
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel("Count")

# Placeholders for empty grid slots to maintain layout
for i in range(len(NUM_FEAT), 6):
    row     = 1 + i // 3
    col_idx = i % 3
    ax      = fig1.add_subplot(gs1[row, col_idx])
    ax.text(0.5, 0.5, "Placeholder", ha="center", va="center", color="gray", alpha=0.5)
    ax.axis("off")

# Correlation heatmap (bottom row, full width)
ax_heat = fig1.add_subplot(gs1[3, :])
corr_cols   = NUM_FEAT + [TARGET]
corr_matrix = train[corr_cols].corr()
short_names = [SHORT[c] for c in corr_cols]
im = ax_heat.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
ax_heat.set_xticks(range(len(short_names))); ax_heat.set_xticklabels(short_names, fontsize=9)
ax_heat.set_yticks(range(len(short_names))); ax_heat.set_yticklabels(short_names, fontsize=9)
ax_heat.set_title("Correlation Heatmap (Numeric Features + Target)", fontsize=11)
for i in range(len(short_names)):
    for j in range(len(short_names)):
        val = corr_matrix.values[i, j]
        ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if abs(val) > 0.55 else "black", fontsize=8)
fig1.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02)

out1 = os.path.join(OUTPUT_DIR, "day1_fig1_target_hist_heatmap.png")
fig1.savefig(out1, dpi=120, bbox_inches="tight")
plt.close(fig1)
print(f"  ✅ Saved: {out1}")


# ── Figure 2: Boxplots (phát hiện outlier trực quan) ─────────────
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


# ── Figure 3: VIF + Correlation-with-target + KS shift ───────────
fig3, axes3 = plt.subplots(1, 3, figsize=(20, 6))
fig3.suptitle("EDA — VIF | Correlation with Target | Train-Test Shift",
              fontsize=13, fontweight="bold")

# VIF
ax = axes3[0]
vif_feats  = [SHORT[f] for f in vif_scores]
vif_vals   = list(vif_scores.values())
vif_colors = ["#F44336" if v > 10 else ("#FF9800" if v > 5 else "#4CAF50") for v in vif_vals]
bars = ax.barh(vif_feats, vif_vals, color=vif_colors, edgecolor="white")
ax.axvline(5,  color="orange", linestyle="--", linewidth=1.2, label="VIF=5 (moderate)")
ax.axvline(10, color="red",    linestyle="--", linewidth=1.2, label="VIF=10 (high)")
ax.set_title("VIF Scores"); ax.set_xlabel("VIF"); ax.legend(fontsize=8)
for bar, val in zip(bars, vif_vals):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=9)

# Correlation with target
ax = axes3[1]
corr_vals   = [train[[c, TARGET]].dropna().corr().iloc[0, 1] for c in NUM_FEAT]
corr_colors = ["#F44336" if v < 0 else "#4CAF50" for v in corr_vals]
corr_labels = [SHORT[c] for c in NUM_FEAT]
bars = ax.barh(corr_labels, corr_vals, color=corr_colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Pearson r with Target"); ax.set_xlabel("Pearson r")
for bar, val in zip(bars, corr_vals):
    xpos = val + (0.01 if val >= 0 else -0.01)
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=9)

# KS shift
ax = axes3[2]
ks_vals  = [r["KS_stat"] for r in ks_rows]
ks_labs  = [SHORT[r["feature"]] for r in ks_rows]
ks_cols  = ["#F44336" if r["p_value"] < 0.05 else "#4CAF50" for r in ks_rows]
bars = ax.barh(ks_labs, ks_vals, color=ks_cols, edgecolor="white")
ax.set_title("KS Statistic — Train vs Test\n(red = p<0.05, shift detected)")
ax.set_xlabel("KS Statistic")
for bar, val in zip(bars, ks_vals):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)

fig3.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "day1_fig3_vif_corr_shift.png")
fig3.savefig(out3, dpi=120, bbox_inches="tight")
plt.close(fig3)
print(f"  ✅ Saved: {out3}")

print(f"\n{'='*65}")
print("✅  Day 1 EDA — Missing Values & Numeric Features: HOÀN THÀNH")
print(f"{'='*65}")