"""
Module phân tích SHAP (SHapley Additive exPlanations) cho mô hình Ridge Regression
trên bộ dữ liệu Tanzania Tourism Expenditure, cung cấp giải thích về tầm quan
trọng đặc trưng vượt ra ngoài phân tích hệ số thông thường.

SHAP dựa trên lý thuyết trò chơi Shapley: giá trị SHAP của một đặc trưng đối
với một mẫu cụ thể là phần đóng góp biên của đặc trưng đó vào giá trị dự đoán,
tính trung bình qua tất cả các hoán vị có thể của tập đặc trưng. Điều này cho
phép giải thích mô hình ở cả cấp độ toàn cục (feature importance trung bình)
lẫn cấp độ cục bộ (tại sao mô hình đưa ra dự đoán cụ thể cho một mẫu). SHAP
được ưu tiên hơn phân tích hệ số thuần túy vì nó phản ánh đúng tầm quan trọng
ngay cả khi các đặc trưng có tương quan với nhau.

Thư viện shap.LinearExplainer được chọn vì cho phép tính SHAP value chính xác
cho mô hình tuyến tính (không cần xấp xỉ như TreeExplainer hay KernelExplainer),
sử dụng phân phối nền (background distribution) từ 200 mẫu ngẫu nhiên của
tập huấn luyện.

Module tạo ba biểu đồ trong thư mục outputs/:
  fig_shap_summary.png    — Beeswarm: phân phối giá trị SHAP theo từng đặc trưng,
                            màu sắc biểu thị giá trị đặc trưng (thấp/cao).
  fig_shap_importance.png — Bar chart: mean |SHAP| feature importance (top 20).
  fig_shap_waterfall.png  — Waterfall: giải thích cục bộ cho 3 mẫu dự đoán
                            điển hình (chi phí thấp, trung bình và cao).

Cách chạy: python shap_analysis.py (từ thư mục part2/src/)
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, SCRIPT_DIR)

# ─── load & preprocess ───────────────────────────────────────────────────────
print("=" * 65)
print("SHAP Analysis: Tanzania Tourism — Ridge Regression")
print("=" * 65)

from data_pipeline import DataPipeline, PipelineConfig

config = PipelineConfig(data_dir=DATA_DIR, missing_method="mean")
result = DataPipeline(config).run()

X_train       = result.X_train.copy()
y_train       = result.y_train.copy()
feature_names = result.feature_names

# Điền NaN còn sót lại (nếu có) bằng 0 để LinearExplainer không bị lỗi;
# lý do dùng fill_value=0 thay vì mean: sau khi StandardScaler thì mean = 0
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="constant", fill_value=0.0)
X_train = imp.fit_transform(X_train)

print(f"  X_train: {X_train.shape}  |  NaN remaining: {np.isnan(X_train).sum()}")

# ─── fit Ridge ───────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge

LAMBDA = 100.0
ridge  = Ridge(alpha=LAMBDA, fit_intercept=False)
ridge.fit(X_train, y_train)
y_hat = ridge.predict(X_train)

rss = np.sum((y_train - y_hat) ** 2)
tss = np.sum((y_train - np.mean(y_train)) ** 2)
print(f"  Ridge R²: {1 - rss/tss:.4f}")

# ─── SHAP ────────────────────────────────────────────────────────────────────
print("\nComputing SHAP values (LinearExplainer)…")

# Lấy 200 mẫu ngẫu nhiên làm background distribution để xác định baseline
# (giá trị kỳ vọng E[f(X)] được tính trên background này)
idx_bg    = np.random.choice(len(X_train), 200, replace=False)
X_bg      = X_train[idx_bg]
explainer = shap.LinearExplainer(ridge, X_bg)

# Tính SHAP values trên 300 mẫu ngẫu nhiên để đủ đại diện mà không quá chậm
idx_samp    = np.random.choice(len(X_train), 300, replace=False)
X_samp      = X_train[idx_samp]
shap_vals   = explainer.shap_values(X_samp)          # ndarray (300, n_features)

mean_abs    = np.abs(shap_vals).mean(axis=0)
top20_idx   = np.argsort(mean_abs)[::-1][:20]
top20_names = [feature_names[i] for i in top20_idx]
top20_vals  = mean_abs[top20_idx]

print(f"  SHAP values: {shap_vals.shape}")
print(f"  Top feature: {top20_names[0]}  mean|SHAP|={top20_vals[0]:,.0f}")

# ─── Figure 1: Beeswarm (manual via scatter) ──────────────────────────────────
print("\nGenerating fig_shap_summary.png …")

sv_top20   = shap_vals[:, top20_idx]          # (300, 20)
feat_top20 = X_samp[:, top20_idx]             # (300, 20)

fig1, ax1 = plt.subplots(figsize=(13, 9))

n_feat = 20
y_pos  = np.arange(n_feat)

for fi in range(n_feat):
    sv_col  = sv_top20[:, fi]
    ft_col  = feat_top20[:, fi]

    # Chuẩn hóa giá trị đặc trưng về [0,1] để ánh xạ lên colormap RdBu_r:
    # màu đỏ = giá trị đặc trưng cao, màu xanh = giá trị đặc trưng thấp
    ft_min, ft_max = ft_col.min(), ft_col.max()
    ft_norm = (ft_col - ft_min) / (ft_max - ft_min + 1e-9)

    colors_ = plt.cm.RdBu_r(ft_norm)

    # Thêm jitter theo chiều y để hiển thị mật độ điểm (beeswarm effect)
    # mà không bị chồng chéo khi nhiều điểm có cùng SHAP value
    jitter  = np.random.uniform(-0.3, 0.3, size=len(sv_col))
    ax1.scatter(sv_col, (n_feat - 1 - fi) + jitter,
                c=colors_, s=12, alpha=0.6, linewidths=0)

ax1.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top20_names[::-1], fontsize=9)
ax1.set_xlabel("SHAP Value (impact on model output in TZS)", fontsize=11, fontweight="bold")
ax1.set_title(
    "SHAP Summary Plot — Top 20 Features\n"
    "Ridge Regression (λ=100) · Tanzania Tourism",
    fontsize=13, fontweight="bold",
)
ax1.grid(axis="x", alpha=0.25, linestyle="--")

# Colorbar legend
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig1.colorbar(sm, ax=ax1, fraction=0.025, pad=0.02)
cbar.set_label("Feature Value (low → high)", fontsize=10)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["Low", "Mid", "High"])

fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "fig_shap_summary.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  ✅ {out1}")

# ─── Figure 2: Bar — Mean |SHAP| top 20 ──────────────────────────────────────
print("Generating fig_shap_importance.png …")

cmap   = plt.cm.get_cmap("RdYlGn_r", 20)
colors = [cmap(i / 19) for i in range(20)]

fig2, ax2 = plt.subplots(figsize=(12, 8))
bars = ax2.barh(
    top20_names[::-1], top20_vals[::-1],
    color=colors[::-1], edgecolor="white", linewidth=1.1,
)
ax2.set_xlabel("Mean |SHAP Value| (average impact on prediction, TZS)",
               fontsize=11, fontweight="bold")
ax2.set_title(
    "Feature Importance via SHAP — Top 20 Features\n"
    "Ridge Regression (λ=100) · Tanzania Tourism",
    fontsize=13, fontweight="bold",
)
ax2.grid(axis="x", alpha=0.3, linestyle="--")
for bar, val in zip(bars, top20_vals[::-1]):
    ax2.text(val + max(top20_vals) * 0.005, bar.get_y() + bar.get_height() / 2,
             f"{val:,.0f}", va="center", fontsize=8.5, fontweight="bold")
fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "fig_shap_importance.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  ✅ {out2}")

# ─── Figure 3: Waterfall (manual) for 3 samples ──────────────────────────────
print("Generating fig_shap_waterfall.png …")

y_samp  = y_hat[idx_samp]
base    = explainer.expected_value

# Chọn ba mẫu đại diện: chi phí thấp nhất, gần trung vị nhất và cao nhất,
# để waterfall chart phủ được toàn bộ phổ dự đoán và giải thích đa dạng
low_i  = int(np.argmin(y_samp))
mid_i  = int(np.argmin(np.abs(y_samp - np.median(y_samp))))
high_i = int(np.argmax(y_samp))

def waterfall_ax(ax, sv_row, feat_row, feat_names, base_val, pred_val,
                 title, n_show=10):
    """Vẽ biểu đồ waterfall thể hiện đóng góp cục bộ của từng đặc trưng cho một dự đoán.

    Waterfall chart minh họa quá trình đi từ giá trị kỳ vọng base_val (trung
    bình dự đoán trên background) đến giá trị dự đoán cụ thể pred_val bằng
    cách cộng dần từng giá trị SHAP theo thứ tự tăng dần. Màu đỏ biểu thị
    đóng góp dương (đặc trưng làm tăng dự đoán), màu xanh biểu thị đóng góp
    âm (đặc trưng làm giảm dự đoán).

    Args:
        ax: Đối tượng matplotlib Axes sẽ vẽ lên.
        sv_row: Vector SHAP values của một mẫu cụ thể, hình dạng (n_top_features,).
        feat_row: Vector giá trị đặc trưng tương ứng của mẫu, hình dạng (n_top_features,).
        feat_names: Danh sách tên các top features tương ứng với sv_row.
        base_val: Giá trị kỳ vọng (baseline) của mô hình, tính từ background set.
        pred_val: Giá trị dự đoán thực tế của mẫu này (đơn vị TZS).
        title: Tiêu đề hiển thị trên biểu đồ con.
        n_show: Số lượng đặc trưng quan trọng nhất cần hiển thị, mặc định 10.
    """
    # Chọn n đặc trưng có |SHAP| lớn nhất từ tập top20 đã lọc trước
    order     = np.argsort(np.abs(sv_row))[::-1][:n_show]
    sv_show   = sv_row[order]
    nm_show   = [feat_names[i] for i in order]

    # Sắp xếp tăng dần để biểu đồ đọc từ dưới lên theo thứ tự ảnh hưởng
    sort_ord  = np.argsort(sv_show)
    sv_show   = sv_show[sort_ord]
    nm_show   = [nm_show[i] for i in sort_ord]

    cumulative = base_val + np.cumsum(sv_show)
    starts     = np.concatenate([[base_val], cumulative[:-1]])
    colors_    = ["#d73027" if v > 0 else "#4575b4" for v in sv_show]

    y_pos_ = np.arange(n_show)
    ax.barh(y_pos_, sv_show, left=starts, color=colors_,
            edgecolor="white", linewidth=0.8, height=0.6)

    ax.axvline(base_val, color="gray", linewidth=1.2, linestyle="--", alpha=0.6, label=f"Base: {base_val:,.0f}")
    ax.axvline(pred_val, color="black", linewidth=1.8, linestyle="-",  alpha=0.8, label=f"Pred: {pred_val:,.0f}")

    ax.set_yticks(y_pos_)
    ax.set_yticklabels(nm_show, fontsize=8.5)
    ax.set_xlabel("Prediction (TZS)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    pos_patch = mpatches.Patch(color="#d73027", label="Positive contribution")
    neg_patch = mpatches.Patch(color="#4575b4", label="Negative contribution")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right")

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 8))

for ax, si, label in zip(axes3,
                          [low_i, mid_i, high_i],
                          ["Low-cost", "Median-cost", "High-cost"]):
    waterfall_ax(
        ax,
        sv_row     = sv_top20[si],
        feat_row   = feat_top20[si],
        feat_names = top20_names,
        base_val   = float(base),
        pred_val   = float(y_samp[si]),
        title      = f"{label} prediction\n{y_samp[si]:,.0f} TZS",
    )

fig3.suptitle(
    "SHAP Waterfall — Local Explanation: 3 Individual Predictions\n"
    "(Top 10 features contributing to each prediction vs. expected baseline)",
    fontsize=13, fontweight="bold",
)
fig3.tight_layout()
out3 = os.path.join(OUT_DIR, "fig_shap_waterfall.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  ✅ {out3}")

# ─── Summary table ────────────────────────────────────────────────────────────
print("\nTop 20 Features by Mean |SHAP|:")
print(f"  {'Rank':<5} {'Feature':<42} {'Mean |SHAP| (TZS)':>18}")
print("  " + "-" * 67)
for rank, (name, val) in enumerate(zip(top20_names, top20_vals), 1):
    print(f"  {rank:<5} {name:<42} {val:>18,.2f}")

print(f"\n{'='*65}")
print("✅  SHAP Analysis: HOÀN THÀNH")
print(f"{'='*65}")
