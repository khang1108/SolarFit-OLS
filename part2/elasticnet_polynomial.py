"""
=============================================================================
ElasticNet + Polynomial Regression — Tanzania Tourism Expenditure
Dataset    : Tanzania Tourism Expenditure (4.8K train rows)
Target     : total_cost (TZS)
Ref        : Report Section 3.9.2 — Đề xuất cải thiện (Elastic Net + Polynomial)
=============================================================================
Cách chạy:
    python elasticnet_polynomial.py
Yêu cầu: Train_new.csv nằm trong thư mục data/
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import (
    StandardScaler, PolynomialFeatures, LabelEncoder
)
from sklearn.linear_model import (
    LinearRegression, ElasticNet, ElasticNetCV,
    Ridge, Lasso
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
np.random.seed(42)

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════
DATA_DIR   = "data"
OUTPUT_DIR = "."
TRAIN_FILE = "Train_new.csv"

TARGET   = "total_cost"
ID_COL   = "ID"
NUM_FEAT = ["total_female", "total_male", "night_mainland", "night_zanzibar"]
CAT_FEAT = [
    "country", "age_group", "travel_with", "purpose",
    "main_activity", "info_source", "tour_arrangement",
    "package_transport_int", "package_accomodation", "package_food",
    "package_transport_tz", "package_sightseeing", "package_guided_tour",
    "package_insurance", "payment_mode", "first_trip_tz", "most_impressing",
]

SEP = "=" * 70
K_FOLDS = 5                # Số fold cho Cross-Validation
POLY_DEGREES = [2, 3]      # Các bậc Polynomial cần thử
RANDOM_STATE = 42

# ════════════════════════════════════════════════════════════════
# 0. LOAD & PREPROCESS (theo các quyết định Preprocessing từ eda.py)
# ════════════════════════════════════════════════════════════════
print(SEP)
print("0. LOADING & PREPROCESSING DATA")
print(SEP)

train = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
print(f"  Raw shape: {train.shape[0]:,} rows × {train.shape[1]} cols")

# ── Drop rows with missing target ───────────────────────────────
train = train.dropna(subset=[TARGET]).copy()

# ── Imputation ──────────────────────────────────────────────────
# Numeric: MEDIAN impute (fit on train only)
for col in NUM_FEAT:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)

# Categorical: "Unknown" impute (tất cả cột categorical)
for col in CAT_FEAT:
    train[col] = train[col].fillna("Unknown")

# ── Target transform: log1p ────────────────────────────────────
train["log_cost"] = np.log1p(train[TARGET])

# ── Encoding: Label Encoding cho categorical ───────────────────
label_encoders = {}
for col in CAT_FEAT:
    le = LabelEncoder()
    train[col + "_enc"] = le.fit_transform(train[col].astype(str))
    label_encoders[col] = le

CAT_ENC = [c + "_enc" for c in CAT_FEAT]

# ── Feature matrix ─────────────────────────────────────────────
FEATURES = NUM_FEAT + CAT_ENC
X = train[FEATURES].values.astype(float)
y = train["log_cost"].values  # log1p(total_cost) — giảm skew

# Safety: drop any remaining NaN rows
mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X = X[mask]
y = y[mask]

print(f"  Features: {len(FEATURES)} ({len(NUM_FEAT)} numeric + {len(CAT_ENC)} encoded cat)")
print(f"  Target  : log1p(total_cost), shape={y.shape}")
print(f"  NaN check: {np.isnan(X).sum()} NaN in X, {np.isnan(y).sum()} NaN in y")

# ── Scaling: StandardScaler (fit on full train — sẽ dùng CV bên trong) ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  Scaled X: mean≈0, std≈1  ✅")


# ════════════════════════════════════════════════════════════════
# 1. BASELINE — OLS (Linear Regression)
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("1. BASELINE — OLS (Linear Regression)")
print(SEP)

ols = LinearRegression()
cv = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Cross-validation scores
ols_r2   = cross_val_score(ols, X_scaled, y, cv=cv, scoring="r2")
ols_rmse = -cross_val_score(ols, X_scaled, y, cv=cv, scoring="neg_root_mean_squared_error")
ols_mae  = -cross_val_score(ols, X_scaled, y, cv=cv, scoring="neg_mean_absolute_error")

print(f"  OLS {K_FOLDS}-Fold CV:")
print(f"    R²   = {ols_r2.mean():.4f} ± {ols_r2.std():.4f}")
print(f"    RMSE = {ols_rmse.mean():.4f} ± {ols_rmse.std():.4f}")
print(f"    MAE  = {ols_mae.mean():.4f} ± {ols_mae.std():.4f}")

# Fit trên toàn bộ để lấy residuals cho diagnostic plots
ols.fit(X_scaled, y)
y_pred_ols = ols.predict(X_scaled)


# ════════════════════════════════════════════════════════════════
# 2. ELASTIC NET — CV tối ưu alpha và l1_ratio
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("2. ELASTIC NET — Cross-Validated Hyperparameter Tuning")
print(SEP)

# Grid search qua l1_ratio: 0.1 (gần Ridge) → 0.9 (gần Lasso)
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
alphas    = np.logspace(-4, 1, 50)  # 50 giá trị alpha từ 0.0001 đến 10

enet_cv = ElasticNetCV(
    l1_ratio=l1_ratios,
    alphas=alphas,
    cv=K_FOLDS,
    random_state=RANDOM_STATE,
    max_iter=10000,
    tol=1e-4,
)
enet_cv.fit(X_scaled, y)

print(f"  Best alpha    = {enet_cv.alpha_:.6f}")
print(f"  Best l1_ratio = {enet_cv.l1_ratio_:.2f}")

# Đánh giá bằng CV
enet_r2   = cross_val_score(enet_cv, X_scaled, y, cv=cv, scoring="r2")
enet_rmse = -cross_val_score(enet_cv, X_scaled, y, cv=cv, scoring="neg_root_mean_squared_error")
enet_mae  = -cross_val_score(enet_cv, X_scaled, y, cv=cv, scoring="neg_mean_absolute_error")

print(f"\n  ElasticNet {K_FOLDS}-Fold CV:")
print(f"    R²   = {enet_r2.mean():.4f} ± {enet_r2.std():.4f}")
print(f"    RMSE = {enet_rmse.mean():.4f} ± {enet_rmse.std():.4f}")
print(f"    MAE  = {enet_mae.mean():.4f} ± {enet_mae.std():.4f}")

# Hệ số non-zero (feature selection tự động)
n_nonzero = np.sum(enet_cv.coef_ != 0)
n_zero    = np.sum(enet_cv.coef_ == 0)
print(f"\n  Feature Selection: {n_nonzero} biến giữ lại, {n_zero} biến bị loại (coef=0)")

y_pred_enet = enet_cv.predict(X_scaled)


# ════════════════════════════════════════════════════════════════
# 3. POLYNOMIAL REGRESSION (degree 2 & 3) + ElasticNet trên Poly
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("3. POLYNOMIAL REGRESSION — Degree 2 & 3")
print(SEP)

# Chỉ dùng NUM_FEAT cho Polynomial (tránh bùng nổ số chiều với cat)
X_num = train[NUM_FEAT].values.astype(float)[mask]
scaler_num = StandardScaler()
X_num_scaled = scaler_num.fit_transform(X_num)

poly_results = {}

for degree in POLY_DEGREES:
    print(f"\n  ── Degree {degree} ──")
    
    # Tạo Polynomial Features
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X_num_scaled)
    n_poly_feats = X_poly.shape[1]
    print(f"    Polynomial features: {len(NUM_FEAT)} → {n_poly_feats} features")
    
    # 3a. OLS trên Polynomial
    ols_poly = LinearRegression()
    r2_poly   = cross_val_score(ols_poly, X_poly, y, cv=cv, scoring="r2")
    rmse_poly = -cross_val_score(ols_poly, X_poly, y, cv=cv, scoring="neg_root_mean_squared_error")
    mae_poly  = -cross_val_score(ols_poly, X_poly, y, cv=cv, scoring="neg_mean_absolute_error")
    
    print(f"    OLS + Poly(deg={degree})  →  R²={r2_poly.mean():.4f} ± {r2_poly.std():.4f}  "
          f"RMSE={rmse_poly.mean():.4f}  MAE={mae_poly.mean():.4f}")
    
    # 3b. ElasticNet trên Polynomial (chống overfitting)
    enet_poly_cv = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=np.logspace(-4, 1, 50),
        cv=K_FOLDS,
        random_state=RANDOM_STATE,
        max_iter=10000,
    )
    enet_poly_cv.fit(X_poly, y)
    
    r2_ep   = cross_val_score(enet_poly_cv, X_poly, y, cv=cv, scoring="r2")
    rmse_ep = -cross_val_score(enet_poly_cv, X_poly, y, cv=cv, scoring="neg_root_mean_squared_error")
    mae_ep  = -cross_val_score(enet_poly_cv, X_poly, y, cv=cv, scoring="neg_mean_absolute_error")
    
    n_nz = np.sum(enet_poly_cv.coef_ != 0)
    print(f"    ElasticNet + Poly(deg={degree})  →  R²={r2_ep.mean():.4f} ± {r2_ep.std():.4f}  "
          f"RMSE={rmse_ep.mean():.4f}  MAE={mae_ep.mean():.4f}")
    print(f"    Best alpha={enet_poly_cv.alpha_:.6f}, l1_ratio={enet_poly_cv.l1_ratio_:.2f}, "
          f"non-zero coefs={n_nz}/{n_poly_feats}")
    
    poly_results[degree] = {
        "poly": poly,
        "X_poly": X_poly,
        "ols_r2": r2_poly.mean(), "ols_rmse": rmse_poly.mean(), "ols_mae": mae_poly.mean(),
        "enet_model": enet_poly_cv,
        "enet_r2": r2_ep.mean(), "enet_rmse": rmse_ep.mean(), "enet_mae": mae_ep.mean(),
        "n_features": n_poly_feats, "n_nonzero": n_nz,
    }


# ════════════════════════════════════════════════════════════════
# 4. COMPARISON TABLE
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("4. MODEL COMPARISON TABLE")
print(SEP)

rows = [
    {"Model": "OLS (baseline)",
     "R²": f"{ols_r2.mean():.4f}", "RMSE": f"{ols_rmse.mean():.4f}", "MAE": f"{ols_mae.mean():.4f}",
     "Features": str(len(FEATURES))},
    {"Model": f"ElasticNet (α={enet_cv.alpha_:.4f}, l1={enet_cv.l1_ratio_:.2f})",
     "R²": f"{enet_r2.mean():.4f}", "RMSE": f"{enet_rmse.mean():.4f}", "MAE": f"{enet_mae.mean():.4f}",
     "Features": f"{n_nonzero}/{len(FEATURES)}"},
]

for deg in POLY_DEGREES:
    pr = poly_results[deg]
    rows.append({
        "Model": f"OLS + Poly(deg={deg})",
        "R²": f"{pr['ols_r2']:.4f}", "RMSE": f"{pr['ols_rmse']:.4f}", "MAE": f"{pr['ols_mae']:.4f}",
        "Features": str(pr["n_features"]),
    })
    rows.append({
        "Model": f"ElasticNet + Poly(deg={deg})",
        "R²": f"{pr['enet_r2']:.4f}", "RMSE": f"{pr['enet_rmse']:.4f}", "MAE": f"{pr['enet_mae']:.4f}",
        "Features": f"{pr['n_nonzero']}/{pr['n_features']}",
    })

df_comp = pd.DataFrame(rows)
print(df_comp.to_string(index=False))


# ════════════════════════════════════════════════════════════════
# 5. ElasticNet COEFFICIENT ANALYSIS
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("5. ELASTIC NET — COEFFICIENT ANALYSIS (Feature Importance)")
print(SEP)

coef_df = pd.DataFrame({
    "feature": FEATURES,
    "OLS_coef": ols.coef_,
    "ElasticNet_coef": enet_cv.coef_,
}).sort_values("ElasticNet_coef", key=abs, ascending=False)

# Top 10 biến quan trọng nhất theo ElasticNet
print("  Top 15 features (by |ElasticNet coef|):")
print(coef_df.head(15).to_string(index=False))

# Biến bị loại (coef = 0)
dropped = coef_df[coef_df["ElasticNet_coef"] == 0]["feature"].tolist()
if dropped:
    print(f"\n  ⚠️  Biến bị loại bởi ElasticNet ({len(dropped)}): {dropped}")
else:
    print(f"\n  ✅ ElasticNet giữ lại tất cả {len(FEATURES)} biến.")


# ════════════════════════════════════════════════════════════════
# 6. DIAGNOSTIC PLOTS
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("6. GENERATING DIAGNOSTIC PLOTS")
print(SEP)

# Lấy predictions cho mô hình tốt nhất poly
best_deg = max(poly_results, key=lambda d: poly_results[d]["enet_r2"])
best_poly = poly_results[best_deg]
y_pred_enet_poly = best_poly["enet_model"].predict(best_poly["X_poly"])

# Residuals
res_ols       = y - y_pred_ols
res_enet      = y - y_pred_enet
res_enet_poly = y - y_pred_enet_poly

# ── Figure 1: Residual Plots (3 mô hình so sánh) ────────────
fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
fig1.suptitle("Residual Diagnostics — OLS vs ElasticNet vs ElasticNet+Poly",
              fontsize=14, fontweight="bold", y=1.02)

models_info = [
    ("OLS (baseline)", y_pred_ols, res_ols, "#2196F3"),
    ("ElasticNet", y_pred_enet, res_enet, "#FF5722"),
    (f"ElasticNet+Poly(d={best_deg})", y_pred_enet_poly, res_enet_poly, "#4CAF50"),
]

# Row 1: Residuals vs Fitted
for i, (name, y_hat, resid, color) in enumerate(models_info):
    ax = axes[0, i]
    ax.scatter(y_hat, resid, alpha=0.15, s=8, color=color, edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fitted Values (ŷ)")
    ax.set_ylabel("Residuals (y − ŷ)")
    ax.set_title(f"{name}\nResiduals vs Fitted")
    # Trendline
    z = np.polyfit(y_hat, resid, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_hat.min(), y_hat.max(), 100)
    ax.plot(x_line, p(x_line), color="red", linewidth=1.5, linestyle="-")

# Row 2: Q-Q Plot
for i, (name, y_hat, resid, color) in enumerate(models_info):
    ax = axes[1, i]
    stats.probplot(resid, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(color)
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[0].set_alpha(0.4)
    ax.set_title(f"{name}\nQ-Q Plot (Normality)")

fig1.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "fig_residual_comparison.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  ✅ Saved: {out1}")


# ── Figure 2: Coefficient Comparison (OLS vs ElasticNet) ─────
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
fig2.suptitle("Coefficient Comparison — OLS vs ElasticNet",
              fontsize=14, fontweight="bold")

# Top 15 by absolute OLS coef
top_feats = coef_df.head(15)

ax = axes2[0]
y_pos = np.arange(len(top_feats))
bars_ols  = ax.barh(y_pos + 0.2, top_feats["OLS_coef"].values,
                    height=0.35, color="#2196F3", label="OLS", alpha=0.8)
bars_enet = ax.barh(y_pos - 0.2, top_feats["ElasticNet_coef"].values,
                    height=0.35, color="#FF5722", label="ElasticNet", alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_feats["feature"].values, fontsize=8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Top 15 Coefficients")
ax.set_xlabel("Coefficient Value")
ax.legend()
ax.invert_yaxis()

# ElasticNet coefficient path (sparsity visualization)
ax = axes2[1]
nonzero_mask = enet_cv.coef_ != 0
colors = ["#4CAF50" if nz else "#F44336" for nz in nonzero_mask]
ax.barh(range(len(FEATURES)), np.abs(enet_cv.coef_), color=colors, alpha=0.7)
ax.set_yticks(range(len(FEATURES)))
ax.set_yticklabels(FEATURES, fontsize=6)
ax.set_title(f"ElasticNet |Coef| — Green=kept, Red=dropped\n"
             f"(α={enet_cv.alpha_:.4f}, l1_ratio={enet_cv.l1_ratio_:.2f})")
ax.set_xlabel("|Coefficient|")
ax.invert_yaxis()

fig2.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "fig_coefficient_comparison.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  ✅ Saved: {out2}")


# ── Figure 3: Model Performance Bar Chart ────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle(f"Model Performance Comparison ({K_FOLDS}-Fold CV)",
              fontsize=14, fontweight="bold")

model_names = [r["Model"] for r in rows]
r2_vals     = [float(r["R²"]) for r in rows]
rmse_vals   = [float(r["RMSE"]) for r in rows]
mae_vals    = [float(r["MAE"]) for r in rows]

colors_bar = ["#2196F3", "#FF5722", "#9C27B0", "#4CAF50", "#FF9800", "#00BCD4"]

for ax, vals, metric in zip(axes3, [r2_vals, rmse_vals, mae_vals], ["R²", "RMSE", "MAE"]):
    bars = ax.barh(model_names, vals, color=colors_bar[:len(model_names)], alpha=0.85)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_xlabel(metric)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.invert_yaxis()

fig3.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "fig_model_performance.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  ✅ Saved: {out3}")


# ════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("7. SUMMARY")
print(SEP)
print(f"""
  ┌────────────────────────────────────────────────────────────────┐
  │ KẾT QUẢ CHÍNH                                                  │
  ├────────────────────────────────────────────────────────────────┤
  │ 1. OLS Baseline:                                               │
  │    R² = {ols_r2.mean():.4f}, RMSE = {ols_rmse.mean():.4f}, MAE = {ols_mae.mean():.4f}               │
  │                                                                │
  │ 2. ElasticNet (α={enet_cv.alpha_:.4f}, l1_ratio={enet_cv.l1_ratio_:.2f}):            │
  │    R² = {enet_r2.mean():.4f}, RMSE = {enet_rmse.mean():.4f}, MAE = {enet_mae.mean():.4f}               │
  │    → Giữ {n_nonzero}/{len(FEATURES)} biến, loại {n_zero} biến dư thừa                │
  │                                                                │
  │ 3. ElasticNet + Polynomial (degree={best_deg}):                     │
  │    R² = {poly_results[best_deg]['enet_r2']:.4f}, RMSE = {poly_results[best_deg]['enet_rmse']:.4f}, MAE = {poly_results[best_deg]['enet_mae']:.4f}               │
  │    → {poly_results[best_deg]['n_nonzero']}/{poly_results[best_deg]['n_features']} polynomial features giữ lại          │
  ├────────────────────────────────────────────────────────────────┤
  │ NHẬN XÉT                                                      │
  │ • ElasticNet thực hiện feature selection tự động nhờ L1 penalty │
  │ • Polynomial features bắt được quan hệ phi tuyến               │
  │ • Kết hợp ElasticNet + Poly giúp chống overfitting hiệu quả    │
  └────────────────────────────────────────────────────────────────┘

  Output files:
    • fig_residual_comparison.png    — So sánh residuals 3 mô hình
    • fig_coefficient_comparison.png — Hệ số OLS vs ElasticNet
    • fig_model_performance.png      — Biểu đồ R²/RMSE/MAE
""")

print(f"{'='*70}")
print("✅  ElasticNet + Polynomial Regression: HOÀN THÀNH")
print(f"{'='*70}")
