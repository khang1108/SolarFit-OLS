# 📋 REVIEW: Phần 2 Code vs Yêu Cầu Đề

## ✅ Checklist Phần 2 (PDF §2.1-§2.5)

### §2.1 TIÊU CHÍ CHỌN BỘ DỮ LIỆU
| Yêu cầu | Hiện tại | Status |
|---------|----------|--------|
| Real-world data | Tanzania Tourism (Zindi) | ✅ |
| n ≥ 200 | Train: 4,820 rows, Test: 1,602 rows | ✅ |
| p ≥ 3 | 21 features (4 numeric + 17 categorical) | ✅ |
| Regression task | `total_cost` (continuous) | ✅ |
| Missing values ≥ 5% | travel_with 23%, most_impressing 6.5% | ✅ |

**PASS:** ✅ Dataset đủ tiêu chuẩn

---

### §2.2 TIỀN XỬ LÝ DỮ LIỆU

#### 2.2.1 EDA (Khảo sát dữ liệu)
| Yêu cầu | Hiện tại | Status |
|---------|----------|--------|
| Mean, median, std, min, max | `main.py`: descriptive_stats đầy đủ | ✅ |
| Histogram, boxplot | `day1_fig1_target_hist_heatmap.png` + `day1_fig2_boxplots.png` | ✅ |
| Heatmap tương quan | `day1_fig1_target_hist_heatmap.png` (correlation matrix) | ✅ |
| Kiểm tra dữ liệu trùng lặp | `main.py`: duplicate check | ✅ |
| Phát hiện outlier (IQR, Z-score) | `main.py`: IQR + Z-score analysis | ✅ |
| **Categorical EDA** | `eda_categorical_shift.ipynb`: cardinality, rare categories | ✅ |

**PASS:** ✅ EDA kỹ lưỡng

#### 2.2.2 Xử lý Missing Values (5 phương pháp)
| Phương pháp | YC Đề | Code | Status |
|-------------|--------|------|--------|
| MV1. Listwise deletion | Nên có | ❌ Chỉ nêu trong comment | ❌ |
| MV2. Mean/Median imputation | Nên có | `main.py`: nêu chiến lược (MEDIAN) | ⚠️ Không implement |
| MV3. Regression imputation | Nên có | ❌ | ❌ |
| MV4. k-NN imputation | Nên có | ❌ | ❌ |
| MV5. MICE imputation | Nên có | ❌ | ❌ |
| MCAR/MAR/MNAR analysis | Yêu cầu | `main.py`: phân tích kỹ lưỡng | ✅ |

**⚠️ PASS PARTIALLY:** Phân tích có nhưng không implement xử lý

#### 2.2.3 Tiền xử lý khác
| Yêu cầu | Hiện tại | Status |
|---------|----------|--------|
| Feature engineering (log, √, polynomial) | `main.py`: log1p(total_cost) | ✅ |
| Encoding (One-Hot, Ordinal) | `eda_categorical_shift.ipynb`: phân loại candidates | ⚠️ |
| Chuẩn hóa (Z-score) | `main.py`: nêu StandardScaler trong plan | ⚠️ |
| Outlier handling (Winsorization) | `main.py`: decision to keep outliers | ✅ |
| VIF check | `main.py`: "Max VIF = 1.15" | ✅ |

**⚠️ PASS PARTIALLY:** Plan chỉ, không implement đầy đủ

---

### §2.3 XÂY DỰNG & ĐÁNH GIÁ MÔ HÌNH

#### 2.3.1 Pipeline (EDA → Preprocessing → Train/Test → Models)
| Thành phần | Hiện tại | Status |
|-----------|----------|--------|
| EDA | ✅ `main.py` + `eda_categorical_shift.ipynb` | ✅ |
| Preprocessing | ⚠️ Plan chỉ | ⚠️ |
| Train/Test split | ❌ | ❌ |
| Model building | ❌ | ❌ |
| Evaluation | ❌ | ❌ |

**❌ FAIL:** Pipeline không hoàn thiện

#### 2.3.2 Mô hình (5 mô hình để so sánh)
| Mô hình | YC Đề | Code | Status |
|--------|--------|------|--------|
| OLS cơ bản | 1.0đ | ❌ | ❌ |
| OLS with feature selection | 1.0đ | ❌ | ❌ |
| Ridge/Lasso | 1.5đ | ❌ | ❌ |
| Polynomial/Interaction | Tùy chọn | ❌ | ❌ |
| Kernel/Bayesian | Bonus +0.5đ | ❌ | ❌ |

**❌ FAIL:** 0/5 mô hình

#### 2.3.3 Metrics trên TEST SET
| Metric | Hiện tại | Status |
|--------|----------|--------|
| MAE | ❌ | ❌ |
| RMSE | ❌ | ❌ |
| R² | ❌ | ❌ |

**❌ FAIL:** Không có metrics

---

### §2.5 YÊU CẦU CÀI ĐẶT PYTHON — PHẦN 2

| Yêu cầu | Hiện tại | Status |
|---------|----------|--------|
| 1. DataPipeline class | ❌ | ❌ |
| 2. 3+ mô hình so sánh | ❌ | ❌ |
| 3. MAE, RMSE, R² trên test | ❌ | ❌ |
| 4. k-fold CV (k=5 hoặc 10) | ❌ | ❌ |
| 5. Feature importance | ❌ | ❌ |
| 6. Nhận xét & kết luận | ⚠️ Plan chỉ | ⚠️ |

**❌ FAIL:** Còn thiếu hầu hết

---

## 📊 ĐIỂM ƯỚC TÍNH

| Mục | Điểm Đề | Hiện Tại | % |
|-----|---------|----------|-----|
| Chọn & chuẩn bị dataset | 0.5đ | ✅ 0.5đ | 100% |
| EDA | 0.5đ | ✅ 0.5đ | 100% |
| Xử lý missing values | 1.0đ | ⚠️ 0.2đ | 20% |
| Tiền xử lý tổng thể | 0.5đ | ⚠️ 0.2đ | 40% |
| Xây dựng 3+ mô hình | 1.5đ | ❌ 0đ | 0% |
| Đánh giá mô hình | 1.0đ | ❌ 0đ | 0% |
| Cross-validation | 0.5đ | ❌ 0đ | 0% |
| Feature importance | 0.5đ | ❌ 0đ | 0% |
| Nhận xét & kết luận | 0.5đ | ⚠️ 0.1đ | 20% |
| **Tổng Phần 2** | **5.5đ** | **~1.5đ** | **27%** |

---

## 🎯 ĐỀ CẦN HOÀN THÀNH

### Priority 1 (Bắt buộc để pass)
- [ ] Implement DataPipeline class (missing value imputation + feature engineering)
- [ ] Code 3-5 models (OLS, Ridge, Lasso, Polynomial, Bayesian)
- [ ] Compute MAE, RMSE, R² trên test set
- [ ] k-fold cross-validation (k=5 hoặc 10)

### Priority 2 (Important)
- [ ] Feature importance analysis
- [ ] Model comparison table + visualization
- [ ] Residual plots (4 diagnostic plots)
- [ ] Kết luận và insights

### Priority 3 (Nice to have)
- [ ] Kernel regression
- [ ] Bayesian regression
- [ ] Hyperparameter tuning
- [ ] Feature selection (RFE, Lasso)

---

## 📁 FILES STATUS

### ✅ Completed
- `part2/src/main.py` — Day 1 EDA (484 lines)
  - Missing value analysis + MCAR/MAR/MNAR
  - Descriptive statistics (mean, median, std, skew, IQR outliers, Z-score)
  - 3 visualizations (histogram, boxplot, VIF/correlation/KS)
  
- `part2/eda_categorical_shift.ipynb` — Categorical EDA
  - Cardinality analysis
  - Train/Test shift detection (KS test)
  - Rare category handling
  
- `part2/data/` — Dataset
  - Train.csv (4,820 rows × 21 cols)
  - Test.csv (1,602 rows × 20 cols)
  - SampleSubmission.csv, VariableDefinitions.csv

- `part2/outputs/` — EDA Plots
  - day1_fig1_target_hist_heatmap.png
  - day1_fig2_boxplots.png
  - day1_fig3_vif_corr_shift.png

### ⏳ In Progress
- Categorical feature analysis (notebook hơn 50%)

### ❌ Not Started
- `data_pipeline.py` — DataPipeline class (Priority 1)
- `model_comparison.py` — Model comparison (Priority 1)
- `model_building.py` — Individual model implementations
- `part2_notebook.ipynb` — Final results and discussion
