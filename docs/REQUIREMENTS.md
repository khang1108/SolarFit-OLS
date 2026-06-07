# ĐỀ ÁN 2: Data Fitting và Phương Pháp OLS

**Môn học:** Toán Ứng Dụng và Thống Kê (MTH00051)  
**Học kỳ:** 2, 2025-2026

---

## 📋 MỤC LỤC

### **Phần 1: Lý Thuyết Data Fitting và Minh Họa** (6.0 điểm)

#### 1.1 Bài Toán Data Fitting
- 1.1.1 Phát biểu bài toán tổng quát
- 1.1.2 Các Giả Thiết Gauss-Markov (GM1-GM5)

#### 1.2 Phương Pháp OLS
- 1.2.1 Hàm mất mát và nghiệm OLS (RSS, Normal Equations)
- 1.2.2 Ma Trận Chiếu (Hat Matrix H)
- 1.2.3 Định Lý Gauss-Markov (BLUE property)
- 1.2.4 Ước Lượng Phương Sai σ̂²

#### 1.3 Đánh Giá Mô Hình
- 1.3.1 Hệ số xác định R² và R̄² hiệu chính
- 1.3.2 Kiểm Định Giả Thuyết (t-test, F-test)

#### 1.4 Các Vấn Đề Nâng Cao trong Data Fitting
- 1.4.1 Đa cộng tuyến (Multicollinearity) - VIF detection
- 1.4.2 Hồi Quy Ridge và Lasso (Regularization)
- 1.4.3 Phân Tích Phần Dư (Residual Analysis)
- 1.4.4 Cross-Validation và Lựa Chọn Mô Hình (k-fold CV, AIC, BIC)

#### 1.5 Yêu Cầu Cài Đặt Python — Phần 1

**9 Functions required:**

| #   | Function                                              | Mô tả                                                                           | Điểm |
| --- | ----------------------------------------------------- | ------------------------------------------------------------------------------- | ---- |
| 1   | `ols_fit(X, y)`                                       | Tính β̂ = (X'X)^{-1}X'y và σ̂²                                                    | -    |
| 2   | `hat_matrix(X)`                                       | Tính H = X(X'X)^{-1}X', kiểm tra idempotent                                     | -    |
| 3   | `model_metrics(y, y_hat, p)`                          | Tính RSS, TSS, R², R̄², RMSE, F-statistic                                        | -    |
| 4   | `coef_inference(X, y, beta_hat, sigma2)`              | Tính standard errors, t-stats, p-values, CI 95%                                 | -    |
| 5   | `vif(X)`                                              | Tính VIF cho từng biến                                                          | -    |
| 6   | `ridge_fit(X, y, lam)`                                | Cài đặt Ridge Regression, về ridge trace                                        | -    |
| 7   | `residual_plots(X, y, beta_hat)`                      | 4 biểu đồ diagnostic: Residuals vs Fitted, Q-Q, Scale-Location, Cook's Distance | -    |
| 8   | `kfold_cv(X, y, k)`                                   | Cài đặt k-fold cross-validation, tính CV score                                  | -    |
| 9   | `gauss_markov_simulation(X, beta_true, sigma, n_sim)` | Monte Carlo kiểm chứng E[β̂]=β và BLUE                                           | -    |

#### 1.6 Tiêu Chí Đánh Giá — Phần 1

| Tiêu chí                     | Mô tả                              | Điểm    |
| ---------------------------- | ---------------------------------- | ------- |
| Trình bày lý thuyết OLS      | Đúng, đầy đủ công thức, chứng minh | 1.0     |
| Cài đặt OLS từ đầu           | Đúng, kiểm chứng với NumPy         | 1.0     |
| Hat Matrix và tính chất      | Cài đặt, kiểm tra idempotent       | 0.5     |
| Kiểm định giả số (t, F)      | Tính đúng t-stat, p-value          | 0.5     |
| Regularization (Ridge/Lasso) | Cài đặt, về ridge trace            | 1.0     |
| Phân tích phần dư            | 4 biểu đồ đầy đủ, nhận xét         | 0.5     |
| Cross-validation             | Cài k-fold CV, so sánh mô hình     | 0.5     |
| Gauss-Markov verification    | Monte Carlo rõ ràng, nhận xét      | 0.5     |
| Trình bày Notebook           | Rõ ràng, có markdown giải thích    | 0.5     |
| **Tổng Phần 1**              |                                    | **6.0** |

---

### **Phần 2: Ứng Dụng Data Fitting vào Dữ Liệu Thực Tế** (5.5 điểm)

#### 2.1 Tiêu Chí Chọn Bộ Dữ Liệu

**Yêu cầu chính:**
1. **Dữ liệu thực (real-world)** - Không dùng synthetic data hoặc Iris
2. **Có missing values** (≥ 5% trên ít nhất một cột)
3. **Bài toán hồi quy** (regression) với target continuous
4. **Kích thước:** n > 200, p > 3
5. **Nguồn:** Kaggle, UCI ML Repository, World Bank, WHO, OECD

**Gợi ý datasets:**
- Kaggle House Prices (79 features, missing values)
- UCI Auto MPG
- UCI Bike Sharing
- World Bank Open Data
- WHO Global Health Observatory
- OECD Data

#### 2.2 Preprocessing Dữ Liệu

##### 2.2.1 Khảo Sát Dữ Liệu (EDA)

**Yêu cầu bao gồm:**
- Thống kê mô tả: mean, median, std, min, max, quartiles
- Phân phối tần biến: histogram, boxplot
- Ma trận tương quan: heatmap
- Kiểm tra dữ liệu trùng lặp
- Phân tích missing values: tìm lệ thiếu theo từng cột
- Phát hiện outliers: phương pháp IQR, z-score hoặc tự định nghĩa

##### 2.2.2 Xử Lý Missing Values

**5 Phương pháp cần biết:**

| Phương pháp                          | Mô tả                                                           | Yêu cầu      |
| ------------------------------------ | --------------------------------------------------------------- | ------------ |
| **MV1. Listwise deletion**           | Xóa toàn bộ hàng có ít nhất một giá trị thiếu                   | Tùy chọn     |
| **MV2. Mean/Median/Mode imputation** | Thay giá trị thiếu bằng trung bình, trung vị, hoặc mode của cột | **Bắt buộc** |
| **MV3. Regression imputation**       | Prediction giá trị thiếu bằng hồi quy từ các biến khác          | Tùy chọn     |
| **MV4. k-NN imputation**             | Thay bằng trung bình k quan sát gần nhất                        | Tùy chọn     |
| **MV5. MICE (Multiple Imputation)**  | Tạo nhiều bản sao dữ liệu đã điền, gộp kết quả                  | Tùy chọn     |

**Lưu ý IMPORTANT:** Tạo viên nên giải thích *lý do* chọn phương pháp xử lý missing values (MCAR, MAR hay MNAR).

##### 2.2.3 Các Bước Preprocessing Khác

- **Feature engineering:** Tạo biến mới, biến đổi (log, √, polynomial)
- **Encoding biến phân loại:** One-Hot encoding hoặc Ordinal encoding
- **Chuẩn hóa (Normalization/Standardization):** z-score: x_std = (x - mean) / std
- **Phát hiện và xử lý outliers:** Winsorization hoặc loại bộ
- **Kiểm tra đa cộng tuyến:** VIF trước khi đưa vào mô hình

#### 2.3 Xây Dựng và Đánh Giá Mô Hình

##### 2.3.1 Quy Trình Xây Dựng Mô Hình

```
EDA → Preprocessing → Train/Test Split → Xây dựng mô hình → Đánh giá → Điều chỉnh lại
```

##### 2.3.2 Các Mô Hình Cần Thử Nghiệm

**Bắt buộc cài đặt (≥3 mô hình):**

| Mô hình                      | Loại         | Mô tả                                                    |
| ---------------------------- | ------------ | -------------------------------------------------------- |
| **OLS cơ bản**               | Bắt buộc     | Hồi quy tuyến tính không có regularization               |
| **OLS + feature selection**  | Bắt buộc     | Loại bỏ biến không giải thích, dựa trên p-value hoặc VIF |
| **Ridge / Lasso**            | Bắt buộc     | Regularization (L2 hoặc L1), chọn λ qua cross-validation |
| **Polynomial / Interaction** | Tùy chọn     | Thêm đặc trưng đa thức hoặc tương tác                    |
| **Kernel / Bayesian**        | Bonus (+0.5) | Kernel Ridge hoặc Bayesian Linear Regression             |

##### 2.3.3 Tiêu Chí So Sánh Mô Hình

**Trên tập test** (không được dùng trong quá trình huấn luyện):

$$\text{MAE} = \frac{1}{n_{test}} \sum |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{n_{test}} \sum (y_i - \hat{y}_i)^2}$$

$$R^2_{test} = 1 - \frac{\text{RSS}_{test}}{\text{TSS}_{test}}$$

#### 2.4 Kỹ Thuật Nâng Cao (Tùy Chọn)

**Kernel Regression:**
$$\hat{y}(x) = \mathbf{k}(x)^T (\mathbf{K} + \lambda I)^{-1} \mathbf{y}$$

với $K_{ij} = k(x_i, x_j)$ (Gram matrix), kernel RBF: $k_{RBF}(x, x') = \exp\left(-\frac{||x - x'||^2}{2\sigma^2}\right)$

**Bayesian Linear Regression:**
- Prior: $\beta \sim \mathcal{N}(m_0, S_0)$
- Likelihood: $y | x, \beta \sim \mathcal{N}(x^T\beta, \sigma^2)$
- Posterior: $\beta | X, y \sim \mathcal{N}(m_n, S_n)$

Ưu điểm: Cung cấp uncertainty quantification (credible intervals).

#### 2.5 Yêu Cầu Cài Đặt Python — Phần 2

**6 yêu cầu chính:**

1. **DataPipeline class** - Xử lý missing values, encoding, chuẩn hóa (fit trên train, transform trên test)
2. **So sánh ≥3 mô hình** - Bảng tổng hợp MAE, RMSE, R² trên test set
3. **Cross-validation** - k-fold (k=5 hoặc 10) cho Ridge/Lasso
4. **Phân tích phần dư** - Với mô hình tốt nhất, về 4 biểu đồ chẩn đoán
5. **Feature importance** - Về biểu đồ hệ số hoặc VIF/permutation importance để giải thích mô hình
6. **Nhân xét và kết luận** - Giải thích kết quả theo ngữ cảnh bộ dữ liệu

#### 2.6 Tiêu Chí Đánh Giá — Phần 2

| Tiêu chí                  | Mô tả                                     | Điểm           |
| ------------------------- | ----------------------------------------- | -------------- |
| Chọn bộ dữ liệu           | Đúng tiêu chí, mô tả rõ ràng              | 0.5            |
| EDA                       | Đầy đủ thống kê mô tả, biểu đồ            | 0.5            |
| Xử lý missing values      | Đúng phương pháp, có giải thích           | 1.0            |
| Preprocessing tổng thể    | Pipeline đầy đủ, fit/transform đúng       | 0.5            |
| Xây dựng ≥3 mô hình       | OLS, Ridge/Lasso, mô hình khác            | 1.5            |
| Đánh giá trên test set    | MAE, RMSE, R², phân tích phần dư          | 1.0            |
| Nhân xét & kết luận       | Phân tích có chiều sâu, liên hệ lý thuyết | 0.5            |
| Kỹ thuật nâng cao (bonus) | Kernel / Bayesian                         | **+0.5**       |
| **Tổng Phần 2**           |                                           | **5.5 (+0.5)** |

---

## 📂 Cấu Trúc Thư Mục

```
Group_<ID>/
├── README.md
├── requirements.txt
├── report/
│   ├── report.pdf
│   └── report.tex
├── part1/                          # OLS implementation
│   ├── ols_implementation.py       # OLS từ scratch
│   ├── model_evaluation.py         # RSS, R², diagnostic plots
│   ├── inference.py                # t-test, VIF, confidence intervals
│   ├── regularization.py           # Ridge/Lasso
│   ├── cross_validation.py         # k-fold CV
│   └── part1_notebook.ipynb        # Theoretical demo
│
└── part2/                          # Real-world application
    ├── data/
    │   └── <dataset>.csv           # Original data
    ├── src/
    │   ├── data_pipeline.py        # Missing values, encoding, scaling
    │   ├── model_comparison.py     # Model building & evaluation
    │   ├── advanced_methods.py     # Kernel / Bayesian (if have)
    │   └── part2_notebook.ipynb    # Results analysis & discussion
    └── outputs/
        ├── figures/                # Plots từ EDA, diagnostics
        ├── results.csv             # Model comparison table
        └── predictions.csv         # Test predictions
```

---

## ✅ Yêu Cầu Chung

### 3.1 Cấu Trúc Báo Cáo

Báo cáo viết bằng **LaTeX hoặc Markdown** (xuất ra PDF), bao gồm:

1. **Trang bìa** - Họ và tên, MSSV, nhóm, giảng viên hướng dẫn
2. **Mục lục**
3. **Phần 1: Lý thuyết và minh họa**
4. **Phần 2: Ứng dụng thực tế**
5. **Kết luận** - Tóm tắt kết quả, bài học rút ra, hướng mở rộng
6. **Tài liệu tham khảo** - Ít nhất 5 tài liệu
7. **Phụ lục** - Bảng số liệu, biểu đồ bổ sung (nếu có)

### 3.2 Yêu Cầu Kỹ Thuật

- Sử dụng **Python 3.10+**
- Viết code **rõ ràng** (clean code), chủ thích code nếu thật sự cần thiết
- Không dùng `sklearn.LinearRegression` để cài đặt OLS chính, nhưng được dùng để kiểm chứng
- Có thể dùng NumPy/SciPy cho **kiểm chứng**, không phải để thay thế cài đặt
- Jupyter Notebook cho phần trình bày kết quả

---

## 🎯 Tổng Điểm

| Phần                     | Điểm     |
| ------------------------ | -------- |
| Part 1: Lý thuyết OLS    | 6.0      |
| Part 2: Ứng dụng thực tế | 5.5      |
| **Tổng**                 | **11.5** |

**Bonus:** Kỹ thuật nâng cao (Kernel/Bayesian) +0.5 điểm

---

## 📚 Tài Liệu Tham Khảo Đề Xuất

1. Strang, G. (2023). *Introduction to Linear Algebra*
2. Wooldridge, J. M. (2013). *Introductory Econometrics*
3. Greene, W. H. (2012). *Econometric Analysis*
4. James, G., et al. (2013). *An Introduction to Statistical Learning*
5. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*

---

**Ngày phát hành:** Theo lịch học kỳ 2, 2025-2026  
**Hạn nộp:** Theo thông báo từ giảng viên
