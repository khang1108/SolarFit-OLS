### 1. Các giả thiết về Phần dư (Residuals) trong mô hình OLS

Trong mô hình Hồi quy tuyến tính (Ordinary Least Squares - OLS), để các ước lượng $\hat{\beta}$ mang ý nghĩa thống kê và là ước lượng tốt nhất, phần dư (hay sai số ngẫu nhiên $\varepsilon$) phải tuân mãn các giả thiết nghiêm ngặt dựa trên định lý **Gauss-Markov**.

*   **Kỳ vọng của phần dư bằng 0 (Zero Conditional Mean):**
    $$E[\varepsilon | X] = 0$$
    Nghĩa là, với bất kỳ giá trị nào của biến độc lập $X$, trung bình của các sai số luôn bằng 0. Nếu giả thiết này bị vi phạm, mô hình đang bị thiên lệch (biased), đường hồi quy không đi qua trung tâm của dữ liệu (thường do thiếu biến quan trọng hoặc chọn sai dạng hàm). Ta kiểm tra giả thiết này thông qua biểu đồ **Residuals vs Fitted**.
*   **Không có hiện tượng tự tương quan (No Autocorrelation):**
    $$Cov(\varepsilon_i, \varepsilon_j) = 0 \quad (\forall i \neq j)$$
    Sai số của quan sát này không được dự đoán hoặc bị ảnh hưởng bởi sai số của quan sát khác. Giả thiết này đặc biệt quan trọng đối với dữ liệu chuỗi thời gian (Time Series).
*   **Phần dư có phân phối chuẩn (Normality of Errors):**
    $$\varepsilon \sim \mathcal{N}(0, \sigma^2)$$
    Dù OLS vẫn có thể tìm ra đường hồi quy mà không cần giả thiết này, nhưng phân phối chuẩn của phần dư là **bắt buộc** để ta có thể thực hiện các kiểm định thống kê (như tính p-value, t-test cho từng hệ số $\beta$, hoặc F-test cho toàn bộ mô hình). Ta kiểm tra giả thiết này bằng biểu đồ **Normal Q-Q Plot**.

---

### 2. Giả thiết Phương sai / Độ lệch chuẩn không đổi (Homoscedasticity)

Đây là một trong những giả thiết quan trọng nhất của OLS (GM4 trong tài liệu):

*   **Định nghĩa (Homoscedasticity):** 
    $$Var(\varepsilon_i | X) = \sigma^2 \quad (\forall i)$$
    Giả thiết này yêu cầu phương sai (và do đó là độ lệch chuẩn $\sigma$) của phần dư phải không đổi trên toàn bộ miền giá trị của $X$. Tức là, mức độ phân tán của các điểm dữ liệu xung quanh đường hồi quy là đồng đều, bất kể giá trị dự đoán là nhỏ hay lớn.
*   **Hiện tượng Phương sai thay đổi (Heteroscedasticity):** 
    Nếu giả thiết này bị vi phạm, phương sai của phần dư sẽ thay đổi (thường là phình to ra theo hình phễu khi giá trị dự đoán tăng lên).
*   **Hậu quả khi vi phạm:**
    Mặc dù ước lượng $\hat{\beta}$ của OLS vẫn là ước lượng không chệch (unbiased), nhưng nó **không còn là ước lượng tốt nhất (mất đi tính BLUE - Best Linear Unbiased Estimator)**. Cụ thể:
    *   Phương sai của $\hat{\beta}$ không còn đạt mức tối thiểu.
    *   Các công thức tính Sai số chuẩn (Standard Errors) của hệ số sẽ bị sai lệch.
    *   Dẫn đến các kiểm định giả thuyết (t-test, p-value) và các khoảng tin cậy (Confidence Intervals) trở nên vô giá trị và gây hiểu lầm.
*   **Cách phát hiện:** Ta sử dụng đồ thị **Scale-Location** (hoặc Spread-Location). Nếu các điểm trên đồ thị phân tán ngẫu nhiên với một đường xu hướng nằm ngang, giả thiết được thỏa mãn.

---

### 3. Đánh giá khả năng tổng quát hóa bằng K-Fold Cross-Validation (CV)

Việc phân tích phần dư (Residual Diagnostics) ở trên giúp chúng ta đánh giá xem mô hình có khớp (fit) tốt với tập dữ liệu huấn luyện (Training data) và có thỏa mãn các giả thiết thống kê hay không. Tuy nhiên, nó không cho biết mô hình sẽ hoạt động ra sao trên dữ liệu thực tế chưa từng thấy. Đó là lúc ta cần đến **K-Fold Cross-Validation**.

*   **Khái niệm cơ bản:** K-Fold CV là kỹ thuật chia ngẫu nhiên toàn bộ tập dữ liệu thành $k$ phần (folds) có kích thước xấp xỉ bằng nhau. 
*   **Cơ chế hoạt động:**
    1. Quá trình huấn luyện và kiểm tra được lặp lại $k$ lần.
    2. Ở mỗi lần lặp thứ $i$, phần thứ $i$ được giữ lại làm tập kiểm tra (Test set).
    3. Mô hình OLS được huấn luyện (fit) trên $k-1$ phần còn lại (Training set).
    4. Sử dụng mô hình vừa huấn luyện để dự đoán trên tập kiểm tra và tính toán độ lỗi (trong đồ án này sử dụng Mean Squared Error - MSE).
*   **Ý nghĩa và ưu điểm so với Train/Test split thông thường:**
    *   **Tận dụng tối đa dữ liệu:** Mỗi quan sát trong bộ dữ liệu đều được sử dụng để làm dữ liệu kiểm tra đúng một lần, và làm dữ liệu huấn luyện $k-1$ lần. Điều này đặc biệt hữu ích khi bộ dữ liệu có kích thước nhỏ.
    *   **Ngăn chặn Quá khớp (Overfitting):** MSE trung bình của $k$ lần lặp (CV Score) là một ước lượng khách quan và đáng tin cậy hơn về sai số dự đoán của mô hình trên dữ liệu mới.
    *   **Đánh giá độ ổn định:** Bằng cách quan sát phương sai của độ lỗi giữa các fold, ta có thể biết được mô hình OLS có nhạy cảm (sensitive) với cách chia dữ liệu hay không. Nếu MSE giữa các fold chênh lệch quá lớn, mô hình có thể đang bị ảnh hưởng bởi các điểm ngoại lai (outliers) hoặc dữ liệu phân bổ không đồng đều.