# **Progress Status**

## **Day 1: Data Audit and EDA**
Mục tiêu của ngày này là phân tích dữ liệu từ file `train.csv` và `test.csv` để đưa ra các quyết định các insight về bài toán, xem nhóm cần làm những gì sắp tới. Trong quá trình này, ta cần áp dụng quá trình thống kê mô tả, vẽ các figures để cung cấp các góc nhìn trực quan về dữ liệu để có thể đánh gíá. 

### 1. **EDA Questions**
#### 1.1 **Overview**
- Mục tiêu chính xác của bài toán được đặt ra tại [Link](https://zindi.africa/competitions/tanzania-tourism-prediction/data) là gì? Là chi phí chuyến đi, tổng chi tiêu của khách du lịch, hay chi phí được báo cáo?
- Đơn vị của mục tiêu là gì?
- Trong bài toán này ta sẽ tiến hành dự đoán cái gì?
- Khi ta dự đoán trên `test.csv` có các features nào mà ta đã thực sự biết trước?
- Có feature nào có khả năng chứa các thông tin nào rò rỉ hay không? Tức là trong tập `train` tồn tại một feature nào đó bất kỳ chỉ xảy ra khi `target` được dự đoán. Ví dụ như, nếu ta dự đoán *một người có bị viêm xoang hay không* mà ta lại dùng một feature là người đó đã uống thuốc kháng sinh, bởi vì thông thường chỉ khi bị bệnh mới uống.

##### **Ouptut**
Đầu ra phải có các thứ sau:
- Một file `.md` mô tả chi tiết về bài toán
- Xác nhận được mục tiêu của đề bài
- Xác định các rủi ro leakage của dữ liệu nếu có và đưa ra biện pháp xử lý.

#### 1.2 **Data Contract**
Nhóm cần xác định được các `schema` trước khi bắt tay vào làm việc, tránh việc các thành viên dùng các `class model` để lưu trữ dữ liệu không giống sau, thiếu sự thống nhất, gây khó khăn khi làm việc.

- `Train` có bao nhiêu vòng và bao nhiêu cột?
- `Test` có bao nhiêu dòng và bao nhiêu cột
- `ID` của bảng `train` và `test` có chung nhau không? Nó là gì?
- `target` tên của nó là gì?
- Sự khác biệt giữa các cột trong `train` và `test`. Liệu có sự khác nhau giữa hai file này hay không?
- Format của `SampleSubmission.csv` là gì? Nó bao gồm những gì?
- `ID` trong `SampleSubmission.csv` có giống với `ID` trong `test.csv` không?
- Có sự trùng nhau về `ID` hay không?
- Missing Values đang được biểu diễn dưới dạng gì?

##### **Ouptut**
- `schema_contract.csv` - file csv mô tả chi tiết `schema` của bài toán
- `column_types.csv` - file csv mô tả chi tiết `column_types` của bài toán
- Xác định `ID/target/submission` format

#### 1.3 **Target Variable**
Hiểu phân phối target để quyết định có cần log-transform, xử lý outlier, hoặc chọn metric phù hợp.

- Target là biến liên tục hay biến không liên tục?
- Target có chứa các Missing Values hay không
- Khoảng giá trị của target là gì?
- Phân phối của target như nào? Có gần phân phối chuẩn hay lệch trái, lệch phải?
- Tính toán khoảng `IQR` của target. Khoảng cách giữa `Q3` và `max` có cách xa nhau hay khoongg?
- `mean` và `median` như nào với nhau?
- Nếu có các dữ liệu target rất cao thì nguyên nhân nó là gì? Nó là lỗi nhập liệu hay thực sự là như vậy?
- Nếu ta dùng các `log-transform` thì dữ liệu có phân phối về phân phối chuẩn hơn không?
- Nếu ta áp dụng `log-transform` thì ta cần chuyển ngược lại kiểu gì?
- Liệu `log-target` có làm MAE, RMSE xấu hơn hay không?
  
##### **Output**
- `target_summary.csv`
- `target_distribution.png`
- `target_boxplot.png`
- `log_target_distribution.png`
- Quyết định: sẽ test target gốc, log-target, hay cả hai.

#### 1.5 **Missing Values**
Không chỉ đếm missing, mà phải hiểu missing có ý nghĩa gì và xử lý ra sao.

- Những cột nào có missing values?
- Cột nào có missing ratio >= 5%?
- Missing tập trung ở numeric features hay categorical features?
- Missing pattern ở train và test có giống nhau không?
- Có cột nào test missing nhiều nhưng train ít missing không?
- Missing có vẻ là MCAR, MAR hay MNAR?
- Missing có thể mang ý nghĩa “không áp dụng” thay vì “không biết” không?
- Với categorical features, nên điền `"Unknown"` hay mode?
- Với numeric features, nên dùng median thay vì mean vì dữ liệu lệch/outlier không?
- Có nên tạo thêm missing-indicator column cho các feature thiếu nhiều không?
- Có cột nào thiếu quá nhiều đến mức nên cân nhắc loại bỏ không?
- Cách xử lý missing có bị data leakage không, ví dụ fit imputer trên cả train + test?

##### **Output**
- `missing_deep_dive.csv`
- `missing_strategy_proposal.csv`
- `missing_values_barplot.png`

#### 1.6. **Numeric Features**
Hiểu biến số để chuẩn bị scaling, VIF, OLS và Ridge/Lasso.

- Những cột nào là numeric features thật sự?
- Có numeric feature nào thực chất là mã code/category không?
- Numeric features có scale khác nhau rất lớn không?
- Feature nào có phân phối lệch mạnh?
- Feature nào có nhiều outlier?
- Feature nào tương quan mạnh nhất với target?
- Tương quan đó là dương hay âm?
- Quan hệ giữa numeric feature và target có gần tuyến tính không?
- Scatter plot có gợi ý quan hệ phi tuyến không?
- Có nên thử log/sqrt transform cho một số feature lệch phải không?
- Có cặp numeric features nào tương quan quá cao với nhau không?
- Có rủi ro đa cộng tuyến làm OLS coefficient không ổn định không?
- Những feature nào nên đưa vào kiểm tra VIF ở các ngày sau?

##### **Output**
- `numeric_summary.csv`
- `correlation_with_target.csv`
- `correlation_heatmap.png`
- Danh sách feature cần scaling.
- Danh sách feature cần kiểm tra VIF.

#### 1.7 **Categorical Features**
Hiểu cấu trúc category để quyết định one-hot encoding, rare-category handling và tránh nổ số chiều.

- Những cột nào là categorical features?
- Mỗi categorical feature có bao nhiêu unique values?
- Cột nào low-cardinality và phù hợp one-hot encoding?
- Cột nào high-cardinality?
- Có category nào rất hiếm không?
- Có nên gom rare categories thành `"Other"` không?
- Có category nào có median target cao/thấp rõ rệt không?
- Phân phối category giữa train và test có giống nhau không?
- Có category xuất hiện trong test nhưng không xuất hiện trong train không?
- Có category xuất hiện trong train nhưng không xuất hiện trong test không?
- One-hot encoding có tạo quá nhiều cột cho OLS không?
- Có categorical feature nào mang tính thứ tự và nên ordinal encode không?
- Category text có bị lỗi viết hoa/thường, khoảng trắng, spelling không?

##### **Output**
- `categorical_summary.csv`
- `categorical_target_report.csv`
- `target_by_<column>.png`
- Danh sách cột one-hot.
- Danh sách cột cần gom rare category.

#### 1.8 **Train-test-val distribution shift**

Kiểm tra test có khác train không để chọn validation strategy đúng.

- Numeric features trong train và test có phân phối giống nhau không?
- Categorical features trong train và test có tần suất category giống nhau không?
- Missing ratio giữa train và test có giống nhau không?
- Có feature quan trọng nào bị shift mạnh giữa train và test không?
- Có category chỉ xuất hiện trong test không?
- Nếu train/test khác nhau nhiều, random validation split có còn đáng tin không?
- Có cần stratified split theo target bins không?
- Có cột thời gian/thứ tự nào khiến nên split theo thời gian thay vì random không?
- Local validation có khả năng lệch nhiều so với leaderboard không?

##### **Output**
- `train_test_distribution_report.csv`
- `category_mismatch_report.csv`
- Đề xuất validation strategy.

#### 1.9 **Outliers**
Xác định outlier là lỗi, trường hợp thật, hay tín hiệu quan trọng.

- Target có bao nhiêu outlier theo IQR?
- Numeric features nào có nhiều outlier?
- Top target outliers là những dòng nào?
- Các target outlier có logic không trong bối cảnh du lịch?
- Outlier có tập trung ở một nhóm du khách cụ thể không?
- Outlier có thể do lỗi nhập liệu không?
- Outlier có khả năng ảnh hưởng mạnh tới OLS coefficient không?
- Nên giữ, cap/winsorize, log-transform, hay loại bỏ outlier?
- Nếu xử lý outlier, nhóm sẽ chứng minh bằng validation metric như thế nào?
- Nếu loại bỏ/cap outlier, báo cáo cần giải thích lý do ra sao?

##### **Output**
- `outlier_report.csv`
- `top_target_outliers.csv`
- Đề xuất chiến lược outlier: keep / log / winsorize / remove sau khi kiểm chứng.
  
#### 1.10 **Leakage**
Tránh mô hình đạt validation ảo vì dùng thông tin không có tại thời điểm dự đoán.

- Có feature nào trực tiếp hoặc gián tiếp chứa target không?
- Có feature nào chỉ biết được sau khi chuyến đi kết thúc không?
- Có feature nào được tính dựa trên chi phí/target không?
- Có feature nào tương quan bất thường quá cao với target không?
- Có categorical feature nào gần như định danh từng target không?
- Có duplicate hoặc near-duplicate giữa train và test không?
- Imputation, scaling, encoding có được fit chỉ trên train split không?
- Cross-validation sau này có fit preprocessing bên trong từng fold không?
- Có dùng thông tin từ validation/test khi chọn feature hoặc xử lý missing không?

##### **Output**
- `leakage_risk_report.md`
- Danh sách feature cần loại khỏi modeling nếu có rủi ro leakage.
  
#### 1.11 **Interaction và segment insights**
Tìm insight theo nhóm du khách, giúp báo cáo sâu hơn và chuẩn bị interaction features.

- Nhóm du khách nào có median spending cao nhất?
- Chi tiêu có khác nhau theo quốc gia cư trú không?
- Chi tiêu có khác nhau theo mục đích du lịch không?
- Chi tiêu có khác nhau theo package type không?
- Chi tiêu có khác nhau theo số ngày lưu trú không?
- Tổ hợp `purpose × package` có tạo khác biệt lớn về target không?
- Tổ hợp `country × duration` có tạo khác biệt lớn không?
- Có segment nào vừa missing nhiều vừa target cao/thấp bất thường không?
- Có segment nào chứa nhiều outlier target không?
- Interaction nào đủ dễ hiểu để đưa vào báo cáo hoặc thử trong model sau này?
- Nhóm du khách nào có median spending cao nhất?
- Chi tiêu có khác nhau theo quốc gia cư trú không?
- Chi tiêu có khác nhau theo mục đích du lịch không?
- Chi tiêu có khác nhau theo package type không?
- Chi tiêu có khác nhau theo số ngày lưu trú không?
- Tổ hợp `purpose × package` có tạo khác biệt lớn về target không?
- Tổ hợp `country × duration` có tạo khác biệt lớn không?
- Có segment nào vừa missing nhiều vừa target cao/thấp bất thường không?
- Có segment nào chứa nhiều outlier target không?
- Interaction nào đủ dễ hiểu để đưa vào báo cáo hoặc thử trong model sau này?

##### **Output**
- `segment_summary.csv`
- Danh sách interaction candidates.

#### 1.12. **Validation strategy**
Chọn cách validation đáng tin trước khi build model.

- Nên dùng holdout split trước hay k-fold CV ngay?
- Nếu target lệch mạnh, có nên stratify theo target quantiles không?
- Có group nào không nên bị chia lẫn train/validation không?
- Ridge/Lasso sẽ chọn lambda bằng k-fold CV như thế nào?
- Preprocessing có được fit riêng trong từng fold để tránh leakage không?
- Baseline đơn giản nhất cần vượt qua là gì?
- Day 3/Day 4 nên build OLS raw target trước hay OLS log-target trước?

##### **Output**
- `validation_strategy.md`
- Baseline plan cho ngày tiếp theo.

#### 1.13. **Kết luận EDA chuyển thành quyết định preprocessing/modeling**
Mỗi insight EDA phải biến thành một hành động cụ thể.

- Cột nào nên drop vì là ID-like, constant, hoặc leakage-risk?
- Numeric columns nào cần median imputation?
- Categorical columns nào cần fill `"Unknown"`?
- Categorical columns nào one-hot encoding được?
- Categorical columns nào cần rare-category grouping?
- Feature nào cần standardization?
- Feature nào cần kiểm tra VIF?
- Có test `log1p(target)` không?
- Có tạo missing-indicator features không?
- Outlier sẽ được giữ, log-transform, winsorize hay kiểm tra thêm?
- `DataPipeline` Day 3 cần implement những bước nào?
- Model baseline đầu tiên là gì?
- Ridge/Lasso sẽ giải quyết vấn đề nào quan sát được từ EDA?

##### **Output**
- `day2_eda_findings.md`
- `decision_log.md`
- Task list cho Day 3 DataPipeline.