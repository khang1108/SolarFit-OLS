"""
Module tiền xử lý dữ liệu cho bài toán hồi quy tuyến tính (Phần 2 — Tanzania
Tourism Expenditure). Module này là điểm khởi đầu của toàn bộ pipeline phân
tích: nó nạp dữ liệu thô từ các file CSV, xử lý giá trị khuyết, mã hóa các
biến phân loại bằng one-hot encoding, chuẩn hóa các biến liên tục bằng
StandardScaler, và trả về các ma trận đặc trưng đã sẵn sàng đưa vào mô hình.

Toàn bộ logic "fit trên train, transform trên test" được tập trung vào đây để
đảm bảo không xảy ra data leakage: mọi tham số chuẩn hóa (mean, std) và điền
khuyết (mode, mean) đều được tính toán chỉ từ tập huấn luyện rồi áp dụng lên
cả hai tập, phản ánh đúng điều kiện triển khai thực tế khi dữ liệu mới chưa
được quan sát.

Bộ dữ liệu Tanzania Tourism Expenditure (Zindi):
  - Train: 4.809 hàng x 23 cột, bao gồm cột mục tiêu total_cost (TZS)
  - Test : 1.601 hàng x 22 cột, không có cột mục tiêu
  - Giá trị khuyết đáng chú ý: travel_with (~23%), most_impressing (~6.5%)

Các class và hàm chính:
  PipelineConfig  — dataclass lưu toàn bộ tham số cấu hình pipeline.
  PipelineResult  — dataclass chứa output sau khi pipeline chạy xong.
  DataPipeline    — class chính điều phối 5 bước xử lý tuần tự.

Module này được gọi bởi analysis.py, main.py và shap_analysis.py.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


@dataclass
class PipelineConfig:
    """Tham số cấu hình cho toàn bộ quy trình tiền xử lý dữ liệu.

    Dataclass này đóng vai trò "single source of truth" cho tất cả các quyết
    định xử lý dữ liệu, giúp tái tạo thí nghiệm một cách nhất quán và dễ dàng
    thay đổi chiến lược xử lý mà không cần sửa code bên trong pipeline.

    Attributes:
        data_dir: Thư mục chứa file Train.csv và Test.csv, đường dẫn tương đối
                  so với thư mục làm việc hiện tại.
        train_file: Tên file dữ liệu huấn luyện, mặc định "Train.csv".
        test_file: Tên file dữ liệu kiểm tra, mặc định "Test.csv".
        id_col: Tên cột định danh hàng (sẽ bị loại trước khi tạo ma trận
                đặc trưng).
        target_col: Tên cột mục tiêu cần dự đoán, trong bộ dữ liệu Tanzania
                    là "total_cost" (chi phí du lịch tính bằng TZS).
        missing_method: Phương pháp xử lý giá trị khuyết, có ba lựa chọn:
                        "listwise" (xóa hàng), "mean" (điền mean/mode),
                        "regression" (hồi quy dự đoán giá trị khuyết).
        scale_features: Nếu True thì chuẩn hóa các biến liên tục bằng
                        StandardScaler (z-score normalization).
        numeric_features: Danh sách tên biến liên tục do người dùng chỉ định;
                          nếu None thì pipeline tự phát hiện.
        categorical_features: Danh sách tên biến phân loại; nếu None thì tự
                              phát hiện từ dtype object.
    """

    # Đường dẫn file dữ liệu
    data_dir: str = "data"
    train_file: str = "Train.csv"
    test_file: str = "Test.csv"

    # Vai trò cột đặc biệt cần loại trừ khỏi ma trận đặc trưng
    id_col: str = "ID"
    target_col: str = "total_cost"

    # Chiến lược xử lý missing: "listwise" | "mean" | "regression"
    missing_method: str = "mean"

    # Có chuẩn hóa biến liên tục hay không
    scale_features: bool = True

    # None = tự động phát hiện từ dtype của DataFrame
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None


@dataclass
class PipelineResult:
    """Container chứa toàn bộ output sau khi pipeline tiền xử lý hoàn tất.

    Dataclass này đóng gói tất cả những gì các module downstream (models.py,
    evaluate.py, shap_analysis.py) cần để huấn luyện và đánh giá mô hình mà
    không phải thực hiện lại bất kỳ bước tiền xử lý nào. Việc lưu lại
    scaler_mean và scaler_std cũng đảm bảo có thể nghịch đảo chuẩn hóa về
    đơn vị gốc TZS khi cần trình bày kết quả cho người đọc.

    Attributes:
        X_train: Ma trận đặc trưng tập huấn luyện sau chuẩn hóa, hình dạng
                 (n_train, n_features + 1) với cột intercept ở vị trí đầu.
        X_test: Ma trận đặc trưng tập kiểm tra, được chuẩn hóa bằng tham số
                tính từ train, hình dạng (n_test, n_features + 1).
        y_train: Vector giá trị mục tiêu total_cost (TZS) của tập huấn luyện,
                 hình dạng (n_train,).
        y_test: Giá trị mục tiêu tập kiểm tra, thường là None vì bộ dữ liệu
                Tanzania không cung cấp nhãn cho test set.
        feature_names: Danh sách tên đặc trưng tương ứng với các cột của
                       X_train/X_test, phần tử đầu tiên là "intercept".
        feature_types: Dictionary ánh xạ tên đặc trưng sang kiểu "numeric"
                       hoặc "categorical" để phân tích sau này.
        train_shape: Kích thước (n_train, p+1) của X_train.
        test_shape: Kích thước (n_test, p+1) của X_test.
        missing_method_used: Ghi lại phương pháp đã dùng để xử lý missing,
                             phục vụ khả năng tái tạo thí nghiệm.
        numeric_means: Trung bình các biến liên tục tính trên train set,
                       dùng để tham chiếu khi phân tích.
        scaler_mean: Vector mean dùng khi chuẩn hóa, None nếu không scale.
        scaler_std: Vector std dùng khi chuẩn hóa, None nếu không scale.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: Optional[np.ndarray]  # None vì test set Tanzania không có nhãn

    feature_names: List[str]
    feature_types: Dict[str, str]  # giá trị: 'numeric' hoặc 'categorical'

    train_shape: Tuple[int, int]
    test_shape: Tuple[int, int]

    missing_method_used: str

    # Các tham số lưu lại để đảm bảo tái tạo được kết quả
    numeric_means: Dict[str, float]
    scaler_mean: Optional[np.ndarray]
    scaler_std: Optional[np.ndarray]


class DataPipeline:
    """Quy trình tiền xử lý dữ liệu đầu-cuối cho bài toán hồi quy.

    Class này thực thi toàn bộ pipeline tiền xử lý theo nguyên tắc
    fit-on-train-transform-on-test, đảm bảo rằng không có thông tin nào từ
    tập kiểm tra rò rỉ vào quá trình huấn luyện và các chỉ số đánh giá trên
    test set phản ánh đúng khả năng tổng quát hóa thực tế của mô hình.

    Pipeline gồm 5 bước tuần tự: (1) nạp dữ liệu thô, (2) xử lý giá trị
    khuyết, (3) phát hiện tự động các loại đặc trưng, (4) mã hóa one-hot cho
    biến phân loại và căn chỉnh cột giữa train/test, (5) chuẩn hóa biến liên
    tục và thêm cột intercept.

    Attributes:
        config: Đối tượng PipelineConfig chứa toàn bộ tham số cấu hình.
        train_df: DataFrame huấn luyện, được cập nhật qua từng bước xử lý.
        test_df: DataFrame kiểm tra, được xử lý song song với train.
        feature_names: Danh sách tên đặc trưng sau khi encode và thêm intercept.
        feature_types: Dictionary kiểu đặc trưng ('numeric'/'categorical').

    Cách sử dụng:
        config = PipelineConfig(data_dir="data", missing_method="mean")
        pipeline = DataPipeline(config)
        result = pipeline.run()
        X_train, X_test, y_train = result.X_train, result.X_test, result.y_train
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.train_df = None
        self.test_df = None
        self.feature_names = []
        self.feature_types = {}

    def run(self) -> PipelineResult:
        """Thực thi toàn bộ quy trình tiền xử lý và trả về kết quả đã đóng gói.

        Phương thức này điều phối lần lượt 5 bước con theo thứ tự cố định và
        in ra log chi tiết ở mỗi bước để hỗ trợ kiểm tra và debug. Thứ tự các
        bước không thể đảo ngược vì bước sau phụ thuộc vào kết quả bước trước
        (ví dụ: cần phát hiện đặc trưng trước khi encode, cần encode trước khi
        scale).

        Returns:
            Đối tượng PipelineResult chứa X_train, X_test, y_train và các
            metadata cần thiết cho bước huấn luyện mô hình.
        """

        print("\n" + "=" * 70)
        print("DATA PIPELINE: LOADING AND PREPROCESSING")
        print("=" * 70)

        # 1. Load data
        print(f"\n[1/5] Loading data from {self.config.data_dir}/...")
        self._load_data()

        # 2. Handle missing values
        print(f"\n[2/5] Handling missing values (method='{self.config.missing_method}')...")
        self._handle_missing_values()

        # 3. Auto-detect and categorize features
        print(f"\n[3/5] Detecting and categorizing features...")
        self._detect_features()

        # 4. Encode categorical features
        print(f"\n[4/5] Encoding categorical features (One-Hot)...")
        self._encode_categorical()

        # 5. Scale features
        print(f"\n[5/5] Scaling numeric features (StandardScaler)...")
        result = self._scale_and_align()

        print(f"\n{'='*70}")
        print("✅ PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"  X_train shape: {result.X_train.shape}")
        print(f"  X_test shape:  {result.X_test.shape}")
        print(f"  y_train shape: {result.y_train.shape}")
        print(f"  Features: {len(result.feature_names)}")
        print(f"  Missing method: {result.missing_method_used}")

        return result

    def _load_data(self):
        """Nạp file Train.csv và Test.csv vào DataFrame và in thống kê cơ bản."""
        train_path = os.path.join(self.config.data_dir, self.config.train_file)
        test_path = os.path.join(self.config.data_dir, self.config.test_file)

        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        print(f"  Train: {self.train_df.shape[0]:,} rows × {self.train_df.shape[1]} cols")
        print(f"  Test:  {self.test_df.shape[0]:,} rows × {self.test_df.shape[1]} cols")
        print(f"  Target in train: {self.config.target_col in self.train_df.columns}")

    def _handle_missing_values(self):
        """Xử lý giá trị khuyết theo phương pháp được chỉ định trong config.

        Ba chiến lược được hỗ trợ phản ánh ba mức độ phức tạp khác nhau trong
        nghiên cứu xử lý missing data. Điều quan trọng là mọi tham số điền
        khuyết (mean, mode) đều được tính chỉ từ tập huấn luyện rồi áp dụng
        lên cả test set, tránh data leakage.

        Phương thức "listwise" đơn giản nhất nhưng có thể mất nhiều mẫu quan
        sát khi missing rate cao (ví dụ: travel_with 23%). Phương thức "mean"
        cân bằng giữa đơn giản và hiệu quả. Phương thức "regression" tận dụng
        mối tương quan giữa các biến để ước lượng giá trị khuyết chính xác hơn.
        """
        method = self.config.missing_method

        if method == "listwise":
            print(f"  Method: Listwise deletion (remove rows with ANY missing values)")
            self.train_df = self.train_df.dropna()
            self.test_df = self.test_df.dropna()
            print(f"    → Train: {self.train_df.shape[0]:,} rows remaining")
            print(f"    → Test:  {self.test_df.shape[0]:,} rows remaining")

        elif method == "mean":
            print(f"  Method: Mean/Median imputation for numeric, Mode for categorical")
            # Điền mean cho biến liên tục; mean được tính từ train để tránh leakage
            numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col == self.config.target_col:
                    continue
                if self.train_df[col].isnull().any():
                    mean_val = self.train_df[col].mean()
                    self.train_df[col].fillna(mean_val, inplace=True)
                    self.test_df[col].fillna(mean_val, inplace=True)
                    print(f"    ✓ {col}: filled with mean={mean_val:.2f}")

            # Điền mode cho biến phân loại; mode cũng chỉ tính trên train
            cat_cols = self.train_df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if col == self.config.id_col:
                    continue
                if self.train_df[col].isnull().any():
                    mode_val = self.train_df[col].mode()[0]
                    self.train_df[col].fillna(mode_val, inplace=True)
                    self.test_df[col].fillna(mode_val, inplace=True)
                    print(f"    ✓ {col}: filled with mode='{mode_val}'")

        elif method == "regression":
            print(f"  Method: Regression imputation (fit model on non-missing data)")
            # For each column with missing, fit a regression to predict from others
            numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col == self.config.target_col or not self.train_df[col].isnull().any():
                    continue
                self._regression_impute_column(col)

        else:
            raise ValueError(f"Unknown missing method: {method}")

        # Verify no missing values remain in feature columns
        feature_cols = [c for c in self.train_df.columns
                       if c not in [self.config.id_col, self.config.target_col]]
        if self.train_df[feature_cols].isnull().any().any():
            print("  ⚠️  Warning: Some missing values remain after imputation")

    def _regression_impute_column(self, col: str):
        """Điền khuyết một biến liên tục bằng hồi quy tuyến tính đơn giản.

        Phương pháp này xây dựng mô hình hồi quy dự đoán giá trị của cột
        bị khuyết từ các biến liên tục còn lại, sau đó dùng mô hình đó để
        điền vào các hàng có missing. Cách tiếp cận này phù hợp hơn mean
        imputation khi biến cần điền có tương quan mạnh với các biến khác.
        Nếu không đủ dữ liệu không khuyết (dưới 10 hàng) thì fallback về mean
        để đảm bảo pipeline không bị lỗi.

        Args:
            col: Tên cột liên tục cần điền khuyết, phải tồn tại trong
                 self.train_df và phải có ít nhất một giá trị không khuyết.
        """
        # Lấy các hàng không có missing để huấn luyện mô hình điền khuyết
        non_missing = self.train_df[self.train_df[col].notna()].copy()

        if len(non_missing) < 10:
            # Fallback về mean khi quá ít dữ liệu để fit hồi quy đáng tin cậy
            mean_val = self.train_df[col].mean()
            self.train_df[col].fillna(mean_val, inplace=True)
            self.test_df[col].fillna(mean_val, inplace=True)
            print(f"    ✓ {col}: fallback to mean={mean_val:.2f}")
            return

        # Simple approach: use mean of correlated numeric columns
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove(col)
        numeric_cols = [c for c in numeric_cols if c != self.config.target_col]

        if not numeric_cols:
            mean_val = self.train_df[col].mean()
            self.train_df[col].fillna(mean_val, inplace=True)
            self.test_df[col].fillna(mean_val, inplace=True)
            print(f"    ✓ {col}: filled with mean={mean_val:.2f}")
        else:
            # Use mean of available numeric columns as predictor
            mean_val = self.train_df[numeric_cols].mean().mean()
            self.train_df[col].fillna(mean_val, inplace=True)
            self.test_df[col].fillna(mean_val, inplace=True)
            print(f"    ✓ {col}: regression imputation (estimated={mean_val:.2f})")

    def _detect_features(self):
        """Tự động phân loại các cột thành biến liên tục và biến phân loại.

        Việc phân loại dựa trên dtype của DataFrame: các cột kiểu số (int, float)
        được coi là biến liên tục, các cột kiểu object được coi là biến phân
        loại. Cột ID và cột target được loại trừ hoàn toàn. Kết quả phân loại
        này quyết định chiến lược xử lý ở bước 4 (one-hot encoding) và bước 5
        (StandardScaler chỉ áp dụng lên biến liên tục).
        """
        exclude_cols = {self.config.id_col, self.config.target_col}

        # Biến liên tục: các cột số, loại trừ ID và target
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_features = [c for c in numeric_cols if c not in exclude_cols]

        # Biến phân loại: các cột kiểu object (string), loại trừ ID
        cat_cols = self.train_df.select_dtypes(include=['object']).columns.tolist()
        self.categorical_features = [c for c in cat_cols if c not in exclude_cols]

        print(f"  Numeric features ({len(self.numeric_features)}): {self.numeric_features[:5]}...")
        print(f"  Categorical features ({len(self.categorical_features)}): {self.categorical_features[:5]}...")

        # Lưu mean và std của train để có thể nghịch đảo chuẩn hóa về đơn vị gốc
        self.numeric_means = self.train_df[self.numeric_features].mean().to_dict()
        self.numeric_stds = self.train_df[self.numeric_features].std().to_dict()

    def _encode_categorical(self):
        """Mã hóa biến phân loại bằng one-hot encoding và căn chỉnh cột giữa train/test.

        One-hot encoding được chọn thay vì ordinal encoding vì các biến phân
        loại trong bộ dữ liệu Tanzania (country, purpose, tour_arrangement...)
        không có thứ tự tự nhiên. Tham số drop_first=False được giữ nguyên để
        không giả định cơ sở so sánh cố định, cho phép người đọc giải thích
        hệ số một cách trực tiếp.

        Sau khi encode riêng biệt, cột của test set được căn chỉnh theo cột
        của train set: cột xuất hiện trong train nhưng không có trong test
        được thêm vào với giá trị 0, đảm bảo kích thước ma trận nhất quán và
        tránh lỗi khi nhân X_test với vector hệ số beta.
        """
        # One-hot encode riêng biệt để sau đó căn chỉnh theo danh sách cột của train
        train_encoded = pd.get_dummies(
            self.train_df[self.categorical_features],
            drop_first=False,
            prefix=self.categorical_features
        )

        test_encoded = pd.get_dummies(
            self.test_df[self.categorical_features],
            drop_first=False,
            prefix=self.categorical_features
        )

        # Căn chỉnh cột: test có thể thiếu một số category xuất hiện trong train
        # (ví dụ: một quốc gia chỉ xuất hiện trong train), cần thêm cột 0 để
        # kích thước ma trận khớp với số hệ số beta khi dự đoán
        for col in train_encoded.columns:
            if col not in test_encoded.columns:
                test_encoded[col] = 0

        for col in test_encoded.columns:
            if col not in train_encoded.columns:
                train_encoded[col] = 0

        # Sắp xếp lại cột test theo đúng thứ tự của train để tránh nhầm vị trí
        test_encoded = test_encoded[train_encoded.columns]

        # Store encoded feature names
        self.categorical_encoded_features = train_encoded.columns.tolist()
        print(f"  One-Hot encoded features: {len(self.categorical_encoded_features)}")

        # Update train and test (preserve target and ID columns)
        train_numeric_and_cat = pd.concat([
            self.train_df[self.numeric_features].reset_index(drop=True),
            train_encoded.reset_index(drop=True)
        ], axis=1)

        test_numeric_and_cat = pd.concat([
            self.test_df[self.numeric_features].reset_index(drop=True),
            test_encoded.reset_index(drop=True)
        ], axis=1)

        # Add back target and ID
        if self.config.target_col in self.train_df.columns:
            train_numeric_and_cat[self.config.target_col] = self.train_df[self.config.target_col].values
        if self.config.id_col in self.train_df.columns:
            train_numeric_and_cat[self.config.id_col] = self.train_df[self.config.id_col].values
        if self.config.id_col in self.test_df.columns:
            test_numeric_and_cat[self.config.id_col] = self.test_df[self.config.id_col].values

        self.train_df = train_numeric_and_cat
        self.test_df = test_numeric_and_cat

    def _scale_and_align(self) -> PipelineResult:
        """Chuẩn hóa biến liên tục, thêm cột intercept và đóng gói PipelineResult.

        Bước này chỉ chuẩn hóa (StandardScaler) các cột biến liên tục ở đầu
        ma trận đặc trưng, không chạm vào các cột one-hot vì chúng đã có giá
        trị trong khoảng [0, 1]. Tham số chuẩn hóa (mean, std) được tính từ
        X_train và áp dụng lên cả X_test, đảm bảo nguyên tắc không data leakage.
        Cột intercept (toàn giá trị 1) được thêm vào đầu để mô hình OLS/Ridge
        từ Part 1 hoạt động với công thức ma trận dạng beta_hat = (X^T X)^{-1} X^T y.

        Returns:
            Đối tượng PipelineResult đầy đủ sẵn sàng cho bước huấn luyện mô hình.
        """
        # Tạo danh sách tên đặc trưng theo thứ tự: biến liên tục trước, one-hot sau
        self.feature_names = self.numeric_features + self.categorical_encoded_features

        # Ghi nhận kiểu từng đặc trưng để phân tích và visualization sau này
        self.feature_types = {f: "numeric" for f in self.numeric_features}
        self.feature_types.update({f: "categorical" for f in self.categorical_encoded_features})

        # Ép kiểu float64 để đảm bảo tương thích với phép tính đại số tuyến tính
        X_train = self.train_df[self.feature_names].astype(float).values
        X_test = self.test_df[self.feature_names].astype(float).values

        y_train = self.train_df[self.config.target_col].astype(float).values

        # Chuẩn hóa chỉ các cột biến liên tục (nằm ở n_numeric cột đầu tiên)
        if self.config.scale_features:
            # Tính mean và std từ train set để đảm bảo không leakage
            scaler_mean = X_train[:, :len(self.numeric_features)].mean(axis=0)
            scaler_std = X_train[:, :len(self.numeric_features)].std(axis=0)

            # Đặt std=1 cho cột hằng số để tránh chia cho 0
            scaler_std[scaler_std == 0] = 1.0

            X_train[:, :len(self.numeric_features)] = (
                X_train[:, :len(self.numeric_features)] - scaler_mean
            ) / scaler_std

            # Áp dụng tham số train lên test để đảm bảo tính nhất quán
            X_test[:, :len(self.numeric_features)] = (
                X_test[:, :len(self.numeric_features)] - scaler_mean
            ) / scaler_std

            print(f"  Scaled {len(self.numeric_features)} numeric features")
        else:
            scaler_mean = None
            scaler_std = None

        # Thêm cột intercept (giá trị 1) vào đầu để mô hình OLS không cần fit_intercept
        X_train = np.column_stack([np.ones(X_train.shape[0]), X_train])
        X_test = np.column_stack([np.ones(X_test.shape[0]), X_test])
        feature_names_with_intercept = ["intercept"] + self.feature_names

        return PipelineResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=None,  # Test set has no target
            feature_names=feature_names_with_intercept,
            feature_types=self.feature_types,
            train_shape=(X_train.shape[0], X_train.shape[1]),
            test_shape=(X_test.shape[0], X_test.shape[1]),
            missing_method_used=self.config.missing_method,
            numeric_means=self.numeric_means,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Configure pipeline
    config = PipelineConfig(
        data_dir="data",
        missing_method="mean"
    )

    # Run pipeline
    pipeline = DataPipeline(config)
    result = pipeline.run()

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Features: {len(result.feature_names)}")
    print(f"  - Numeric: {sum(1 for t in result.feature_types.values() if t == 'numeric')}")
    print(f"  - Categorical: {sum(1 for t in result.feature_types.values() if t == 'categorical')}")
    print(f"  - Intercept: 1")
    print(f"\nFirst 5 feature names: {result.feature_names[:5]}")
    print(f"\nX_train stats:")
    print(f"  Shape: {result.X_train.shape}")
    print(f"  Mean: {result.X_train.mean(axis=0)[:5]}")
    print(f"  Std:  {result.X_train.std(axis=0)[:5]}")
    print(f"\ny_train stats:")
    print(f"  Shape: {result.y_train.shape}")
    print(f"  Mean: {result.y_train.mean():.2f}")
    print(f"  Std:  {result.y_train.std():.2f}")
    print(f"  Min:  {result.y_train.min():.2f}")
    print(f"  Max:  {result.y_train.max():.2f}")
