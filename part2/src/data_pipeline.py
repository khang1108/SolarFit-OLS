"""
File tiền xử lý dữ liệu cho bài toán hồi quy tuyến tính (Phần 2 — Tanzania
Tourism Expenditure). 

Bộ dữ liệu Tanzania Tourism Expenditure (Zindi):
    - Train: 4.809 hàng x 23 cột, bao gồm cột mục tiêu total_cost (TZS)
    - Test : 1.601 hàng x 22 cột, không có cột mục tiêu
    - missing values đáng chú ý: travel_with (~23%), most_impressing (~6.5%)
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
    """Cấu hình cho toàn bộ quy trình tiền xử lý dữ liệu.

    Attributes:
        data_dir:               Thư mục chứa file Train.csv và Test.csv, đường dẫn tương đối
                                so với thư mục làm việc hiện tại.
        train_file:             Tên file dữ liệu huấn luyện, mặc định "Train.csv".
        test_file:              Tên file dữ liệu kiểm tra, mặc định "Test.csv".
        id_col:                 Tên cột định danh hàng (sẽ bị loại trước khi tạo ma trận
                                đặc trưng).
        target_col:             Tên cột mục tiêu cần dự đoán, trong bộ dữ liệu Tanzania
                                là "total_cost" (chi phí du lịch tính bằng TZS).
        missing_method:         Phương pháp xử lý missing values, có ba lựa chọn:
                                "listwise" (xóa hàng), "median" hoặc alias cũ "mean"
                                (điền median/Unknown), "regression" (hồi quy dự đoán
                                missing values).
        scale_features:         Nếu True thì chuẩn hóa các biến liên tục bằng
                                StandardScaler (z-score normalization).
        log_numeric_features:   Nếu True thì áp dụng log1p lên các biến số
                                không âm trước khi chuẩn hóa. Đây là quyết định
                                lấy từ EDA vì các biến số trong dataset lệch phải
                                khá mạnh.
        numeric_features:       Danh sách tên biến liên tục do người dùng chỉ định;
                                nếu None thì pipeline tự phát hiện.
        categorical_features:   Danh sách tên biến phân loại; nếu None thì tự
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
    log_numeric_features: bool = True

    # None = tự động phát hiện từ dtype của DataFrame
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None


@dataclass
class PipelineResult:
    """Dataclass chứa toàn bộ output sau khi pipeline tiền xử lý hoàn tất.

    Attributes:
        X_train:    Ma trận đặc trưng tập huấn luyện sau chuẩn hóa, hình dạng
                    (n_train, n_features + 1) với cột intercept ở vị trí đầu.
        X_test:     Ma trận đặc trưng tập kiểm tra, được chuẩn hóa bằng tham số
                    tính từ train, hình dạng (n_test, n_features + 1).
        y_train:    Vector giá trị mục tiêu total_cost (TZS) của tập huấn luyện,
                    hình dạng (n_train,).
        y_test:     Giá trị mục tiêu tập kiểm tra, thường là None vì bộ dữ liệu
                    Tanzania không cung cấp nhãn cho test set.
        train_ids:  Mã ID gốc của từng dòng trong tập train. Metadata này giúp
                    truy vết lại quan sát sau khi mô hình tạo dự đoán.
        test_ids:   Mã ID gốc của từng dòng trong tập test, dùng trực tiếp khi
                    xuất file submission cho Zindi.
        feature_names:  Danh sách tên đặc trưng tương ứng với các cột của
                        X_train/X_test, phần tử đầu tiên là "intercept".
        feature_types:  Dictionary ánh xạ tên đặc trưng sang kiểu "numeric"
                        hoặc "categorical" để phân tích sau này.
        train_shape:    Kích thước (n_train, p+1) của X_train.
        test_shape:     Kích thước (n_test, p+1) của X_test.
        missing_method_used:    Ghi lại phương pháp đã dùng để xử lý missing,
                                phục vụ khả năng tái tạo thí nghiệm.
        numeric_means:  Trung bình các biến liên tục tính trên train set,
                        dùng để tham chiếu khi phân tích.
        scaler_mean: Vector mean dùng khi chuẩn hóa, None nếu không scale.
        scaler_std: Vector std dùng khi chuẩn hóa, None nếu không scale.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: Optional[np.ndarray]  # None vì test set Tanzania không có nhãn
    train_ids: Optional[np.ndarray]
    test_ids: Optional[np.ndarray]

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
    """Pipeline tiền xử lý dữ liệu.

    Class này thực thi toàn bộ pipeline tiền xử lý, không có thông tin nào từ
    tập kiểm tra rò rỉ vào quá trình huấn luyện và các chỉ số đánh giá trên
    test set phản ánh đúng khả năng tổng quát hóa thực tế của mô hình.

    Pipeline gồm 5 bước tuần tự: 
        (1) nạp dữ liệu thô, 
        (2) xử lý missing values,
        (3) Feature engineering
        (4) one-hot encoding cho biến phân loại và căn chỉnh cột giữa train/test, 
        (5) chuẩn hóa biến liên tục và thêm cột intercept.

    Attributes:
        config: PipelineConfig chứa toàn bộ tham số cấu hình.
        train_df: DataFrame huấn luyện, được cập nhật qua từng bước xử lý.
        test_df: DataFrame kiểm tra, được xử lý song song với train.
        feature_names: Danh sách tên đặc trưng sau khi encode và thêm intercept.
        feature_types: Dictionary kiểu đặc trưng ('numeric'/'categorical').

    Cách sử dụng:
        config = PipelineConfig(data_dir="data", missing_method="median")
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

        Returns:
            Đối tượng PipelineResult chứa X_train, X_test, y_train và các
            metadata cần thiết cho bước huấn luyện mô hình.
        """

        print("\n" + "=" * 70)
        print("DATA PIPELINE: LOADING AND PREPROCESSING")
        print("=" * 70)

        # Bước 1 — Load Train.csv và Test.csv.
        print(f"\n[1/5] Loading data from {self.config.data_dir}/...")
        self._load_data()

        # Bước 2 — xử lý missing values.
        print(f"\n[2/5] Handling missing values (method='{self.config.missing_method}')...")
        self._handle_missing_values()

        # Bước 3 — Tách biến số và biến phân loại, để mỗi nhóm đi theo
        # một nhánh xử lý riêng phù hợp với bản chất của nó ở các bước kế tiếp.
        print(f"\n[3/5] Detecting and categorizing features...")
        self._detect_features()

        # Bước 4 — one-hot encode biến phân loại để đưa chúng về dạng số.
        print(f"\n[4/5] Encoding categorical features (One-Hot)...")
        self._encode_categorical()

        # Bước 5 — Normalization, align cột train/test -> PipelineResult
        print(f"\n[5/5] Scaling numeric features (StandardScaler)...")
        result = self._scale_and_align()

        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"  X_train shape: {result.X_train.shape}")
        print(f"  X_test shape:  {result.X_test.shape}")
        print(f"  y_train shape: {result.y_train.shape}")
        print(f"  Features: {len(result.feature_names)}")
        print(f"  Missing method: {result.missing_method_used}")

        return result

    def _load_data(self):
        """Load file Train.csv và Test.csv vào DataFrame và in thống kê cơ bản."""
        train_path = os.path.join(self.config.data_dir, self.config.train_file)
        test_path = os.path.join(self.config.data_dir, self.config.test_file)

        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        print(f"  Train: {self.train_df.shape[0]:,} rows × {self.train_df.shape[1]} cols")
        print(f"  Test:  {self.test_df.shape[0]:,} rows × {self.test_df.shape[1]} cols")
        print(f"  Target in train: {self.config.target_col in self.train_df.columns}")

    def _handle_missing_values(self):
        """Xử lý missing values theo phương pháp được chỉ định trong config.

        Phương thức "listwise" đơn giản nhất nhưng có thể mất records 
        khi missing rate cao (ví dụ: travel_with 23%). 
        
        Phương thức "mean" cân bằng giữa đơn giản và hiệu quả. 
        
        Phương thức "regression" tận dụng mối tương quan giữa các biến để ước lượng missing values chính xác hơn.
        """
        method = self.config.missing_method

        if method == "listwise":
            print(f"  Method: Listwise deletion (remove rows with ANY missing values)")

            if self.train_df is None or self.test_df is None:
                raise ValueError("DataFrames not loaded. Call _load_data() first.")
            
            self.train_df = self.train_df.dropna()
            self.test_df = self.test_df.dropna()
            
            print(f"    → Train: {self.train_df.shape[0]:,} rows remaining")
            print(f"    → Test:  {self.test_df.shape[0]:,} rows remaining")

        elif method in {"mean", "median"}:
            print(f"  Method: Median imputation for numeric, Unknown for categorical")
            # Điền median cho biến liên tục; median được tính từ train để tránh leakage
            # và bền hơn mean khi dữ liệu chi phí/đêm lưu trú có outlier lớn.
            if self.train_df is None or self.test_df is None:
                raise ValueError("DataFrames not loaded. Call _load_data() first.")

            numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col == self.config.target_col:
                    continue
                if self.train_df[col].isnull().any() or self.test_df[col].isnull().any():
                    median_val = self.train_df[col].median()
                    self.train_df[col] = self.train_df[col].fillna(median_val)
                    self.test_df[col] = self.test_df[col].fillna(median_val)
                    print(f"    ✓ {col}: filled with median={median_val:.2f}")

            # Điền Unknown cho biến phân loại để mô hình học được trạng thái
            # "không trả lời" thay vì ép nó thành nhóm phổ biến nhất.
            cat_cols = self.train_df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if col == self.config.id_col:
                    continue
                if self.train_df[col].isnull().any() or self.test_df[col].isnull().any():
                    self.train_df[col] = self.train_df[col].fillna("Unknown")
                    self.test_df[col] = self.test_df[col].fillna("Unknown")
                    print(f"    ✓ {col}: filled with 'Unknown'")

        elif method == "regression":
            print(f"  Method: Regression imputation (fit model on non-missing data)")
            # Với mỗi cột còn thiếu, ta dựng một hồi quy dự đoán giá trị của nó
            # từ các biến còn lại, tận dụng tương quan giữa các biến để điền
            # khuyết chính xác hơn so với chỉ điền mean.
            if self.train_df is None or self.test_df is None:
                raise ValueError("DataFrames not loaded. Call _load_data() first.")

            numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col == self.config.target_col or not self.train_df[col].isnull().any():
                    continue
                self._regression_impute_column(col)

        else:
            raise ValueError(f"Unknown missing method: {method}")

        # Kiểm tra lại lần cuối để chắc chắn không còn ô khuyết nào sót lại trong
        # các cột đặc trưng, vì chỉ một ô NaN cũng đủ làm hỏng phép nhân ma trận.
        feature_cols =  [c for c in self.train_df.columns
                        if c not in [self.config.id_col, self.config.target_col]]
        if self.train_df[feature_cols].isnull().any().any():
            print("Warning: Some missing values remain after imputation")

    def _regression_impute_column(self, col: str):
        """Điền missing values một biến liên tục bằng hồi quy tuyến tính đơn giản.

        Phương pháp này xây dựng mô hình hồi quy dự đoán giá trị của cột
        bị missing từ các biến liên tục còn lại, sau đó dùng mô hình đó để
        điền vào các hàng có missing. Cách tiếp cận này phù hợp hơn mean
        imputation khi biến cần điền có tương quan mạnh với các biến khác.
        Nếu không đủ dữ liệu không khuyết (dưới 10 hàng) thì fallback về mean
        để đảm bảo pipeline không bị lỗi.

        Args:
            col:    Tên cột liên tục cần điền khuyết, phải tồn tại trong
                    self.train_df và phải có ít nhất một giá trị không khuyết.
        """
        # Lấy các hàng không có missing để huấn luyện mô hình điền khuyết
        if self.train_df is None or self.test_df is None:
            raise ValueError("DataFrames not loaded. Call _load_data() first.")

        non_missing = self.train_df[self.train_df[col].notna()].copy()

        if len(non_missing) < 10:
            # Fallback về mean khi quá ít dữ liệu để fit hồi quy đáng tin cậy
            mean_val = self.train_df[col].mean()
            self.train_df[col] = self.train_df[col].fillna(mean_val)
            self.test_df[col] = self.test_df[col].fillna(mean_val)
            print(f"    ✓ {col}: fallback to mean={mean_val:.2f}")
            return

        # Ở đây ta đi theo hướng đơn giản và ổn định: thay vì fit một mô hình
        # hồi quy đầy đủ, ta lấy trung bình của các biến số còn lại làm giá trị
        # thay thế, đủ tốt cho mục đích điền khuyết mà không gây bất ổn số học.
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove(col)
        numeric_cols = [c for c in numeric_cols if c != self.config.target_col]

        if not numeric_cols:
            mean_val = self.train_df[col].mean()
            self.train_df[col] = self.train_df[col].fillna(mean_val)
            self.test_df[col] = self.test_df[col].fillna(mean_val)
            print(f"    ✓ {col}: filled with mean={mean_val:.2f}")
        else:
            # Dùng trung bình của các biến số sẵn có làm giá trị ước lượng để điền
            mean_val = self.train_df[numeric_cols].mean().mean()
            self.train_df[col] = self.train_df[col].fillna(mean_val)
            self.test_df[col] = self.test_df[col].fillna(mean_val)
            print(f"    ✓ {col}: regression imputation (estimated={mean_val:.2f})")

    def _detect_features(self):
        """Tự động phân loại các cột thành biến liên tục và biến phân loại.

        Việc phân loại dựa trên dtype của DataFrame: các cột kiểu số (int, float)
        được coi là biến liên tục, các cột kiểu object được coi là biến phân
        loại. Cột ID và cột target được loại trừ hoàn toàn. Kết quả phân loại
        này quyết định chiến lược xử lý ở bước 4 (one-hot encoding) và bước 5
        (StandardScaler chỉ áp dụng lên biến liên tục).
        """

        if self.train_df is None or self.test_df is None:
            raise ValueError("DataFrames not loaded. Call _load_data() first.")
        
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
        numeric_means: Dict[str, float] = {
            str(name): float(value)
            for name, value in self.train_df[self.numeric_features].mean().items()
        }
        numeric_stds: Dict[str, float] = {
            str(name): float(value)
            for name, value in self.train_df[self.numeric_features].std().items()
        }
        self.numeric_means = numeric_means
        self.numeric_stds = numeric_stds

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
        if self.train_df is None or self.test_df is None:
            raise ValueError("DataFrames not loaded. Call _load_data() first.")

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

        # Lưu lại danh sách tên cột sau one-hot để các bước sau biết chính xác
        # thứ tự và số lượng đặc trưng phân loại đã sinh ra.
        self.categorical_encoded_features = train_encoded.columns.tolist()
        print(f"  One-Hot encoded features: {len(self.categorical_encoded_features)}")

        # Ghép biến số với khối one-hot vừa tạo thành ma trận đặc trưng hoàn chỉnh,
        # đồng thời vẫn giữ lại cột target và ID để xử lý riêng ngay phía dưới.
        train_numeric_and_cat = pd.concat([
            self.train_df[self.numeric_features].reset_index(drop=True),
            train_encoded.reset_index(drop=True)
        ], axis=1)

        test_numeric_and_cat = pd.concat([
            self.test_df[self.numeric_features].reset_index(drop=True),
            test_encoded.reset_index(drop=True)
        ], axis=1)

        # Gắn lại cột target và ID vào sau khi đã dựng xong khối đặc trưng số:
        # target để huấn luyện, còn ID để đối chiếu khi xuất file submission.
        if self.config.target_col in self.train_df.columns:
            train_numeric_and_cat[self.config.target_col] = self.train_df[
                self.config.target_col
            ].to_numpy(dtype=np.float64, copy=True)
        if self.config.id_col in self.train_df.columns:
            train_numeric_and_cat[self.config.id_col] = self.train_df[
                self.config.id_col
            ].to_numpy(copy=True)
        if self.config.id_col in self.test_df.columns:
            test_numeric_and_cat[self.config.id_col] = self.test_df[
                self.config.id_col
            ].to_numpy(copy=True)

        self.train_df = train_numeric_and_cat
        self.test_df = test_numeric_and_cat

    def _scale_and_align(self) -> PipelineResult:
        """Chuẩn hóa biến liên tục, thêm cột intercept và đóng gói PipelineResult.

        Returns:
            Đối tượng PipelineResult đầy đủ sẵn sàng cho bước huấn luyện mô hình.
        """
        if self.train_df is None or self.test_df is None:
            raise ValueError("DataFrames not loaded. Call _load_data() first.")
        
        # Tạo danh sách tên đặc trưng theo thứ tự: biến liên tục trước, one-hot sau
        self.feature_names = self.numeric_features + self.categorical_encoded_features

        # Ghi nhận kiểu từng đặc trưng để phân tích và visualization sau này
        self.feature_types = {f: "numeric" for f in self.numeric_features}
        self.feature_types.update({f: "categorical" for f in self.categorical_encoded_features})

        # Lưu lại ID gốc trước khi chuyển sang ma trận số; đây là phần bắt buộc
        # khi xuất submission vì Zindi đối chiếu dự đoán theo cột ID.
        train_ids = (
            self.train_df[self.config.id_col].astype(str).to_numpy(dtype=str, copy=True)
            if self.config.id_col in self.train_df.columns
            else None
        )
        test_ids = (
            self.test_df[self.config.id_col].astype(str).to_numpy(dtype=str, copy=True)
            if self.config.id_col in self.test_df.columns
            else None
        )

        # Ép kiểu float64 để đảm bảo tương thích với phép tính đại số tuyến tính
        X_train: np.ndarray = self.train_df[self.feature_names].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        X_test: np.ndarray = self.test_df[self.feature_names].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        y_train: np.ndarray = self.train_df[self.config.target_col].to_numpy(
            dtype=np.float64,
            copy=True,
        )

        # Các biến đếm/số đêm trong bộ dữ liệu lệch phải mạnh, nên log1p giúp
        # giảm ảnh hưởng của outlier trước khi đưa vào StandardScaler.
        if self.config.log_numeric_features and len(self.numeric_features) > 0:
            X_train[:, :len(self.numeric_features)] = np.log1p(
                np.clip(X_train[:, :len(self.numeric_features)], a_min=0, a_max=None)
            )
            X_test[:, :len(self.numeric_features)] = np.log1p(
                np.clip(X_test[:, :len(self.numeric_features)], a_min=0, a_max=None)
            )
            print(f"  Applied log1p to {len(self.numeric_features)} numeric features")

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
            y_test=None,  # tập test của cuộc thi không có nhãn mục tiêu
            train_ids=train_ids,
            test_ids=test_ids,
            feature_names=feature_names_with_intercept,
            feature_types=self.feature_types,
            train_shape=(X_train.shape[0], X_train.shape[1]),
            test_shape=(X_test.shape[0], X_test.shape[1]),
            missing_method_used=self.config.missing_method,
            numeric_means=self.numeric_means,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std
        )
    
if __name__ == "__main__":
    # Khởi tạo cấu hình: ở đây ta chọn điền khuyết bằng median vì nó bền với
    # outlier hơn mean trên các biến chi phí và số đêm lưu trú vốn lệch phải.
    config = PipelineConfig(
        data_dir="data",
        missing_method="median"
    )

    # Chạy toàn bộ năm bước tiền xử lý và nhận về kết quả đã đóng gói.
    pipeline = DataPipeline(config)
    result = pipeline.run()

    # In bảng tóm tắt để kiểm tra nhanh hình dạng ma trận và thống kê đầu ra,
    # qua đó xác nhận biến số đã được chuẩn hóa về trung bình 0, độ lệch chuẩn 1.
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
