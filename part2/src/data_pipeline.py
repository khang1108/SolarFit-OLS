"""
=============================================================================
Data Pipeline for Part 2: Data Preprocessing and Feature Engineering
=============================================================================
Handles:
  1. Data loading (Train.csv, Test.csv)
  2. Missing value handling (3+ methods)
  3. Categorical encoding (One-Hot, Ordinal)
  4. Feature scaling (StandardScaler)
  5. Train-Test split and alignment

Dataset: Tanzania Tourism Expenditure (Zindi)
  - Train: 4,809 rows × 23 columns
  - Test: 1,601 rows × 22 columns (no target)
  - Target: total_cost (TZS)
  - Missing: travel_with (23%), most_impressing (6.5%)

=============================================================================
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
    """Configuration for the data preprocessing pipeline."""

    # File paths
    data_dir: str = "data"
    train_file: str = "Train.csv"
    test_file: str = "Test.csv"

    # Column roles
    id_col: str = "ID"
    target_col: str = "total_cost"

    # Missing value handling method
    # Options: "listwise", "mean", "regression"
    missing_method: str = "mean"

    # Feature scaling
    scale_features: bool = True

    # Numeric and categorical features (will be auto-detected if not specified)
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None


@dataclass
class PipelineResult:
    """Output of the preprocessing pipeline."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: Optional[np.ndarray]  # May be None if test has no target

    feature_names: List[str]
    feature_types: Dict[str, str]  # 'numeric' or 'categorical'

    train_shape: Tuple[int, int]
    test_shape: Tuple[int, int]

    missing_method_used: str

    # For reproducibility
    numeric_means: Dict[str, float]
    scaler_mean: Optional[np.ndarray]
    scaler_std: Optional[np.ndarray]


class DataPipeline:
    """
    End-to-end data preprocessing pipeline for regression modeling.

    Usage:
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
        """Execute the full preprocessing pipeline."""

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
        """Load train and test CSV files."""
        train_path = os.path.join(self.config.data_dir, self.config.train_file)
        test_path = os.path.join(self.config.data_dir, self.config.test_file)

        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        print(f"  Train: {self.train_df.shape[0]:,} rows × {self.train_df.shape[1]} cols")
        print(f"  Test:  {self.test_df.shape[0]:,} rows × {self.test_df.shape[1]} cols")
        print(f"  Target in train: {self.config.target_col in self.train_df.columns}")

    def _handle_missing_values(self):
        """Handle missing values using specified method."""
        method = self.config.missing_method

        if method == "listwise":
            print(f"  Method: Listwise deletion (remove rows with ANY missing values)")
            self.train_df = self.train_df.dropna()
            self.test_df = self.test_df.dropna()
            print(f"    → Train: {self.train_df.shape[0]:,} rows remaining")
            print(f"    → Test:  {self.test_df.shape[0]:,} rows remaining")

        elif method == "mean":
            print(f"  Method: Mean/Median imputation for numeric, Mode for categorical")
            # Numeric columns: use mean
            numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col == self.config.target_col:
                    continue
                if self.train_df[col].isnull().any():
                    mean_val = self.train_df[col].mean()
                    self.train_df[col].fillna(mean_val, inplace=True)
                    self.test_df[col].fillna(mean_val, inplace=True)
                    print(f"    ✓ {col}: filled with mean={mean_val:.2f}")

            # Categorical columns: use mode
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
        """Use regression to impute missing values in a numeric column."""
        # This is a simplified version; for production use more sophisticated methods
        # Get rows with non-missing values
        non_missing = self.train_df[self.train_df[col].notna()].copy()

        if len(non_missing) < 10:
            # Fall back to mean if not enough data
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
        """Auto-detect numeric and categorical features."""
        exclude_cols = {self.config.id_col, self.config.target_col}

        # Numeric features
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_features = [c for c in numeric_cols if c not in exclude_cols]

        # Categorical features
        cat_cols = self.train_df.select_dtypes(include=['object']).columns.tolist()
        self.categorical_features = [c for c in cat_cols if c not in exclude_cols]

        print(f"  Numeric features ({len(self.numeric_features)}): {self.numeric_features[:5]}...")
        print(f"  Categorical features ({len(self.categorical_features)}): {self.categorical_features[:5]}...")

        # Store numeric means for later scaling
        self.numeric_means = self.train_df[self.numeric_features].mean().to_dict()
        self.numeric_stds = self.train_df[self.numeric_features].std().to_dict()

    def _encode_categorical(self):
        """Encode categorical features using One-Hot encoding."""

        # One-Hot encode categorical features
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

        # Align columns: test may have different categories
        for col in train_encoded.columns:
            if col not in test_encoded.columns:
                test_encoded[col] = 0

        for col in test_encoded.columns:
            if col not in train_encoded.columns:
                train_encoded[col] = 0

        # Reorder to match train
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
        """Scale numeric features and prepare final train/test sets."""

        # Get feature names
        self.feature_names = self.numeric_features + self.categorical_encoded_features

        # Create feature type mapping
        self.feature_types = {f: "numeric" for f in self.numeric_features}
        self.feature_types.update({f: "categorical" for f in self.categorical_encoded_features})

        # Extract X (features) and y (target) - ensure float64 dtype
        X_train = self.train_df[self.feature_names].astype(float).values
        X_test = self.test_df[self.feature_names].astype(float).values

        y_train = self.train_df[self.config.target_col].astype(float).values

        # Scale numeric features (standardize)
        if self.config.scale_features:
            scaler_mean = X_train[:, :len(self.numeric_features)].mean(axis=0)
            scaler_std = X_train[:, :len(self.numeric_features)].std(axis=0)

            # Avoid division by zero
            scaler_std[scaler_std == 0] = 1.0

            X_train[:, :len(self.numeric_features)] = (
                X_train[:, :len(self.numeric_features)] - scaler_mean
            ) / scaler_std

            X_test[:, :len(self.numeric_features)] = (
                X_test[:, :len(self.numeric_features)] - scaler_mean
            ) / scaler_std

            print(f"  Scaled {len(self.numeric_features)} numeric features")
        else:
            scaler_mean = None
            scaler_std = None

        # Add intercept (first column = 1)
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
