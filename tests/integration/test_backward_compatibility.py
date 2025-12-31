"""
Backward compatibility tests for numeric-only datasets.

Tests verify that existing functionality (models trained without categorical columns)
continues to work unchanged after adding categorical encoding support.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from src.core.data_preparation import prepare_data_for_training
from src.utils.model_utils import (
    detect_categorical_columns,
    create_categorical_encoders,
)


class TestNumericOnlyData:
    """Test that numeric-only datasets work as before."""
    
    def test_numeric_only_no_categorical_columns_detected(self):
        """Test that numeric-only DataFrames don't trigger categorical encoding."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "income": [30000, 50000, 60000, 70000],
            "score": [100, 95, 85, 90],
        })
        
        cat_cols = detect_categorical_columns(df)
        assert len(cat_cols) == 0
        
        # No encoders should be created
        assert len(create_categorical_encoders(df, cat_cols)) == 0
    
    def test_numeric_only_data_preparation(self):
        """Test prepare_data_for_training with numeric-only data."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45, 50],
            "income": [30000, 50000, 60000, 70000, 80000, 90000],
            "target": [0, 1, 0, 1, 0, 1],
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.33, random_state=42)
        )
        
        # No categorical columns
        assert len(cat_cols) == 0
        assert len(encoders) == 0
        
        # All columns are numeric
        assert set(num_cols) == {"age", "income"}
        
        # Data should be unchanged (no encoding applied)
        assert X_train.shape[1] == 2
        assert X_test.shape[1] == 2
        assert X_train["age"].dtype in [np.int64, np.int32, np.float64, np.float32]
    
    def test_numeric_only_logistic_regression(self):
        """Test LogisticRegression still works with numeric-only data."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "target": [0, 1, 0, 1, 0, 1],
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.33, random_state=42)
        )
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Should predict successfully
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
    
    def test_numeric_only_random_forest(self):
        """Test RandomForest still works with numeric-only data."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "target": [0, 1, 0, 1, 0, 1],
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.33, random_state=42)
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Should predict successfully
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
    
    def test_numeric_only_linear_regression(self):
        """Test LinearRegression still works with numeric-only data."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "target": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.33, random_state=42)
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Should predict successfully
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)


class TestDataIntegrity:
    """Test that data integrity is preserved in numeric-only case."""
    
    def test_numeric_values_preserved(self):
        """Test that numeric values are not modified."""
        df = pd.DataFrame({
            "age": [25.5, 30.7, 35.2, 40.9],
            "salary": [50000, 60000, 70000, 80000],
            "target": [0, 1, 0, 1],
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.5, random_state=42)
        )
        
        # Values should match original (just rearranged by train-test split)
        all_values = pd.concat([X_train, X_test]).sort_index()
        original_features = df.drop(columns=["target"]).sort_index()
        
        # Check values are preserved (may be reordered, so just check counts)
        assert set(X_train["age"].tolist() + X_test["age"].tolist()).issubset(
            set(df["age"].tolist())
        )
    
    def test_no_unnecessary_dtype_changes(self):
        """Test that numeric dtypes are preserved."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3, 4],
            "float_col": [1.5, 2.5, 3.5, 4.5],
            "target": [0, 1, 0, 1],
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.5, random_state=42)
        )
        
        # Dtypes should be numeric (not converted to string/categorical)
        assert X_train["int_col"].dtype in [np.int64, np.int32]
        assert X_train["float_col"].dtype in [np.float64, np.float32]


class TestMixedDataBackwardCompatibility:
    """Test that previous behavior on mixed data is preserved."""
    
    def test_mixed_data_with_categorical_target(self):
        """Test classification with only target categorical (not features)."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "target": ["A", "B", "A", "B", "A", "B"],  # Categorical target
        })
        
        # Should work - target is separate from features
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.33, random_state=42)
        )
        
        # Features should be numeric
        assert len(cat_cols) == 0
        assert "feature1" in num_cols
        assert "feature2" in num_cols


class TestLargeNumericDatasets:
    """Test with larger numeric-only datasets."""
    
    def test_large_numeric_dataset(self):
        """Test preparation of larger numeric dataset."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(1000),
            "feature2": np.random.randn(1000),
            "feature3": np.random.randn(1000),
            "target": np.random.randint(0, 2, 1000),
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.2, random_state=42)
        )
        
        # Should handle large dataset efficiently
        assert len(X_train) == 800
        assert len(X_test) == 200
        assert len(cat_cols) == 0
        assert len(encoders) == 0
    
    def test_many_numeric_features(self):
        """Test with many numeric features."""
        n_features = 50
        n_samples = 100
        
        df = pd.DataFrame({
            f"feature_{i}": np.random.randn(n_samples)
            for i in range(n_features)
        })
        df["target"] = np.random.randint(0, 2, n_samples)
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.2, random_state=42)
        )
        
        # All features should be numeric
        assert len(cat_cols) == 0
        assert len(num_cols) == n_features
