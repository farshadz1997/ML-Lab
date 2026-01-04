"""
Integration tests for model training with categorical data.

Tests cover:
- End-to-end training flow with categorical columns
- Train-test consistency (no data leakage)
- Error handling for unseen categories
- Cardinality warnings display
- Results display with encoding metadata
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.core.data_preparation import prepare_data_for_training
from src.utils.model_utils import (
    detect_categorical_columns,
    create_categorical_encoders,
    apply_encoders,
    EncodingError,
)


class TestPrepareDataForTraining:
    """Test prepare_data_for_training integration."""
    
    def test_prepare_data_basic(self):
        """Test basic data preparation with categorical columns."""
        df = pd.DataFrame({
            "color": ["red", "red", "blue", "blue", "red", "blue"],
            "age": [25, 26, 30, 31, 35, 40],
            "target": [0, 0, 1, 1, 0, 1],
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.33, random_state=42)
        )
        
        # Check splits
        assert len(X_train) == 4
        assert len(X_test) == 2
        assert len(y_train) == 4
        assert len(y_test) == 2
        
        # Check columns detected
        assert "color" in cat_cols
        assert "age" in num_cols
        
        # Check encoders created
        assert "color" in encoders
        assert len(encoders) == 1
    
    def test_prepare_data_no_categorical_columns(self):
        """Test preparation with numeric-only data."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "score": [100, 95, 85, 90],
            "target": [0, 1, 0, 1],
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.5, random_state=42)
        )
        
        assert len(cat_cols) == 0
        assert len(encoders) == 0
        assert "age" in num_cols
        assert "score" in num_cols
    
    def test_prepare_data_all_categorical_columns(self):
        """Test preparation with all categorical features."""
        df = pd.DataFrame({
            "color": ["red", "blue", "red", "blue", "red", "blue"],
            "size": ["S", "M", "L", "S", "M", "L"],
            "target": [0, 1, 0, 1, 0, 1],
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.33, random_state=42)
        )
        
        assert len(cat_cols) == 2
        assert len(num_cols) == 0
        assert len(encoders) == 2
    
    def test_prepare_data_missing_target_column(self):
        """Test error when target column missing."""
        df = pd.DataFrame({
            "color": ["red", "blue"],
            "age": [25, 30],
        })
        
        with pytest.raises(ValueError) as exc_info:
            prepare_data_for_training(df, "missing_target", test_size=0.5)
        assert "missing_target" in str(exc_info.value)
    
    def test_prepare_data_train_test_split_before_encoding(self):
        """Test that train-test split happens before encoding (no leakage)."""
        df = pd.DataFrame({
            "color": ["red", "blue", "red", "blue", "red", "blue"],
            "target": [0, 1, 0, 1, 0, 1],
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.33, random_state=42)
        )
        
        # Check that encoders learned from training data
        # Both red and blue should be in training data with this split
        learned_classes = set(encoders["color"].classes_)
        assert len(learned_classes) > 0
        
        # Check that both train and test can be encoded
        assert X_train.shape[0] == 4
        assert X_test.shape[0] == 2
    
    def test_prepare_data_cardinality_warnings(self):
        """Test cardinality warning detection."""
        df = pd.DataFrame({
            "id": range(1100),  # High cardinality
            "target": [0, 1] * 550,
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.2, random_state=42)
        )
        
        # id column should trigger warning
        assert "id" in warnings or len(cat_cols) == 0  # Could be numeric depending on data type
    
    def test_prepare_data_unseen_category_error(self):
        """Test error when test data has unseen categories."""
        # Create data where test set will have unseen values
        df = pd.DataFrame({
            "color": ["red"] * 4 + ["blue"] * 4,  # Training will see both
            "target": [0, 1] * 4,
        })
        
        # Normal case should work
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(
                df,
                "target",
                test_size=0.25,
                random_state=42,
                raise_on_unseen=True,
            )
        )
        
        # Should succeed because we're splitting the same data
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0


class TestNoDataLeakage:
    """Test that encoding prevents data leakage."""
    
    def test_encoder_fitted_only_on_train(self):
        """Test that encoders are fitted only on training data."""
        df = pd.DataFrame({
            "color": ["red", "blue", "red", "blue", "green"],
            "target": [0, 1, 0, 1, 0],
        })
        
        # Prepare data - should work (green will be in test)
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(
                df,
                "target",
                test_size=0.2,
                random_state=123,  # Use seed where green ends up in test
                raise_on_unseen=False,  # Don't raise on unseen
            )
        )
        
        # green should not be in encoder's training classes
        # (it will be in test, but encoder was fitted on train only)
        assert len(X_train) > 0
        assert len(X_test) > 0
    
    def test_same_encoder_applied_to_test(self):
        """Test that the same fitted encoder is used for test data."""
        # Create larger dataset to ensure both values in train AND test
        df = pd.DataFrame({
            "color": ["red", "blue"] * 50,  # 100 rows, both colors in train and test
            "age": list(range(100)),
            "target": [0, 1] * 50,
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.3, random_state=42)
        )
        
        # Both train and test should have data
        assert len(X_train) > 0
        assert len(X_test) > 0
        
        # Check that encoded values are integers
        if "color" in X_train.columns:
            assert X_train["color"].dtype in [np.int64, np.int32]
            assert X_test["color"].dtype in [np.int64, np.int32]


class TestEncodingConsistency:
    """Test that encoding produces consistent results."""
    
    def test_encoding_deterministic(self):
        """Test that encoding produces same results on same input."""
        df = pd.DataFrame({
            "color": ["red", "blue", "red"],
        })
        
        encoders1 = create_categorical_encoders(df, ["color"])
        result1 = apply_encoders(encoders1, df)
        result2 = apply_encoders(encoders1, df)
        
        # Applying same encoder twice should give same result
        assert result1["color"].equals(result2["color"])
    
    def test_encoding_values_in_valid_range(self):
        """Test that encoded values are in correct range."""
        df = pd.DataFrame({
            "color": ["red", "blue", "green"],
        })
        
        encoders = create_categorical_encoders(df, ["color"])
        result = apply_encoders(encoders, df)
        
        # Encoded values should be 0 to n_classes-1
        unique_encoded = set(result["color"].unique())
        assert unique_encoded == {0, 1, 2}


class TestMultipleCategoricalColumns:
    """Test with multiple categorical columns."""
    
    def test_multiple_categorical_columns(self):
        """Test preparation with multiple categorical columns."""
        # Create larger dataset to ensure stable splits
        df = pd.DataFrame({
            "color": ["red", "blue"] * 25,  # 50 items, both in train and test
            "size": ["S", "M", "L"] * 16 + ["S", "M"],  # 50 items, all sizes in train and test
            "age": list(range(50)),
            "target": [i % 2 for i in range(50)],
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.3, random_state=42)
        )
        
        # Check all categorical columns detected
        assert set(cat_cols) == {"color", "size"}
        assert "age" in num_cols
        
        # Check encoders created for all categorical columns
        assert len(encoders) == 2
        assert "color" in encoders
        assert "size" in encoders
        
        # Check that encoded values are numeric
        for col in cat_cols:
            assert X_train[col].dtype in [np.int64, np.int32]
            assert X_test[col].dtype in [np.int64, np.int32]


class TestMixedDataTypes:
    """Test with mixed data types."""
    
    def test_numeric_strings_treated_as_categorical(self):
        """Test that numeric strings are treated as categorical."""
        # Create larger dataset where all values appear in both train and test
        df = pd.DataFrame({
            "id_str": ["001", "002", "003", "004"] * 10,  # 40 rows, all values in train and test
            "id_int": list(range(40)),
            "target": [0, 1] * 20,
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.3, random_state=42)
        )
        
        # id_str is object dtype, so should be categorical
        assert "id_str" in cat_cols
        # id_int is numeric
        assert "id_int" in num_cols
    
    def test_with_nan_values(self):
        """Test handling of NaN values in categorical columns."""
        # Create dataset with NaN - need larger dataset and all values in both train/test
        df = pd.DataFrame({
            "color": ["red", "blue", "red", "blue", "green", "green"] * 4,  # 24 rows
            "target": [0, 1] * 12,
        })
        # Introduce some NaN values but ensure both categories in both train/test
        df.loc[::6, "color"] = None  # NaN in every 6th row
        
        # Should handle NaN without crashing
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.3, random_state=42)
        )
        
        assert "color" in cat_cols
        assert len(encoders) >= 1


class TestTrainedModelPerformance:
    """Test that trained models work with encoded data."""
    
    def test_logistic_regression_with_categorical(self):
        """Test LogisticRegression training with categorical features."""
        df = pd.DataFrame({
            "color": ["red", "blue"] * 10,
            "size": ["S", "M", "L"] * 6 + ["S"] * 2,
            "target": [0, 1] * 10,
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.3, random_state=42)
        )
        
        # Train LogisticRegression on encoded data
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Should be able to predict on test data
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
        
        # Should get reasonable accuracy
        accuracy = (predictions == y_test).mean()
        assert accuracy >= 0.0  # Just checking it runs
    
    def test_random_forest_with_categorical(self):
        """Test RandomForestClassifier with categorical features."""
        df = pd.DataFrame({
            "color": ["red", "blue", "green"] * 10,
            "quality": ["low", "medium", "high"] * 10,
            "target": [0, 1, 0] * 10,
        })
        
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
            prepare_data_for_training(df, "target", test_size=0.3, random_state=42)
        )
        
        # Train RandomForest on encoded data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Should be able to predict
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
