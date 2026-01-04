"""
Unit tests for categorical column encoding utilities.

Tests cover:
- Categorical column detection
- Cardinality validation and warnings
- LabelEncoder creation and application
- Error handling for unseen categories
- Encoding info and mapping extraction
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.utils.model_utils import (
    detect_categorical_columns,
    validate_cardinality,
    create_categorical_encoders,
    apply_encoders,
    get_encoding_mappings,
    get_categorical_encoding_info,
    CardinalityWarning,
    CategoricalEncodingInfo,
    EncodingError,
)


class TestDetectCategoricalColumns:
    """Test categorical column detection."""
    
    def test_detect_categorical_columns_basic(self):
        """Test detection of categorical columns in basic DataFrame."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "NYC"],
        })
        result = detect_categorical_columns(df)
        assert result == ["city", "name"]  # Sorted
    
    def test_detect_categorical_columns_all_numeric(self):
        """Test with DataFrame containing only numeric columns."""
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "score": [100, 95, 85],
        })
        result = detect_categorical_columns(df)
        assert result == []
    
    def test_detect_categorical_columns_all_categorical(self):
        """Test with DataFrame containing only categorical columns."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "city": ["NYC", "LA"],
        })
        result = detect_categorical_columns(df)
        assert result == ["city", "name"]  # Sorted
    
    def test_detect_categorical_columns_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = detect_categorical_columns(df)
        assert result == []
    
    def test_detect_categorical_columns_mixed_types(self):
        """Test with mixed numeric types and categorical."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        })
        result = detect_categorical_columns(df)
        assert result == ["str_col"]


class TestValidateCardinality:
    """Test cardinality validation and warnings."""
    
    def test_validate_cardinality_below_threshold(self):
        """Test columns with cardinality below threshold."""
        df = pd.DataFrame({
            "color": ["red", "blue", "green"] * 100,
        })
        warnings = validate_cardinality(df, ["color"], threshold=1000)
        assert len(warnings) == 0
    
    def test_validate_cardinality_above_threshold(self):
        """Test columns with cardinality above threshold."""
        df = pd.DataFrame({
            "id": range(1100),
        })
        warnings = validate_cardinality(df, ["id"], threshold=1000)
        assert "id" in warnings
        assert isinstance(warnings["id"], CardinalityWarning)
        assert warnings["id"].cardinality == 1100
    
    def test_validate_cardinality_multiple_columns(self):
        """Test multiple columns with mixed cardinalities."""
        # Create columns with same length (1100 rows)
        df = pd.DataFrame({
            "color": ["red", "blue"] * 550,  # 1100 items
            "id": list(range(1100)),  # 1100 unique values (>1000 threshold)
            "category": ["A", "B", "C"] * 366 + ["A", "B"],  # 1100 items, only 3 unique
        })
        warnings = validate_cardinality(
            df,
            ["color", "id", "category"],
            threshold=1000,
        )
        assert len(warnings) == 1
        assert "id" in warnings
    
    def test_cardinality_warning_to_dict(self):
        """Test CardinalityWarning serialization."""
        warning = CardinalityWarning(
            column="color",
            cardinality=1500,
            threshold=1000,
        )
        result = warning.to_dict()
        assert result["column"] == "color"
        assert result["cardinality"] == 1500
        assert result["threshold"] == 1000


class TestCreateCategoricalEncoders:
    """Test LabelEncoder creation and fitting."""
    
    def test_create_categorical_encoders_basic(self):
        """Test basic encoder creation."""
        df = pd.DataFrame({
            "color": ["red", "blue", "red"],
        })
        encoders = create_categorical_encoders(df, ["color"])
        assert "color" in encoders
        assert isinstance(encoders["color"], LabelEncoder)
        # Check that encoder learned the classes
        assert set(encoders["color"].classes_) == {"red", "blue"}
    
    def test_create_categorical_encoders_multiple_columns(self):
        """Test encoder creation for multiple columns."""
        df = pd.DataFrame({
            "color": ["red", "blue", "red"],
            "size": ["S", "M", "L"],
        })
        encoders = create_categorical_encoders(df, ["color", "size"])
        assert len(encoders) == 2
        assert "color" in encoders
        assert "size" in encoders
    
    def test_create_categorical_encoders_missing_column(self):
        """Test error when column not in DataFrame."""
        df = pd.DataFrame({
            "color": ["red", "blue"],
        })
        with pytest.raises(EncodingError) as exc_info:
            create_categorical_encoders(df, ["missing_col"])
        assert "missing_col" in str(exc_info.value)
    
    def test_create_categorical_encoders_fit_only_on_train(self):
        """Test that encoders learn only from provided data."""
        df_train = pd.DataFrame({
            "color": ["red", "blue"],
        })
        df_test = pd.DataFrame({
            "color": ["red", "green"],  # green is not in training
        })
        encoders = create_categorical_encoders(df_train, ["color"])
        # Encoder should only know about red and blue
        assert set(encoders["color"].classes_) == {"red", "blue"}
        # Test data's "green" is not in encoder's classes
        assert "green" not in encoders["color"].classes_


class TestApplyEncoders:
    """Test encoder application and unseen value detection."""
    
    def test_apply_encoders_basic(self):
        """Test basic encoder application."""
        df_train = pd.DataFrame({
            "color": ["red", "blue"],
        })
        df_test = pd.DataFrame({
            "color": ["red", "blue"],
        })
        encoders = create_categorical_encoders(df_train, ["color"])
        result = apply_encoders(encoders, df_test)
        
        # Check that values are encoded
        assert result["color"].dtype in [np.int64, np.int32]
        assert list(result["color"]) == [1, 0] or list(result["color"]) == [0, 1]
    
    def test_apply_encoders_preserves_numeric_columns(self):
        """Test that non-encoded columns are preserved."""
        df_train = pd.DataFrame({
            "color": ["red", "blue"],
            "age": [25, 30],
        })
        df_test = pd.DataFrame({
            "color": ["red", "blue"],
            "age": [25, 30],
        })
        encoders = create_categorical_encoders(df_train, ["color"])
        result = apply_encoders(encoders, df_test)
        
        # Age should remain unchanged
        assert list(result["age"]) == [25, 30]
    
    def test_apply_encoders_unseen_category_error(self):
        """Test that unseen categories raise error."""
        df_train = pd.DataFrame({
            "color": ["red", "blue"],
        })
        df_test = pd.DataFrame({
            "color": ["red", "green"],
        })
        encoders = create_categorical_encoders(df_train, ["color"])
        
        with pytest.raises(EncodingError) as exc_info:
            apply_encoders(encoders, df_test, raise_on_unknown=True)
        
        error = exc_info.value
        assert error.column == "color"
        assert "green" in str(error.unseen_values)
    
    def test_apply_encoders_returns_copy(self):
        """Test that apply_encoders returns a copy, not a view."""
        df_train = pd.DataFrame({
            "color": ["red", "blue"],
        })
        df_test = pd.DataFrame({
            "color": ["red", "blue"],
        })
        encoders = create_categorical_encoders(df_train, ["color"])
        result = apply_encoders(encoders, df_test)
        
        # Modifying result should not affect df_test
        result.loc[0, "color"] = 999
        assert df_test.loc[0, "color"] != 999


class TestGetEncodingMappings:
    """Test encoding mapping extraction."""
    
    def test_get_encoding_mappings_basic(self):
        """Test basic mapping extraction."""
        df = pd.DataFrame({
            "color": ["red", "blue", "red"],
        })
        encoders = create_categorical_encoders(df, ["color"])
        mappings = get_encoding_mappings(encoders)
        
        assert "color" in mappings
        assert isinstance(mappings["color"], dict)
        # Check that all original values are in mapping
        assert set(mappings["color"].keys()) == {"red", "blue"}
    
    def test_get_encoding_mappings_values_are_integers(self):
        """Test that mapped values are integers."""
        df = pd.DataFrame({
            "color": ["red", "blue"],
        })
        encoders = create_categorical_encoders(df, ["color"])
        mappings = get_encoding_mappings(encoders)
        
        for original, encoded in mappings["color"].items():
            assert isinstance(encoded, int)


class TestCategoricalEncodingInfo:
    """Test CategoricalEncodingInfo dataclass."""
    
    def test_categorical_encoding_info_creation(self):
        """Test creating encoding info."""
        df = pd.DataFrame({
            "color": ["red", "blue", "red"],
        })
        encoders = create_categorical_encoders(df, ["color"])
        info = get_categorical_encoding_info(encoders)
        
        assert "color" in info
        encoding_info = info["color"]
        assert isinstance(encoding_info, CategoricalEncodingInfo)
        assert encoding_info.column_name == "color"
        assert encoding_info.cardinality == 2
    
    def test_categorical_encoding_info_to_dict(self):
        """Test encoding info serialization."""
        df = pd.DataFrame({
            "color": ["red", "blue"],
        })
        encoders = create_categorical_encoders(df, ["color"])
        info = get_categorical_encoding_info(encoders)
        
        info_dict = info["color"].to_dict()
        assert "column_name" in info_dict
        assert "cardinality" in info_dict
        assert "mapping" in info_dict
        assert info_dict["cardinality"] == 2


class TestEncodingError:
    """Test EncodingError exception."""
    
    def test_encoding_error_basic(self):
        """Test basic error creation."""
        error = EncodingError("Test error")
        assert "Test error" in str(error)
    
    def test_encoding_error_with_details(self):
        """Test error with column and unseen values."""
        error = EncodingError(
            message="Unseen values detected",
            column="color",
            unseen_values=["green"],
            original_classes=["red", "blue"],
        )
        error_str = str(error)
        assert "color" in error_str
        assert "green" in error_str
        assert "red" in error_str or "blue" in error_str
    
    def test_encoding_error_to_dict(self):
        """Test error serialization."""
        error = EncodingError(
            message="Test",
            column="color",
            unseen_values=["green"],
        )
        error_dict = error.to_dict()
        assert error_dict["column"] == "color"
        assert error_dict["unseen_values"] == ["green"]


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_single_unique_value_column(self):
        """Test column with single unique value."""
        df = pd.DataFrame({
            "constant": ["A", "A", "A"],
            "varied": [1, 2, 3],
        })
        encoders = create_categorical_encoders(df, ["constant"])
        result = apply_encoders(encoders, df)
        
        # Single unique value should be encoded as 0
        assert all(result["constant"] == 0)
    
    def test_all_categorical_dataframe(self):
        """Test DataFrame with all categorical columns."""
        df = pd.DataFrame({
            "col1": ["A", "B"],
            "col2": ["X", "Y"],
            "col3": ["red", "blue"],
        })
        cat_cols = detect_categorical_columns(df)
        assert len(cat_cols) == 3
        
        encoders = create_categorical_encoders(df, cat_cols)
        assert len(encoders) == 3
    
    def test_all_numeric_dataframe(self):
        """Test DataFrame with all numeric columns."""
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": [3.5, 4.5],
        })
        cat_cols = detect_categorical_columns(df)
        assert len(cat_cols) == 0
    
    def test_missing_values_in_categorical_column(self):
        """Test categorical column with NaN values."""
        df = pd.DataFrame({
            "color": ["red", None, "blue"],
        })
        cat_cols = detect_categorical_columns(df)
        assert "color" in cat_cols  # NaN doesn't change dtype to object
        
        # Should be able to create encoders despite NaN
        encoders = create_categorical_encoders(df, ["color"])
        assert "color" in encoders
    
    def test_numeric_strings_as_categorical(self):
        """Test numeric values stored as strings."""
        df = pd.DataFrame({
            "id": ["001", "002", "003"],
        })
        cat_cols = detect_categorical_columns(df)
        assert "id" in cat_cols  # Detected as categorical due to object dtype
        
        encoders = create_categorical_encoders(df, ["id"])
        result = apply_encoders(encoders, df)
        # Should be encoded despite being numeric strings
        assert result["id"].dtype in [np.int64, np.int32]
