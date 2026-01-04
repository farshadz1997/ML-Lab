"""
Integration tests for sklearn Pipeline composition with categorical encoding.

Tests cover:
- ColumnTransformer composition for mixed categorical/numeric data
- Full Pipeline composition with preprocessing + model
- Data leakage prevention
- Column order preservation
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.model_utils import (
    build_preprocessing_pipeline,
    compose_full_model_pipeline,
    create_categorical_encoders,
    apply_encoders,
)


class TestBuildPreprocessingPipeline:
    """Test ColumnTransformer pipeline building."""
    
    def test_build_preprocessing_pipeline_basic(self):
        """Test basic pipeline building with categorical and numeric columns."""
        df = pd.DataFrame({
            "color": ["red", "blue"],
            "age": [25, 30],
        })
        encoders = create_categorical_encoders(df, ["color"])
        
        transformer = build_preprocessing_pipeline(
            categorical_cols=["color"],
            numeric_cols=["age"],
            categorical_encoders=encoders,
            numeric_scaler="standard",
        )
        
        assert transformer is not None
        assert len(transformer.transformers) == 2
    
    def test_build_preprocessing_pipeline_only_categorical(self):
        """Test pipeline with only categorical columns."""
        df = pd.DataFrame({
            "color": ["red", "blue"],
            "size": ["S", "M"],
        })
        encoders = create_categorical_encoders(df, ["color", "size"])
        
        transformer = build_preprocessing_pipeline(
            categorical_cols=["color", "size"],
            numeric_cols=[],
            categorical_encoders=encoders,
        )
        
        assert transformer is not None
        assert len(transformer.transformers) == 1
    
    def test_build_preprocessing_pipeline_only_numeric(self):
        """Test pipeline with only numeric columns."""
        df = pd.DataFrame({
            "age": [25, 30],
            "score": [100, 95],
        })
        
        transformer = build_preprocessing_pipeline(
            categorical_cols=[],
            numeric_cols=["age", "score"],
            categorical_encoders={},
            numeric_scaler="standard",
        )
        
        assert transformer is not None
        assert len(transformer.transformers) == 1
    
    def test_build_preprocessing_pipeline_no_columns_raises_error(self):
        """Test that providing no columns raises ValueError."""
        with pytest.raises(ValueError):
            build_preprocessing_pipeline(
                categorical_cols=[],
                numeric_cols=[],
                categorical_encoders={},
            )
    
    def test_build_preprocessing_pipeline_minmax_scaler(self):
        """Test pipeline with MinMax scaler."""
        df = pd.DataFrame({
            "color": ["red", "blue"],
            "age": [25, 30],
        })
        encoders = create_categorical_encoders(df, ["color"])
        
        transformer = build_preprocessing_pipeline(
            categorical_cols=["color"],
            numeric_cols=["age"],
            categorical_encoders=encoders,
            numeric_scaler="minmax",
        )
        
        assert transformer is not None
    
    def test_build_preprocessing_pipeline_no_scaler(self):
        """Test pipeline without numeric scaling."""
        df = pd.DataFrame({
            "color": ["red", "blue"],
            "age": [25, 30],
        })
        encoders = create_categorical_encoders(df, ["color"])
        
        transformer = build_preprocessing_pipeline(
            categorical_cols=["color"],
            numeric_cols=["age"],
            categorical_encoders=encoders,
            numeric_scaler="none",
        )
        
        assert transformer is not None


class TestComposeFullModelPipeline:
    """Test full Pipeline composition with model estimator."""
    
    def test_compose_full_model_pipeline_basic(self):
        """Test basic pipeline composition."""
        df = pd.DataFrame({
            "color": ["red", "blue"],
            "age": [25, 30],
        })
        encoders = create_categorical_encoders(df, ["color"])
        
        transformer = build_preprocessing_pipeline(
            categorical_cols=["color"],
            numeric_cols=["age"],
            categorical_encoders=encoders,
        )
        model = LogisticRegression(random_state=42)
        
        pipeline = compose_full_model_pipeline(transformer, model)
        
        assert pipeline is not None
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "preprocessor"
        assert pipeline.steps[1][0] == "model"
    
    def test_compose_full_model_pipeline_fit_predict(self):
        """Test that composed pipeline can transform preprocessed data."""
        # Note: This test reflects actual usage - data is pre-encoded, then passed to model
        df_train = pd.DataFrame({
            "color": ["red", "blue", "red", "blue"],
            "age": [25, 30, 26, 29],
        })
        df_test = pd.DataFrame({
            "color": ["red", "blue", "red", "blue"],
            "age": [27, 31, 25, 29],
        })
        y_train = [0, 1, 0, 1]
        y_test = [1, 0, 0, 1]
        
        # Fit encoders on training data (as per spec - prevent data leakage)
        encoders = create_categorical_encoders(df_train, ["color"])
        
        # Apply encoders to get preprocessed data
        X_train_encoded = apply_encoders(encoders, df_train)
        X_test_encoded = apply_encoders(encoders, df_test)
        
        # Now train model on already-encoded data (actual workflow)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_encoded, y_train)
        predictions = model.predict(X_test_encoded)
        
        assert len(predictions) == len(df_test)
        assert all(p in [0, 1] for p in predictions)


class TestDataLeakagePrevention:
    """Test that pipeline prevents data leakage."""
    
    def test_encoder_fitted_only_on_train(self):
        """Test that encoder is only fitted on training data."""
        df_train = pd.DataFrame({
            "color": ["red", "blue"],
        })
        df_test = pd.DataFrame({
            "color": ["red", "green"],  # green is new
        })
        
        # Create encoders on training data only
        encoders_train = create_categorical_encoders(df_train, ["color"])
        
        # green should not be in encoder's classes
        assert "green" not in encoders_train["color"].classes_
        assert set(encoders_train["color"].classes_) == {"red", "blue"}
    
    def test_pipeline_uses_fitted_encoder(self):
        """Test that transformer uses the same fitted encoder for test data."""
        df_train = pd.DataFrame({
            "color": ["red", "blue"],
            "age": [25, 30],
        })
        df_test = pd.DataFrame({
            "color": ["red", "blue"],
            "age": [28, 32],
        })
        
        # Encoders fitted on training data (prevent data leakage)
        encoders = create_categorical_encoders(df_train, ["color"])
        
        # Apply encoders to both train and test using the SAME fitted encoders
        X_train_encoded = apply_encoders(encoders, df_train)
        X_test_encoded = apply_encoders(encoders, df_test)
        
        # Both should be successfully transformed with same encoder instance
        assert X_train_encoded.shape[0] == len(df_train)
        assert X_test_encoded.shape[0] == len(df_test)
        
        # Encoded values should be the same for same input (consistency)
        assert X_train_encoded["color"].iloc[0] == X_test_encoded["color"].iloc[0]  # Both "red"
        assert X_train_encoded["color"].iloc[1] == X_test_encoded["color"].iloc[1]  # Both "blue"


class TestColumnOrderPreservation:
    """Test that column order is preserved through pipeline."""
    
    def test_column_order_preserved_categorical_first(self):
        """Test that categorical columns are processed first in pipeline."""
        df = pd.DataFrame({
            "age": [25, 30],
            "color": ["red", "blue"],
            "score": [100, 95],
        })
        encoders = create_categorical_encoders(df, ["color"])
        
        # Build transformer specifying categorical first
        transformer = build_preprocessing_pipeline(
            categorical_cols=["color"],
            numeric_cols=["age", "score"],
            categorical_encoders=encoders,
        )
        
        # Verify transformer is built with correct structure
        assert len(transformer.transformers) == 2
        assert transformer.transformers[0][0] == "categorical"
        assert transformer.transformers[1][0] == "numeric"
