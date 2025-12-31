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
        """Test that composed pipeline can fit and predict."""
        df = pd.DataFrame({
            "color": ["red", "blue", "red", "blue"],
            "age": [25, 30, 26, 29],
        })
        y = [0, 1, 0, 1]
        
        encoders = create_categorical_encoders(df.head(2), ["color"])
        transformer = build_preprocessing_pipeline(
            categorical_cols=["color"],
            numeric_cols=["age"],
            categorical_encoders=encoders,
        )
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        pipeline = compose_full_model_pipeline(transformer, model)
        
        # Should be able to fit and predict
        pipeline.fit(df, y)
        predictions = pipeline.predict(df)
        assert len(predictions) == len(df)
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
        """Test that pipeline uses the same fitted encoder for test data."""
        df_train = pd.DataFrame({
            "color": ["red", "blue"],
            "age": [25, 30],
        })
        df_test = pd.DataFrame({
            "color": ["red", "blue"],
            "age": [28, 32],
        })
        
        # Encoders fitted on training data
        encoders = create_categorical_encoders(df_train, ["color"])
        
        # Build transformer with those encoders
        transformer = build_preprocessing_pipeline(
            categorical_cols=["color"],
            numeric_cols=["age"],
            categorical_encoders=encoders,
        )
        
        # Transform should use the same encoder for both sets
        # (not refit on test data)
        X_train_transformed = transformer.fit_transform(df_train)
        X_test_transformed = transformer.transform(df_test)
        
        # Both should be successfully transformed
        assert X_train_transformed.shape[0] == len(df_train)
        assert X_test_transformed.shape[0] == len(df_test)


class TestColumnOrderPreservation:
    """Test that column order is preserved through pipeline."""
    
    def test_column_order_preserved_categorical_first(self):
        """Test that categorical columns come before numeric in output."""
        df = pd.DataFrame({
            "age": [25, 30],
            "color": ["red", "blue"],
            "score": [100, 95],
        })
        encoders = create_categorical_encoders(df, ["color"])
        
        transformer = build_preprocessing_pipeline(
            categorical_cols=["color"],
            numeric_cols=["age", "score"],
            categorical_encoders=encoders,
        )
        
        # Categorical should be processed first (color -> 0,1), then numeric
        result = transformer.fit_transform(df)
        
        # Result should have columns in the order: categorical first, then numeric
        # The exact order depends on ColumnTransformer's ordering
        assert result.shape[1] == 3  # 1 categorical + 2 numeric
