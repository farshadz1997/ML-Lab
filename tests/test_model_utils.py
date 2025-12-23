"""
Unit tests for src/utils/model_utils.py

Tests cover:
- Hyperparameter validation (validate_hyperparameters)
- Metrics calculation (classification, regression, clustering)
- Results formatting (markdown generation)
- Feature importance extraction (get_feature_importance)
- Data quality checking (check_data_quality)
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split

from src.utils.model_utils import (
    validate_hyperparameters,
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_clustering_metrics,
    format_results_markdown,
    get_feature_importance,
    check_data_quality,
)


class TestValidateHyperparameters:
    """Test hyperparameter validation."""
    
    def test_valid_hyperparameters(self):
        """Valid hyperparameters should return (True, None)."""
        hyperparams = {'C': 1.0, 'max_iter': 100}
        rules = {
            'C': {'type': float, 'min': 0.001, 'max': 100},
            'max_iter': {'type': int, 'min': 1, 'max': 10000},
        }
        is_valid, error = validate_hyperparameters(hyperparams, 'test', rules)
        assert is_valid is True
        assert error is None
    
    def test_invalid_type(self):
        """Wrong type should return (False, error_msg)."""
        hyperparams = {'C': '1.0'}  # String instead of float
        rules = {'C': {'type': float, 'min': 0.001, 'max': 100}}
        is_valid, error = validate_hyperparameters(hyperparams, 'test', rules)
        assert is_valid is False
        assert 'must be of type' in error
    
    def test_value_too_small(self):
        """Value below minimum should return error."""
        hyperparams = {'C': 0.00001}  # Below min of 0.001
        rules = {'C': {'type': float, 'min': 0.001, 'max': 100}}
        is_valid, error = validate_hyperparameters(hyperparams, 'test', rules)
        assert is_valid is False
        assert '>=' in error
    
    def test_value_too_large(self):
        """Value above maximum should return error."""
        hyperparams = {'C': 1000}  # Above max of 100
        rules = {'C': {'type': float, 'min': 0.001, 'max': 100}}
        is_valid, error = validate_hyperparameters(hyperparams, 'test', rules)
        assert is_valid is False
        assert '<=' in error
    
    def test_no_rules(self):
        """No validation rules should always return (True, None)."""
        hyperparams = {'C': 'anything', 'n_estimators': -999}
        is_valid, error = validate_hyperparameters(hyperparams, 'test', None)
        assert is_valid is True
        assert error is None


class TestCalculateClassificationMetrics:
    """Test classification metrics calculation."""
    
    def test_perfect_predictions(self):
        """Perfect predictions should give accuracy=1.0."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = calculate_classification_metrics(y_true, y_pred)
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_all_wrong_predictions(self):
        """All wrong predictions should give accuracy=0.0."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        metrics = calculate_classification_metrics(y_true, y_pred)
        assert metrics['accuracy'] == 0.0
    
    def test_confusion_matrix_shape(self):
        """Confusion matrix should have correct shape."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        metrics = calculate_classification_metrics(y_true, y_pred)
        assert metrics['confusion_matrix'].shape == (2, 2)
    
    def test_class_distribution(self):
        """Class distribution should count correctly."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        metrics = calculate_classification_metrics(y_true, y_pred)
        assert metrics['class_distribution']['0'] == 2
        assert metrics['class_distribution']['1'] == 3


class TestCalculateRegressionMetrics:
    """Test regression metrics calculation."""
    
    def test_perfect_predictions(self):
        """Perfect predictions should give R²=1.0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = calculate_regression_metrics(y_true, y_pred)
        assert metrics['r2_score'] == 1.0
        assert metrics['mse'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['mae'] == 0.0
    
    def test_constant_predictions(self):
        """Constant predictions (mean) should give R²=0.0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.5, 2.5, 2.5, 2.5])  # Mean value
        metrics = calculate_regression_metrics(y_true, y_pred)
        assert metrics['r2_score'] == 0.0
    
    def test_metrics_non_negative(self):
        """MSE, RMSE, MAE should always be non-negative."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        metrics = calculate_regression_metrics(y_true, y_pred)
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0


class TestCalculateClusteringMetrics:
    """Test clustering metrics calculation."""
    
    def test_silhouette_score_range(self):
        """Silhouette score should be in range [-1, 1]."""
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        labels = np.array([0, 0, 1, 1])
        metrics = calculate_clustering_metrics(X, labels)
        assert -1 <= metrics['silhouette_score'] <= 1
    
    def test_cluster_count(self):
        """Number of clusters should match unique labels."""
        X = np.array([[0, 0], [1, 1], [10, 10]])
        labels = np.array([0, 0, 1])
        metrics = calculate_clustering_metrics(X, labels)
        assert metrics['n_clusters'] == 2
    
    def test_cluster_sizes(self):
        """Cluster sizes should sum to total samples."""
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11], [12, 12]])
        labels = np.array([0, 0, 1, 1, 1])
        metrics = calculate_clustering_metrics(X, labels)
        total_samples = sum(metrics['cluster_sizes'].values())
        assert total_samples == 5


class TestFormatResultsMarkdown:
    """Test results markdown formatting."""
    
    def test_classification_markdown_contains_accuracy(self):
        """Classification markdown should contain accuracy."""
        metrics = {
            'accuracy': 0.95,
            'precision': 0.94,
            'recall': 0.93,
            'f1_score': 0.935,
            'confusion_matrix': np.array([[95, 5], [7, 93]]),
            'class_distribution': {'0': 100, '1': 100},
        }
        md = format_results_markdown(metrics, 'classification')
        assert 'Accuracy' in md
        assert '0.95' in md
    
    def test_regression_markdown_contains_r2(self):
        """Regression markdown should contain R² score."""
        metrics = {
            'r2_score': 0.87,
            'mse': 0.25,
            'rmse': 0.50,
            'mae': 0.40,
            'residual_stats': {'mean': 0.0, 'std': 0.5, 'min': -1.0, 'max': 1.0},
        }
        md = format_results_markdown(metrics, 'regression')
        assert 'R² Score' in md
        assert '0.87' in md
    
    def test_clustering_markdown_contains_silhouette(self):
        """Clustering markdown should contain silhouette score."""
        metrics = {
            'silhouette_score': 0.65,
            'n_clusters': 3,
            'cluster_sizes': {0: 50, 1: 40, 2: 10},
            'n_noise_points': 0,
            'inertia': 150.5,
        }
        md = format_results_markdown(metrics, 'clustering')
        assert 'Silhouette Score' in md
        assert '0.65' in md


class TestGetFeatureImportance:
    """Test feature importance extraction."""
    
    def test_random_forest_importance(self):
        """RandomForest should return feature importance."""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Get importance
        importance = get_feature_importance(model, iris.feature_names)
        
        assert importance is not None
        assert len(importance) == 4
        # Importance should sum to ~1.0 (normalized)
        assert 0.95 < sum(importance.values()) <= 1.05
    
    def test_non_tree_model_returns_none(self):
        """Non-tree models should return None."""
        from sklearn.linear_model import LogisticRegression
        
        iris = load_iris()
        X, y = iris.data, iris.target
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        importance = get_feature_importance(model, iris.feature_names)
        assert importance is None


class TestCheckDataQuality:
    """Test data quality checking."""
    
    def test_clean_data_passes(self):
        """Clean data should pass validation."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        is_valid, error = check_data_quality(df)
        assert is_valid is True
        assert error is None
    
    def test_missing_values_detected(self):
        """Missing values should be detected."""
        df = pd.DataFrame({'a': [1, 2, np.nan], 'b': [4, 5, 6]})
        is_valid, error = check_data_quality(df)
        assert is_valid is False
        assert 'Missing values' in error
    
    def test_empty_dataframe_rejected(self):
        """Empty dataframe should be rejected."""
        df = pd.DataFrame()
        is_valid, error = check_data_quality(df)
        assert is_valid is False
        assert 'empty' in error.lower()
    
    def test_missing_values_in_multiple_columns(self):
        """Missing values in multiple columns should list all."""
        df = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': [np.nan, 5, 6],
            'c': [7, 8, 9]
        })
        is_valid, error = check_data_quality(df)
        assert is_valid is False
        assert 'a' in error and 'b' in error


class TestIntegration:
    """Integration tests combining multiple utilities."""
    
    def test_full_classification_pipeline(self):
        """Test complete classification metrics pipeline."""
        # Load data
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_classification_metrics(y_test, y_pred)
        
        # Format results
        markdown = format_results_markdown(metrics, 'classification')
        
        # Get feature importance
        importance = get_feature_importance(model, iris.feature_names)
        
        # Assertions
        assert metrics['accuracy'] > 0.8
        assert 'Accuracy' in markdown
        assert importance is not None
        assert len(importance) == 4
    
    def test_full_regression_pipeline(self):
        """Test complete regression metrics pipeline."""
        # Load data
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_regression_metrics(y_test, y_pred)
        
        # Format results
        markdown = format_results_markdown(metrics, 'regression')
        
        # Assertions
        assert metrics['r2_score'] > 0.0
        assert 'R² Score' in markdown


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
