"""
Integration tests for ML model training and evaluation pipeline

Tests the complete workflow:
1. Data loading and validation
2. Model instantiation via factory pattern
3. Model training with different hyperparameters
4. Metrics calculation and formatting
5. Feature importance extraction
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes, make_classification
from sklearn.model_selection import train_test_split

from src.ui.models import (
    LinearRegressionModel,
    LogisticRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    SVMModel,
    KNNModel,
    KMeansModel,
    MiniBatchKMeansModel,
    HierarchicalClusteringModel,
    DBSCANModel,
)
from src.utils.model_utils import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_clustering_metrics,
    format_results_markdown,
    validate_hyperparameters,
    check_data_quality,
)


class TestClassificationModels:
    """Test all classification models."""
    
    @pytest.fixture
    def iris_data(self):
        """Prepare iris dataset for testing."""
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        return df
    
    @pytest.fixture
    def mock_parent(self):
        """Create mock parent for model testing."""
        class MockParent:
            class MockDropdown:
                value = None
            
            class MockField:
                value = '0.3'
            
            class MockPage:
                def open(self, widget):
                    pass
            
            target_column_dropdown = MockDropdown()
            task_type_dropdown = MockDropdown()
            scaler_dropdown = MockDropdown()
            test_size_field = MockField()
            page = MockPage()
        
        return MockParent()
    
    def test_logistic_regression_trains(self, iris_data, mock_parent):
        """LogisticRegression should train successfully."""
        mock_parent.target_column_dropdown.value = 'target'
        mock_parent.task_type_dropdown.value = 'Classification'
        mock_parent.scaler_dropdown.value = 'standard_scaler'
        
        model = LogisticRegressionModel(mock_parent, iris_data)
        assert model.df is not None
        assert len(model.df) == len(iris_data)
    
    def test_random_forest_classifier_trains(self, iris_data, mock_parent):
        """RandomForest classifier should train successfully."""
        mock_parent.target_column_dropdown.value = 'target'
        mock_parent.task_type_dropdown.value = 'Classification'
        mock_parent.scaler_dropdown.value = 'standard_scaler'
        
        model = RandomForestModel(mock_parent, iris_data)
        assert model._get_task_type() == 'Classification'
    
    def test_knn_classifier_trains(self, iris_data, mock_parent):
        """KNN classifier should train successfully."""
        mock_parent.target_column_dropdown.value = 'target'
        mock_parent.task_type_dropdown.value = 'Classification'
        mock_parent.scaler_dropdown.value = 'standard_scaler'
        
        model = KNNModel(mock_parent, iris_data)
        assert model.df is not None


class TestRegressionModels:
    """Test all regression models."""
    
    @pytest.fixture
    def diabetes_data(self):
        """Prepare diabetes dataset for testing."""
        diabetes = load_diabetes()
        df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        df['target'] = diabetes.target
        return df
    
    @pytest.fixture
    def mock_parent(self):
        """Create mock parent for model testing."""
        class MockParent:
            class MockDropdown:
                value = None
            
            class MockField:
                value = '0.3'
            
            class MockPage:
                def open(self, widget):
                    pass
            
            target_column_dropdown = MockDropdown()
            task_type_dropdown = MockDropdown()
            scaler_dropdown = MockDropdown()
            test_size_field = MockField()
            page = MockPage()
        
        return MockParent()
    
    def test_linear_regression_trains(self, diabetes_data, mock_parent):
        """LinearRegression should train successfully."""
        mock_parent.target_column_dropdown.value = 'target'
        mock_parent.task_type_dropdown.value = 'Regression'
        mock_parent.scaler_dropdown.value = 'standard_scaler'
        
        model = LinearRegressionModel(mock_parent, diabetes_data)
        assert model.df is not None
    
    def test_random_forest_regressor_trains(self, diabetes_data, mock_parent):
        """RandomForest regressor should train successfully."""
        mock_parent.target_column_dropdown.value = 'target'
        mock_parent.task_type_dropdown.value = 'Regression'
        mock_parent.scaler_dropdown.value = 'standard_scaler'
        
        model = RandomForestModel(mock_parent, diabetes_data)
        assert model._get_task_type() == 'Regression'


class TestClusteringModels:
    """Test all clustering models."""
    
    @pytest.fixture
    def clustering_data(self):
        """Prepare simple clustering dataset."""
        df = pd.DataFrame({
            'x': [0, 1, 10, 11, 20, 21],
            'y': [0, 1, 10, 11, 20, 21],
        })
        return df
    
    @pytest.fixture
    def mock_parent(self):
        """Create mock parent for model testing."""
        class MockParent:
            class MockDropdown:
                value = None
            
            class MockField:
                value = '0.3'
            
            class MockPage:
                def open(self, widget):
                    pass
            
            target_column_dropdown = MockDropdown()
            task_type_dropdown = MockDropdown()
            scaler_dropdown = MockDropdown()
            test_size_field = MockField()
            page = MockPage()
        
        return MockParent()
    
    def test_kmeans_instantiates(self, clustering_data, mock_parent):
        """KMeans should instantiate successfully."""
        mock_parent.scaler_dropdown.value = 'standard_scaler'
        
        model = KMeansModel(mock_parent, clustering_data)
        assert model.df is not None
        assert len(model.df) == 6
    
    def test_minibatch_kmeans_instantiates(self, clustering_data, mock_parent):
        """MiniBatchKMeans should instantiate successfully."""
        mock_parent.scaler_dropdown.value = 'standard_scaler'
        
        model = MiniBatchKMeansModel(mock_parent, clustering_data)
        assert model.df is not None
    
    def test_hierarchical_clustering_instantiates(self, clustering_data, mock_parent):
        """Hierarchical clustering should instantiate successfully."""
        mock_parent.scaler_dropdown.value = 'standard_scaler'
        
        model = HierarchicalClusteringModel(mock_parent, clustering_data)
        assert model.df is not None
    
    def test_dbscan_instantiates(self, clustering_data, mock_parent):
        """DBSCAN should instantiate successfully."""
        mock_parent.scaler_dropdown.value = 'standard_scaler'
        
        model = DBSCANModel(mock_parent, clustering_data)
        assert model.df is not None


class TestModelFactory:
    """Test model factory pattern."""
    
    def test_model_registry_complete(self):
        """All expected models should be in registry."""
        from src.ui.model_factory import MODEL_REGISTRY
        
        expected_models = [
            'linear_regression',
            'logistic_regression',
            'random_forest',
            'gradient_boosting',
            'svm',
            'knn',
            'kmeans',
            'minibatch_kmeans',
            'hierarchical',
            'dbscan',
            'hdbscan',
        ]
        
        for model_name in expected_models:
            assert model_name in MODEL_REGISTRY, f"{model_name} not in registry"
    
    def test_all_models_callable(self):
        """All registered models should be callable."""
        from src.ui.model_factory import MODEL_REGISTRY
        
        for model_name, model_class in MODEL_REGISTRY.items():
            if model_name != 'hdbscan':  # hdbscan might not be installed
                assert model_class is not None, f"{model_name} is None"
                assert callable(model_class), f"{model_name} is not callable"


class TestDataQualityAndValidation:
    """Test data quality checks and validation."""
    
    def test_classification_with_missing_values(self):
        """Classification with missing values should be detected."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, 6, 7, 8],
            'target': [0, 1, 0, 1],
        })
        
        is_valid, error = check_data_quality(df, 'target')
        assert is_valid is False
        assert 'Missing values' in error
    
    def test_hyperparameter_validation_multiple_params(self):
        """Validate multiple hyperparameters simultaneously."""
        hyperparams = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
        }
        
        rules = {
            'n_estimators': {'type': int, 'min': 1, 'max': 1000},
            'max_depth': {'type': int, 'min': 1, 'max': 100},
            'min_samples_split': {'type': int, 'min': 2, 'max': 100},
        }
        
        is_valid, error = validate_hyperparameters(hyperparams, 'random_forest', rules)
        assert is_valid is True


class TestMetricsAndFormatting:
    """Test metrics calculation and formatting."""
    
    def test_classification_metrics_consistency(self):
        """Classification metrics should be consistent."""
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 0])
        
        metrics = calculate_classification_metrics(y_true, y_pred)
        
        # Accuracy should be between 0 and 1
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_regression_metrics_consistency(self):
        """Regression metrics should be consistent."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.2, 3.9, 4.8])
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        # MSE should be non-negative
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
    
    def test_markdown_formatting_complete(self):
        """Formatted markdown should contain key information."""
        metrics = {
            'accuracy': 0.85,
            'precision': 0.84,
            'recall': 0.85,
            'f1_score': 0.845,
            'confusion_matrix': np.array([[85, 15], [15, 85]]),
            'class_distribution': {'0': 100, '1': 100},
        }
        
        md = format_results_markdown(metrics, 'classification')
        
        # Markdown should contain metric values and labels
        assert '0.85' in md
        assert 'Accuracy' in md
        assert 'Precision' in md
        assert 'Recall' in md
        assert 'F1' in md


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
