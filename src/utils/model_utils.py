"""
Shared utility functions for ML model training and evaluation.

This module provides centralized functions for:
- Hyperparameter validation
- Metrics calculation (classification, regression, clustering)
- Results formatting
- Feature importance extraction
- UI control management

Purpose: Eliminate duplication across model implementations (DRY principle, AR-008)
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import flet as ft


def disable_navigation_bar(page: ft.Page) -> None:
    """
    Disable all navigation bar items during model training.
    
    Args:
        page: Flet page object
    """
    if page.navigation_bar:
        for destination in page.navigation_bar.destinations:
            destination.disabled = True
        page.update()


def enable_navigation_bar(page: ft.Page) -> None:
    """
    Enable navigation bar items after model training completes.
    
    Args:
        page: Flet page object
    """
    if page.navigation_bar:
        # Enable all but first (Dataset, which should stay enabled)
        for i, destination in enumerate(page.navigation_bar.destinations):
            destination.disabled = False
        page.update()


def check_data_quality(df, target_col=None):
    """
    Check data quality before training.
    
    Args:
        df: DataFrame to check
        target_col: Target column name (for regression validation)
        
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
        - If valid: (True, None)
        - If invalid: (False, "Error description")
    """
    try:
        # Check for missing values
        if df.isnull().any().any():
            missing_cols = df.columns[df.isnull().any()].tolist()
            return False, f"Missing values detected in columns: {', '.join(missing_cols)}. Please review in Dataset Overview tab."
        
        # Check for empty dataframe
        if len(df) == 0:
            return False, "Dataset is empty. Please load a dataset first."
        
        # Check target column if provided
        if target_col and target_col in df.columns:
            # For regression, target must be numeric
            if not df[target_col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Check if target might be categorical
                if df[target_col].dtype == 'object':
                    # This is OK for classification; will be encoded
                    pass
        
        return True, None
    
    except Exception as e:
        return False, f"Data quality check error: {str(e)}"


def validate_hyperparameters(
    hyperparameters: Dict[str, Any],
    model_name: str,
    validation_rules: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate hyperparameter values for a specific model.
    
    Args:
        hyperparameters: Dictionary of hyperparameter name -> value
        model_name: Name of the model (e.g., "logistic_regression", "kmeans")
        validation_rules: Optional dict specifying min/max/allowed values per parameter
        
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
        - If valid: (True, None)
        - If invalid: (False, "Error description")
        
    Raises:
        None (catches and returns error message instead)
        
    Examples:
        >>> is_valid, error = validate_hyperparameters(
        ...     {"C": 1.0, "max_iter": 100},
        ...     "logistic_regression",
        ...     {"C": {"min": 0.001, "max": 100},
        ...      "max_iter": {"min": 1, "max": 10000}}
        ... )
        >>> assert is_valid
    """
    try:
        if not validation_rules:
            return True, None
        
        for param_name, param_value in hyperparameters.items():
            if param_name not in validation_rules:
                continue
            
            rules = validation_rules[param_name]
            
            # Type validation
            expected_type = rules.get("type")
            if expected_type and not isinstance(param_value, expected_type):
                return False, f"Parameter '{param_name}' must be of type {expected_type.__name__}"
            
            # Numeric range validation
            if "min" in rules:
                min_val = rules["min"]
                if isinstance(param_value, (int, float)) and param_value < min_val:
                    return False, f"Parameter '{param_name}' must be >= {min_val} (got {param_value})"
            
            if "max" in rules:
                max_val = rules["max"]
                if isinstance(param_value, (int, float)) and param_value > max_val:
                    return False, f"Parameter '{param_name}' must be <= {max_val} (got {param_value})"
            
            # Allowed values validation (for categorical parameters)
            if "allowed" in rules:
                allowed = rules["allowed"]
                if param_value not in allowed:
                    return False, f"Parameter '{param_name}' must be one of {allowed} (got {param_value})"
        
        return True, None
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate classification metrics (accuracy, precision, recall, F1, confusion matrix).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with keys:
        - accuracy: float [0, 1]
        - precision: float [0, 1] (weighted average for multiclass)
        - recall: float [0, 1] (weighted average for multiclass)
        - f1_score: float [0, 1] (weighted average for multiclass)
        - confusion_matrix: np.ndarray [n_classes, n_classes]
        - class_distribution: dict {class_label: count}
        
    Examples:
        >>> metrics = calculate_classification_metrics(
        ...     np.array([0, 1, 1, 0]),
        ...     np.array([0, 1, 0, 0])
        ... )
        >>> assert metrics['accuracy'] == 0.75
    """
    try:
        # Calculate metrics with weighted average for multiclass
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        class_dist = {str(cls): int(count) for cls, count in zip(unique, counts)}
        
        return {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': cm,
            'class_distribution': class_dist,
        }
    
    except Exception as e:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'confusion_matrix': None,
            'class_distribution': {},
            'error': str(e),
        }


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate regression metrics (R², MSE, RMSE, MAE, residuals).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary with keys:
        - r2_score: float [-inf, 1]
        - mse: float [0, inf)
        - rmse: float [0, inf)
        - mae: float [0, inf)
        - residual_stats: dict {mean, std, min, max} of residuals
        
    Examples:
        >>> metrics = calculate_regression_metrics(
        ...     np.array([1.0, 2.0, 3.0]),
        ...     np.array([1.1, 2.0, 2.9])
        ... )
        >>> assert 0.9 < metrics['r2_score'] <= 1.0
    """
    try:
        # Flatten predictions if needed
        y_true_flat = y_true
        y_pred_flat = y_pred
        
        # Calculate metrics
        r2 = r2_score(y_true_flat, y_pred_flat)
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        
        # Adjusted R² calculation
        n = len(y_true_flat)
        p = y_pred_flat.shape[1] if len(y_pred_flat.shape) > 1 else 1
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else 0.0

        # Calculate residual statistics
        residuals = y_true_flat - y_pred_flat
        residual_stats = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
        }
        
        return {
            'r2_score': float(r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'adjusted_r2_score': float(adjusted_r2),
            'residual_stats': residual_stats,
        }
    
    except Exception as e:
        return {
            'r2_score': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'adjusted_r2_score': 0.0,
            'residual_stats': {},
            'error': str(e),
        }


def calculate_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    inertia: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate clustering metrics (silhouette score, Calinski-Harabasz score, cluster statistics).
    
    Args:
        X: Feature array [n_samples, n_features]
        labels: Cluster labels for each sample
        inertia: Optional inertia value (for K-Means)
        
    Returns:
        Dictionary with keys:
        - silhouette_score: float [-1, 1]
        - calinski_harabasz_score: float [0, inf) (higher is better)
        - n_clusters: int (number of clusters found)
        - cluster_sizes: dict {cluster_id: count}
        - n_noise_points: int (for DBSCAN; -1 labels)
        - inertia: float or None
        
    Examples:
        >>> X = np.array([[0, 0], [1, 1], [10, 10]])
        >>> labels = np.array([0, 0, 1])
        >>> metrics = calculate_clustering_metrics(X, labels)
        >>> assert metrics['n_clusters'] == 2
    """
    try:
        # Handle DBSCAN noise points (-1 label)
        valid_mask = labels != -1
        n_noise = np.sum(labels == -1)
        
        # Calculate silhouette score only if more than 1 cluster
        unique_labels = np.unique(labels[valid_mask])
        n_clusters = len(unique_labels)
        
        sil_score = 0.0
        ch_score = 0.0
        db_score = 0.0
        
        if n_clusters > 1 and len(X[valid_mask]) > 0:
            sil_score = silhouette_score(X[valid_mask], labels[valid_mask])
            ch_score = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
            db_score = davies_bouldin_score(X[valid_mask], labels[valid_mask])
        
        # Calculate cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(cls): int(count) for cls, count in zip(unique, counts)}
        
        return {
            'silhouette_score': float(sil_score),
            'calinski_harabasz_score': float(ch_score),
            'davies_bouldin_score': float(db_score),
            'n_clusters': int(n_clusters),
            'cluster_sizes': cluster_sizes,
            'n_noise_points': int(n_noise),
            'inertia': float(inertia) if inertia is not None else None,
        }
    
    except Exception as e:
        return {
            'silhouette_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'davies_bouldin_score': 0.0,
            'n_clusters': 0,
            'cluster_sizes': {},
            'n_noise_points': 0,
            'inertia': None,
            'error': str(e),
        }


def format_results_markdown(
    metrics: Dict[str, Any],
    task_type: str = "classification",
) -> str:
    """
    Format evaluation metrics as markdown for display in results dialog.
    
    Args:
        metrics: Dictionary of metrics from calculate_*_metrics()
        task_type: One of "classification", "regression", "clustering"
        
    Returns:
        Formatted markdown string suitable for ft.Markdown() display
        
    Examples:
        >>> metrics = {
        ...     'accuracy': 0.95,
        ...     'precision': 0.93,
        ...     'recall': 0.94,
        ...     'f1_score': 0.93,
        ...     'confusion_matrix': np.array([[90, 5], [5, 100]])
        ... }
        >>> md = format_results_markdown(metrics, "classification")
        >>> assert "**Accuracy**" in md
    """
    try:
        if task_type == "classification":
            return _format_classification_markdown(metrics)
        elif task_type == "regression":
            return _format_regression_markdown(metrics)
        elif task_type == "clustering":
            return _format_clustering_markdown(metrics)
        else:
            return "Unknown task type"
    
    except Exception as e:
        return f"Error formatting results: {str(e)}"


def _format_classification_markdown(metrics: Dict[str, Any]) -> str:
    """Format classification metrics as markdown."""
    md = f"""**Accuracy:** {metrics['accuracy']:.4f}

*Accuracy measures the proportion of correct predictions out of total predictions.*

---

**Precision:** {metrics['precision']:.4f}

*Precision measures the accuracy of positive predictions: of all positive predictions, how many were correct.*

---

**Recall:** {metrics['recall']:.4f}

*Recall measures the proportion of actual positives correctly identified: of all actual positives, how many did we find.*

---

**F1 Score:** {metrics['f1_score']:.4f}

*F1 is the harmonic mean of precision and recall, balancing both metrics.*

---

**Confusion Matrix:**

"""
    if metrics['confusion_matrix'] is not None:
        cm = metrics['confusion_matrix']
        n_classes = cm.shape[0]
        
        if n_classes == 2:
            # Binary classification - simple table
            md += "| | Predicted Negative | Predicted Positive |\n"
            md += "|---|---|---|\n"
            md += f"| **Actual Negative** | {cm[0,0]} | {cm[0,1]} |\n"
            md += f"| **Actual Positive** | {cm[1,0]} | {cm[1,1]} |\n"
        else:
            # Multiclass - show matrix
            md += "| Predicted \\ Actual |"
            for j in range(n_classes):
                md += f" Class {j} |"
            md += "\n"
            md += "|---|" + "|".join(["---"] * n_classes) + "|\n"
            for i in range(n_classes):
                md += f"| **Class {i}** |"
                for j in range(n_classes):
                    md += f" {cm[i,j]} |"
                md += "\n"
    
    # Add class distribution if available
    if 'class_distribution' in metrics and metrics['class_distribution']:
        md += "\n\n**Class Distribution in Test Set:**\n\n"
        for cls, count in sorted(metrics['class_distribution'].items()):
            md += f"- Class {cls}: {count} samples\n"
    
    if 'CV' in metrics:
        md += "\n"
        md += "---\n"
        md += f"**Cross Validation scores:** {metrics['CV'].mean():.4f}\n"
        cvs = metrics['CV']
        n_folds = len(cvs)
        for i in range(1, n_folds + 1):
            md += f" Fold {i} |"
        md += "\n"
        md += "|".join(["---"] * n_folds) + "|\n"
        for score in cvs:
            md += f"| **{score:.4f}** "
    
    return md


def _format_regression_markdown(metrics: Dict[str, Any]) -> str:
    """Format regression metrics as markdown."""
    r2 = metrics['r2_score']
    interpretation = "Good fit" if r2 > 0.7 else "Moderate fit" if r2 > 0.4 else "Poor fit"
    
    md = f"""**R² Score:** {r2:.4f}

*R² (coefficient of determination) measures the proportion of variance explained by the model. Range: [-∞, 1]*

**Interpretation:** {interpretation} - The model explains {r2*100:.1f}% of the variance in the target variable.

---

**Adjusted R² Score:** {metrics['adjusted_r2_score']:.4f}

*Adjusted R² accounts for the number of predictors (features) in the model. It penalizes adding irrelevant features. Range: [-∞, 1]*

---

**Mean Squared Error (MSE):** {metrics['mse']:.4f}

*MSE is the average of squared differences between predicted and actual values. Lower is better.*

---

**Root Mean Squared Error (RMSE):** {metrics['rmse']:.4f}

*RMSE is the square root of MSE, in the same units as the target variable. Easier to interpret than MSE.*

---

**Mean Absolute Error (MAE):** {metrics['mae']:.4f}

*MAE is the average absolute difference between predicted and actual values. Lower is better.*

"""
    if 'CV' in metrics:
        md += "\n"
        md += "---\n"
        md += f"**Cross Validation scores:** {metrics['CV'].mean():.4f}\n"
        cvs = metrics['CV']
        n_folds = len(cvs)
        for i in range(1, n_folds + 1):
            md += f" Fold {i} |"
        md += "\n"
        md += "|".join(["---"] * n_folds) + "|\n"
        for score in cvs:
            md += f"| **{score:.4f}** "
    
    # Add residual statistics if available
    if 'residual_stats' in metrics and metrics['residual_stats']:
        stats = metrics['residual_stats']
        md += f"\n\n**Residual Statistics:**\n\n"
        md += f"- Mean: {stats['mean']:.4f}\n"
        md += f"- Std Dev: {stats['std']:.4f}\n"
        md += f"- Min: {stats['min']:.4f}\n"
        md += f"- Max: {stats['max']:.4f}\n"
    
    return md


def _format_clustering_markdown(metrics: Dict[str, Any]) -> str:
    """Format clustering metrics as markdown."""
    md = f"""**Silhouette Score:** {metrics['silhouette_score']:.4f}

*Silhouette score measures cluster cohesion and separation. Range: [-1, 1]*

**Interpretation:**
- 0.51-1.0: Strong structure
- 0.26-0.50: Reasonable structure  
- <0.25: Weak structure
- Negative: Overlapping clusters

---

**Calinski-Harabasz Score:** {metrics['calinski_harabasz_score']:.4f}

*Calinski-Harabasz score is the ratio of between-cluster to within-cluster dispersion. Higher is better (better defined clusters).*

---

**Davies-Bouldin Score:** {metrics['davies_bouldin_score']:.4f}

*Davies-Bouldin score measures average similarity ratio of each cluster with its most similar cluster. Lower is better (well-separated clusters).*

---

**Number of Clusters:** {metrics['n_clusters']}

**Cluster Sizes:**

"""
    
    for cluster_id, count in sorted(metrics['cluster_sizes'].items()):
        if cluster_id != -1:  # Skip noise points (-1) in this section
            md += f"- Cluster {cluster_id}: {count} points\n"
    
    # Add noise points info if present (for DBSCAN)
    if metrics['n_noise_points'] > 0:
        md += f"\n**Noise Points:** {metrics['n_noise_points']}\n"
    
    # Add inertia if available (K-Means)
    if metrics['inertia'] is not None:
        md += f"\n**Inertia:** {metrics['inertia']:.4f}\n\n*Inertia is the sum of squared distances from cluster centers. Lower is better for K-Means.*\n"

    # BIC score and AIC score for Gaussian Mixture Models could be added here if available.
    if 'bic_score' in metrics:
        md += "\n---\n"
        md += f"\n**BIC Score:** {metrics['bic_score']:.4f}\n\n*Bayesian Information Criterion (BIC) for model selection. Lower is better.*\n"
        md += f"\n**AIC Score:** {metrics['aic_score']:.4f}\n\n*Akaike Information Criterion (AIC) for model selection. Lower is better.*\n"
    
    return md


def get_feature_importance(
    model: Any,
    feature_names: list,
) -> Optional[Dict[str, float]]:
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Fitted sklearn estimator or Pipeline
        feature_names: List of feature column names
        
    Returns:
        Dictionary of {feature_name: importance_value} or None if model doesn't support feature importance
        
    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> clf = RandomForestClassifier(random_state=42).fit(X, y)
        >>> importance = get_feature_importance(clf, ['f0', 'f1', 'f2', 'f3'])
        >>> assert len(importance) == 4
    """
    try:
        # Handle Pipeline objects
        if hasattr(model, 'named_steps'):
            # Get the final estimator from pipeline
            estimator = model.named_steps.get(list(model.named_steps.keys())[-1])
        else:
            estimator = model
        
        # Check if model has feature_importances_ attribute (tree-based models)
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            
            # Normalize importances to sum to 1 for easier interpretation
            importances = importances / importances.sum()
            
            # Create dictionary of feature -> importance
            importance_dict = {}
            for fname, fimportance in zip(feature_names, importances):
                importance_dict[fname] = float(fimportance)
            
            return importance_dict
        
        return None
    
    except Exception as e:
        return None


def format_feature_importance_markdown(
    importance_dict: Dict[str, float],
    top_n: int = 10,
) -> str:
    """
    Format feature importance dictionary as markdown table.
    
    Args:
        importance_dict: Dictionary of {feature_name: importance_value}
        top_n: Number of top features to display
        
    Returns:
        Formatted markdown string with feature importance table
        
    Examples:
        >>> importance = {'age': 0.4, 'income': 0.3, 'score': 0.3}
        >>> md = format_feature_importance_markdown(importance, top_n=3)
        >>> assert 'age' in md
    """
    if not importance_dict:
        return "*No feature importance data available*"
    
    try:
        # Sort by importance descending
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        md = "**Feature Importance** (Top Features)\n\n"
        md += "| Feature | Importance |\n"
        md += "|---|---|\n"
        
        for feature_name, importance_value in top_features:
            # Create visual bar representation
            bar_width = int(importance_value * 50)  # Scale to max 50 chars
            bar = "█" * bar_width
            md += f"| {feature_name} | {importance_value:.4f} {bar} |\n"
        
        return md
    
    except Exception as e:
        return f"*Error formatting feature importance: {str(e)}*"


def create_results_dialog(
    page: ft.Page,
    title: str,
    result_text: str,
    model_name: str = "Model",
) -> "ft.AlertDialog":
    """
    Create a results dialog with copy button functionality.
    
    Args:
        page: Flet Page object for opening dialog and clipboard operations
        title: Title of the results dialog
        result_text: Markdown-formatted results text
        model_name: Name of the model for dialog title context
        
    Returns:
        ft.AlertDialog with copy button and markdown content
        
    Examples:
        >>> dialog = create_results_dialog(page, "Classification Results", "**Accuracy:** 0.95")
        >>> assert isinstance(dialog, ft.AlertDialog)
    """
    def _copy_results(e):
        """Copy results text to clipboard."""
        try:
            # Plain text version for clipboard
            plain_text = result_text.replace("**", "").replace("*", "").replace("##", "")
            page.set_clipboard(plain_text)
            page.open(ft.SnackBar(
                ft.Text("Results copied to clipboard!", font_family="SF regular")
            ))
        except Exception as ex:
            page.open(ft.SnackBar(
                ft.Text(f"Copy failed: {str(ex)}", font_family="SF regular")
            ))
    
    # Create actions row with Close and Copy buttons
    dialog = ft.AlertDialog(
        modal=True,
        title=ft.Row([
            ft.Text(title,
                   font_family="SF thin",
                   size=20,
                   expand=1,
                   text_align="center")
        ]),
        content=ft.Container(
            expand=True,
            content=ft.Column(
                scroll=ft.ScrollMode.AUTO,
                controls=[
                    ft.Markdown(
                        result_text,
                        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                        selectable=True,
                    ),
                ]
            )
        ),
        actions_alignment=ft.MainAxisAlignment.CENTER,
        actions=[
            ft.Row([
                ft.OutlinedButton(
                    text="Copy Results",
                    icon=ft.Icons.COPY,
                    expand=1,
                    on_click=_copy_results,
                ),
                ft.FilledButton(
                    text="Close",
                    expand=1,
                    on_click=lambda _: page.close(dialog),  # Will close any open dialog
                )
            ])
        ],
    )
    
    return dialog


# ==================== CATEGORICAL ENCODING UTILITIES (Phase 1) ====================


class EncodingError(Exception):
    """
    Exception raised when categorical encoding fails.
    
    Attributes:
        message: Error message describing the issue
        column: Column name where encoding failed (optional)
        unseen_values: List of unseen categorical values (optional)
        original_classes: Expected class values (optional)
    """
    
    def __init__(
        self,
        message: str,
        column: Optional[str] = None,
        unseen_values: Optional[List[Any]] = None,
        original_classes: Optional[List[Any]] = None,
    ):
        self.message = message
        self.column = column
        self.unseen_values = unseen_values
        self.original_classes = original_classes
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format a user-friendly error message."""
        if self.column and self.unseen_values:
            unseen_str = ", ".join(str(v) for v in self.unseen_values[:5])
            if len(self.unseen_values) > 5:
                unseen_str += f", ... and {len(self.unseen_values) - 5} more"
            
            expected_str = ", ".join(str(v) for v in self.original_classes[:10]) if self.original_classes else "unknown"
            
            return (
                f"Encoding Error in column '{self.column}':\n"
                f"  Found unseen values: {unseen_str}\n"
                f"  Expected values: {expected_str}\n"
                f"  Action: Review your data for unexpected values.\n"
                f"  Details: {self.message}"
            )
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize error to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "column": self.column,
            "unseen_values": self.unseen_values,
            "original_classes": self.original_classes,
        }


@dataclass
class CardinalityWarning:
    """
    Warning for high-cardinality categorical columns.
    
    Attributes:
        column: Column name
        cardinality: Number of unique values
        threshold: Cardinality threshold
        severity: "warning" or "error"
        message: User-friendly message
    """
    column: str
    cardinality: int
    threshold: int = 1000
    severity: str = "warning"
    message: Optional[str] = None
    
    def __post_init__(self):
        """Generate message if not provided."""
        if not self.message:
            if self.cardinality > self.threshold:
                self.message = (
                    f"Column '{self.column}' has {self.cardinality} unique values "
                    f"(exceeds threshold {self.threshold}). "
                    f"Consider reviewing this column."
                )
            else:
                self.message = (
                    f"Column '{self.column}' has {self.cardinality} unique values."
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize warning to dictionary."""
        return asdict(self)


@dataclass
class CategoricalEncodingInfo:
    """
    Metadata for a categorical column's encoding.
    
    Attributes:
        column_name: Name of the column
        cardinality: Number of unique values
        original_classes: List of original class values
        mapping: Dict mapping original value -> encoded integer
        cardinality_warning: Whether column exceeds cardinality threshold
    """
    column_name: str
    cardinality: int
    original_classes: List[Any]
    mapping: Dict[Any, int]
    cardinality_warning: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize encoding info to dictionary."""
        return {
            "column_name": self.column_name,
            "cardinality": self.cardinality,
            "original_classes": [str(v) for v in self.original_classes],
            "mapping": {str(k): int(v) for k, v in self.mapping.items()},
            "cardinality_warning": self.cardinality_warning,
        }


def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect all categorical (object dtype) columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Sorted list of column names with object dtype
        
    Examples:
        >>> df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        >>> detect_categorical_columns(df)
        ['name']
    """
    if df.empty:
        return []
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return sorted(cat_cols)


def validate_cardinality(
    df: pd.DataFrame,
    columns: List[str],
    threshold: int = 1000,
) -> Dict[str, CardinalityWarning]:
    """
    Validate cardinality (unique value count) for categorical columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to validate
        threshold: Cardinality threshold (default: 1000)
        
    Returns:
        Dictionary mapping column_name -> CardinalityWarning for columns exceeding threshold
        
    Examples:
        >>> df = pd.DataFrame({"color": ["red", "blue"] * 1000})
        >>> warnings = validate_cardinality(df, ["color"], threshold=1000)
        >>> len(warnings)
        0
    """
    warnings = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        unique_count = df[col].nunique()
        
        if unique_count > threshold:
            warnings[col] = CardinalityWarning(
                column=col,
                cardinality=unique_count,
                threshold=threshold,
                severity="warning",
            )
    
    return warnings


def create_categorical_encoders(
    X_train: pd.DataFrame,
    categorical_cols: List[str],
) -> Dict[str, LabelEncoder]:
    """
    Create and fit LabelEncoders for categorical columns.
    
    **CRITICAL**: Encoders are fitted ONLY on training data to prevent data leakage.
    
    Args:
        X_train: Training feature DataFrame
        categorical_cols: List of categorical column names
        
    Returns:
        Dictionary mapping column_name -> fitted LabelEncoder
        
    Raises:
        EncodingError: If categorical column not found in X_train
        
    Examples:
        >>> df = pd.DataFrame({"color": ["red", "blue", "red"]})
        >>> encoders = create_categorical_encoders(df, ["color"])
        >>> encoders["color"].transform(["red", "blue"])
        array([1, 0], dtype=int64)
    """
    encoders = {}
    
    for col in categorical_cols:
        if col not in X_train.columns:
            raise EncodingError(
                message=f"Column '{col}' not found in training data",
                column=col,
            )
        
        encoder = LabelEncoder()
        encoder.fit(X_train[col].astype(str))  # Convert to string to handle mixed types
        encoders[col] = encoder
    
    return encoders


def apply_encoders(
    encoders: Dict[str, LabelEncoder],
    X: pd.DataFrame,
    raise_on_unknown: bool = True,
) -> pd.DataFrame:
    """
    Apply fitted encoders to a DataFrame.
    
    Args:
        encoders: Dictionary of fitted LabelEncoders (from create_categorical_encoders)
        X: DataFrame to encode (can be training or test data)
        raise_on_unknown: If True, raise error on unseen categories; if False, log warning
        
    Returns:
        Copy of X with categorical columns encoded as integers
        
    Raises:
        EncodingError: If raise_on_unknown=True and unseen categories found
        
    Examples:
        >>> df_train = pd.DataFrame({"color": ["red", "blue"]})
        >>> df_test = pd.DataFrame({"color": ["red", "blue"]})
        >>> encoders = create_categorical_encoders(df_train, ["color"])
        >>> encoded = apply_encoders(encoders, df_test)
        >>> encoded["color"].tolist()
        [1, 0]
    """
    X_copy = X.copy()
    
    for col, encoder in encoders.items():
        if col not in X_copy.columns:
            continue
        
        # Convert to string for consistent comparison
        X_values_str = X_copy[col].astype(str)
        encoder_classes_str = set(encoder.classes_)
        
        # Detect unseen values
        unseen = set(X_values_str.unique()) - encoder_classes_str
        
        if unseen:
            unseen_list = sorted(list(unseen))
            
            if raise_on_unknown:
                raise EncodingError(
                    message=f"Column '{col}' contains {len(unseen)} unseen categorical value(s)",
                    column=col,
                    unseen_values=unseen_list,
                    original_classes=list(encoder.classes_),
                )
            else:
                # Log warning (skip for MVP)
                pass
        
        # Apply encoder
        X_copy[col] = encoder.transform(X_values_str)
    
    return X_copy


def get_encoding_mappings(
    encoders: Dict[str, LabelEncoder],
) -> Dict[str, Dict[Any, int]]:
    """
    Extract original → encoded value mappings from fitted encoders.
    
    Args:
        encoders: Dictionary of fitted LabelEncoders
        
    Returns:
        Nested dict: column_name → {original_value: encoded_int}
        
    Examples:
        >>> df = pd.DataFrame({"color": ["red", "blue", "red"]})
        >>> encoders = create_categorical_encoders(df, ["color"])
        >>> mappings = get_encoding_mappings(encoders)
        >>> mappings["color"]["red"]
        1
    """
    mappings = {}
    
    for col, encoder in encoders.items():
        col_mapping = {}
        for i, class_label in enumerate(encoder.classes_):
            col_mapping[class_label] = i
        mappings[col] = col_mapping
    
    return mappings


def get_categorical_encoding_info(
    encoders: Dict[str, LabelEncoder],
    cardinality_warnings: Optional[Dict[str, CardinalityWarning]] = None,
) -> Dict[str, CategoricalEncodingInfo]:
    """
    Create CategoricalEncodingInfo objects for each encoded column.
    
    Args:
        encoders: Dictionary of fitted LabelEncoders
        cardinality_warnings: Optional dict of CardinalityWarnings
        
    Returns:
        Dictionary mapping column_name -> CategoricalEncodingInfo
        
    Examples:
        >>> df = pd.DataFrame({"color": ["red", "blue", "red"]})
        >>> encoders = create_categorical_encoders(df, ["color"])
        >>> info = get_categorical_encoding_info(encoders)
        >>> info["color"].cardinality
        2
    """
    cardinality_warnings = cardinality_warnings or {}
    info = {}
    
    for col, encoder in encoders.items():
        original_classes = list(encoder.classes_)
        mapping = {original_classes[i]: i for i in range(len(original_classes))}
        
        info[col] = CategoricalEncodingInfo(
            column_name=col,
            cardinality=len(original_classes),
            original_classes=original_classes,
            mapping=mapping,
            cardinality_warning=col in cardinality_warnings,
        )
    
    return info


def build_preprocessing_pipeline(
    categorical_cols: List[str],
    numeric_cols: List[str],
    categorical_encoders: Dict[str, LabelEncoder],
    numeric_scaler: str = "standard",
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing mixed categorical/numeric data.
    
    **Important Note**: This function builds a ColumnTransformer that is suitable for
    transforming data AFTER encoders are fitted. It should NOT be used with Pipeline.fit()
    on new data (which would try to refit encoders). Instead, data should be pre-encoded
    using prepare_data_for_training() before being passed to models.
    
    **Order**: Categorical transformations followed by numeric transformations,
    maintaining consistent column order for reproducibility.
    
    Args:
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        categorical_encoders: Pre-fitted LabelEncoders for categorical columns
        numeric_scaler: Type of numeric scaling ("standard", "minmax", or "none")
        
    Returns:
        Configured ColumnTransformer ready for transform operations
        
    Raises:
        ValueError: If no columns provided
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.base import BaseEstimator, TransformerMixin
    
    transformers = []
    
    # Helper class that works with sklearn ColumnTransformer
    class PreFittedEncoderTransformer(BaseEstimator, TransformerMixin):
        """Transformer that applies pre-fitted LabelEncoders to categorical data."""
        def __init__(self, encoders_dict: Dict[str, LabelEncoder], col_list: List[str]):
            self.encoders_dict = encoders_dict
            self.col_list = col_list
        
        def fit(self, X, y=None):
            # Never refit encoders - they should already be fitted
            return self
        
        def transform(self, X):
            X_copy = X.copy()
            for col in self.col_list:
                if col in X_copy.columns and col in self.encoders_dict:
                    X_copy[col] = self.encoders_dict[col].transform(X_copy[col].astype(str))
            return X_copy[self.col_list]
        
        def get_params(self, deep=True):
            # Return minimal params - encoders are not cloned
            return {'encoders_dict': self.encoders_dict, 'col_list': self.col_list}
        
        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self
    
    # Categorical transformer: Use pre-fitted LabelEncoders
    if categorical_cols:
        cat_transformer = PreFittedEncoderTransformer(categorical_encoders, categorical_cols)
        transformers.append(("categorical", cat_transformer, categorical_cols))
    
    # Numeric transformer: Scale numeric columns
    if numeric_cols:
        if numeric_scaler.lower() == "standard":
            scaler = StandardScaler()
        elif numeric_scaler.lower() == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = None  # Passthrough
        
        if scaler:
            transformers.append(("numeric", scaler, numeric_cols))
        else:
            transformers.append(("numeric", "passthrough", numeric_cols))
    
    if not transformers:
        raise ValueError("Must provide at least one column (categorical or numeric)")
    
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drop any remaining columns
        sparse_threshold=0,  # Keep as DataFrame
    )


def compose_full_model_pipeline(
    preprocessor: ColumnTransformer,
    model_estimator,
) -> Pipeline:
    """
    Compose a full sklearn Pipeline combining preprocessing and model estimator.
    
    Args:
        preprocessor: ColumnTransformer with preprocessing steps
        model_estimator: sklearn estimator (model) to add after preprocessing
        
    Returns:
        Configured Pipeline ready for fit/predict
        
    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> df_train = pd.DataFrame({
        ...     "color": ["red", "blue"],
        ...     "age": [25, 30]
        ... })
        >>> y_train = [0, 1]
        >>> encoders = create_categorical_encoders(df_train, ["color"])
        >>> transformer = build_preprocessing_pipeline(
        ...     ["color"], ["age"], encoders
        ... )
        >>> model = LogisticRegression()
        >>> pipeline = compose_full_model_pipeline(transformer, model)
        >>> pipeline.fit(df_train, y_train)
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model_estimator),
    ])
