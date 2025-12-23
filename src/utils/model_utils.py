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

from typing import Dict, Any, Optional, Tuple
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
)
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
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calculate metrics
        r2 = r2_score(y_true_flat, y_pred_flat)
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        
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
            'residual_stats': residual_stats,
        }
    
    except Exception as e:
        return {
            'r2_score': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
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
        
        if n_clusters > 1 and len(X[valid_mask]) > 0:
            sil_score = silhouette_score(X[valid_mask], labels[valid_mask])
            ch_score = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
        
        # Calculate cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(cls): int(count) for cls, count in zip(unique, counts)}
        
        return {
            'silhouette_score': float(sil_score),
            'calinski_harabasz_score': float(ch_score),
            'n_clusters': int(n_clusters),
            'cluster_sizes': cluster_sizes,
            'n_noise_points': int(n_noise),
            'inertia': float(inertia) if inertia is not None else None,
        }
    
    except Exception as e:
        return {
            'silhouette_score': 0.0,
            'calinski_harabasz_score': 0.0,
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
    
    return md


def _format_regression_markdown(metrics: Dict[str, Any]) -> str:
    """Format regression metrics as markdown."""
    r2 = metrics['r2_score']
    interpretation = "Good fit" if r2 > 0.7 else "Moderate fit" if r2 > 0.4 else "Poor fit"
    
    md = f"""**R² Score:** {r2:.4f}

*R² (coefficient of determination) measures the proportion of variance explained by the model. Range: [-∞, 1]*

**Interpretation:** {interpretation} - The model explains {r2*100:.1f}% of the variance in the target variable.

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


