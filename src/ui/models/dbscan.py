"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

Density-based clustering that finds clusters of varying shapes and sizes.
Can identify outliers (noise points) not assigned to any cluster.

Configurable hyperparameters:
- eps: Maximum distance between samples in a neighborhood
- min_samples: Minimum samples to form a core point
- metric: Distance metric ('euclidean', 'manhattan', etc.)
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer

from utils.model_utils import (
    validate_hyperparameters,
    calculate_clustering_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from core.data_preparation import prepare_data_for_training_no_split

if TYPE_CHECKING:
    from ..model_factory import ModelFactory


@dataclass
class DBSCANModel:
    """DBSCAN clustering model."""
    
    parent: ModelFactory
    df: DataFrame
    
    def __post_init__(self):
        """Ensure dataset is copied to avoid mutations."""
        self.df = self.df.copy()
    
    def _prepare_data(self) -> Optional[Tuple]:
        """
        Prepare data for clustering with categorical encoding support (no train-test split).
        
        Uses prepare_data_for_training_no_split() which:
        - Detects and encodes categorical columns
        - Validates cardinality
        - Fits encoders on full dataset (clustering is unsupervised)
        - Returns encoded features and metadata
        
        Returns:
            Tuple of (X_encoded, categorical_cols, numeric_cols, encoders, warnings) or None if error
        """
        try:
            # Call spec-compliant data preparation for clustering (no split)
            (
                X_encoded,
                _,  # y is None for clustering
                categorical_cols,
                numeric_cols,
                encoders,
                cardinality_warnings,
            ) = prepare_data_for_training_no_split(
                self.df.copy(),
                target_col=None,  # No target for clustering
                raise_on_unseen=True,
            )
            
            # Store encoding metadata
            self.categorical_cols = categorical_cols
            self.numeric_cols = numeric_cols
            self.encoders = encoders
            self.cardinality_warnings = cardinality_warnings
            
            # Warn about high-cardinality columns
            if cardinality_warnings:
                warning_msgs = [
                    f"{col}: {w.message}"
                    for col, w in cardinality_warnings.items()
                ]
                self.parent.page.open(ft.SnackBar(
                    ft.Text(
                        "Cardinality warnings: " + "; ".join(warning_msgs),
                        font_family="SF regular",
                    ),
                    bgcolor="#FF9800"
                ))
            
            # Apply scaling if requested
            if self.parent.scaler_dropdown.value == "standard_scaler":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_encoded)
            elif self.parent.scaler_dropdown.value == "minmax_scaler":
                scaler = MinMaxScaler(feature_range=(0, 1))
                X_scaled = scaler.fit_transform(X_encoded)
            else:
                X_scaled = X_encoded.values
            
            return X_scaled, X_encoded.columns.tolist()
        
        except Exception as e:
            self.parent.page.open(ft.SnackBar(
                ft.Text(f"Data preparation error: {str(e)}", font_family="SF regular")
            ))
            return None
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train DBSCAN model and display evaluation results."""
        try:
            e.control.disabled = True
            self.parent.enable_model_selection()
            disable_navigation_bar(self.parent.page)
            self.parent.page.update()
            
            data = self._prepare_data()
            if data is None:
                enable_navigation_bar(self.parent.page)
                return
            
            X_scaled, feature_cols = data
            
            # Validate hyperparameters
            hyperparams = {
                'eps': float(self.eps_field.value),
                'min_samples': int(self.min_samples_field.value),
                'leaf_size': int(self.leaf_size_field.value),
                'p': float(self.p_field.value.strip()) if self.p_field.value.strip().lower() != "none" else "none"
            }
            
            validation_rules = {
                'eps': {'type': float, 'min': 0.01},
                'min_samples': {'type': int},
                'leaf_size': {'type': int, 'min': 1},
                'p': {'type': float if isinstance(hyperparams['p'], float) else str}
            }
            
            is_valid, error_msg = validate_hyperparameters(hyperparams, 'dbscan', validation_rules)
            if not is_valid:
                self.parent.page.open(ft.SnackBar(
                    ft.Text(f"Hyperparameter error: {error_msg}", font_family="SF regular")
                ))
                return
            
            if self.p_field.value.strip().lower() == "none":
                p_value = None
            else:
                p_value = float(self.p_field.value.strip())
            
            if self.metric_dropdown.value == "minkowski" and p_value is None:
                p_value = 2
            
            # Train DBSCAN
            model = DBSCAN(
                eps=float(self.eps_field.value),
                min_samples=int(self.min_samples_field.value),
                metric=self.metric_dropdown.value,
                algorithm=self.algorithm_dropdown.value,
                leaf_size=int(self.leaf_size_field.value),
                p=p_value
                # p=None if self.metric_dropdown.value != "minkowski" else 2,
            )
            labels = model.fit_predict(X_scaled)
            
            # Calculate metrics
            metrics_dict = calculate_clustering_metrics(X_scaled, labels)
            result_text = format_results_markdown(metrics_dict, task_type="clustering")
            
            # Display results dialog with copy button
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "DBSCAN Clustering Results",
                result_text,
                "DBSCAN"
            )
            self.parent.page.open(evaluation_dialog)
            enable_navigation_bar(self.parent.page)
        
        except Exception as e:
            enable_navigation_bar(self.parent.page)
            self.parent.page.open(ft.SnackBar(
                ft.Text(f"Training failed: {str(e)}", font_family="SF regular")
            ))
        
        finally:
            self.parent.enable_model_selection()
            self.train_btn.disabled = False
            self.parent.page.update()
    
    def _algorithm_on_change(self, e: ft.ControlEvent) -> None:
        algorithm = e.control.value
        if algorithm in ("ball_tree", "kd_tree"):
            self.leaf_size_field.visible = True
        else:
            self.leaf_size_field.visible = False
        self.parent.page.update()
        
    def _reset_p_field_to_none(self, e: ft.ControlEvent) -> None:
        self.p_field.value = "None"
        self.parent.page.update()
        
    def _p_on_click(self, e: ft.ControlEvent) -> None:
        if e.control.value.strip() == "None":
            e.control.value = ""
            self.parent.page.update()
            
    def _p_on_blur(self, e: ft.ControlEvent) -> None:
        if e.control.value.strip() == "":
            e.control.value = "None"
            self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for DBSCAN hyperparameter configuration."""
        
        self.eps_field = ft.TextField(
            label="Epsilon (eps)",
            value="0.5",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.",
        )
        
        self.p_field = ft.TextField(
            label="P",
            value="None",
            expand=1,
            input_filter=ft.InputFilter(r'^$|^[+-]?(\d+(\.\d*)?|\.\d+)$'),
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="The power of the Minkowski metric to be used to calculate distance between points. If None, then p=2 (equivalent to the Euclidean distance). When p=1, this is equivalent to Manhattan distance",
            on_click=self._p_on_click,
            on_blur=self._p_on_blur,
            suffix_icon=ft.IconButton(ft.Icons.RESTART_ALT, on_click=self._reset_p_field_to_none, tooltip="Reset to None")
        )

        self.metric_dropdown = ft.Dropdown(
            label="Distance Metric",
            value="euclidean",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("euclidean", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("manhattan", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("chebyshev", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("cityblock", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("cosine", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("l1", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("l2", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("nan_euclidean", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("braycurtis", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("canberra", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("correlation", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("dice", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("hamming", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("jaccard", text_style=ft.TextStyle(font_family="SF regular")),
                # ft.DropdownOption("mahalanobis", text_style=ft.TextStyle(font_family="SF regular")), #! Must provide either V or VI for Mahalanobis distance
                ft.DropdownOption("minkowski", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("rogerstanimoto", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("russellrao", text_style=ft.TextStyle(font_family="SF regular")),
                # ft.DropdownOption("seuclidean", text_style=ft.TextStyle(font_family="SF regular")), #! __init__() takes exactly 1 positional argument (0 given)
                ft.DropdownOption("sokalmichener", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("sokalsneath", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("sqeuclidean", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("yule", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="The metric to use when calculating distance between instances in a feature array. If metric is a string or callable, it must be one of the options allowed by sklearn.metrics.pairwise_distances for its metric parameter. If metric is 'precomputed', X is assumed to be a distance matrix and must be square. X may be a sparse graph, in which case only 'nonzero' elements may be considered neighbors for DBSCAN.",
        )

        self.min_samples_field = ft.TextField(
            label="Minimum Samples",
            value="5",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. If min_samples is set to a higher value, DBSCAN will find denser clusters, whereas if it is set to a lower value, the found clusters will be more sparse.",
        )
        
        self.algorithm_dropdown = ft.Dropdown(
            label="Algorithm",
            value="auto",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("auto", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("ball_tree", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("kd_tree", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("brute", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors.",
            on_change=self._algorithm_on_change
        )
        
        self.leaf_size_field = ft.TextField(
            label="Leaf size",
            value="30",
            expand=1,
            visible=False,
            input_filter=ft.NumbersOnlyInputFilter(),
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.",
        )

        self.train_btn = ft.FilledButton(
            text="Train and evaluate model",
            icon=ft.Icons.PSYCHOLOGY,
            on_click=self._train_and_evaluate_model,
            expand=1,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        
        return ft.Card(
            expand=2,
            content=ft.Container(
                expand=True,
                margin=ft.margin.all(15),
                alignment=ft.alignment.center,
                content=ft.Column(
                    scroll=ft.ScrollMode.AUTO,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Text("DBSCAN Clustering",
                                       font_family="SF thin",
                                       size=24,
                                       text_align="center",
                                       expand=True)
                            ]
                        ),
                        ft.Divider(),
                        ft.Text("Hyperparameters",
                               font_family="SF regular",
                               weight="bold",
                               size=14),
                        ft.Row([self.eps_field, self.p_field]),
                        ft.Row([self.metric_dropdown, self.min_samples_field]),
                        ft.Row([self.algorithm_dropdown, self.leaf_size_field]),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
