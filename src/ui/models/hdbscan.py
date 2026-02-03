"""
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)

Hierarchical extension of DBSCAN that's more robust to density variations.
Better handling of clusters with varying densities.

Configurable hyperparameters:
- min_cluster_size: Minimum size of a cluster
- min_samples: Minimum samples to consider a point core
- cluster_selection_epsilon: Distance threshold for final cluster formation
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
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
class HDBSCANModel:
    """HDBSCAN clustering model."""
    
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
            Tuple of (X_scaled, feature_cols) or None if error
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
        """Train HDBSCAN model and display evaluation results."""
        try:
            e.control.disabled = True
            self.parent.disable_model_selection()
            disable_navigation_bar(self.parent.page)
            self.parent.page.update()
            
            data = self._prepare_data()
            if data is None:
                enable_navigation_bar(self.parent.page)
                return
            
            X_scaled, feature_cols = data
            
            # Validate hyperparameters
            hyperparams = {
                'min_cluster_size': int(self.min_cluster_size_field.value),
                'min_samples': int(self.min_samples_field.value) if self.min_samples_field.value.strip() != "None" else None,
                'leaf_size': int(self.leaf_size_field.value),
            }
            
            validation_rules = {
                'min_cluster_size': {'type': int, 'min': 2},
                'min_samples': {'type': (int, type(None)), 'min': 1},
                'leaf_size': {'type': int, 'min': 1},
            }
            
            is_valid, error_msg = validate_hyperparameters(hyperparams, 'hdbscan', validation_rules)
            if not is_valid:
                self.parent.page.open(ft.SnackBar(
                    ft.Text(f"Hyperparameter error: {error_msg}", font_family="SF regular")
                ))
                return
            
            model = HDBSCAN(
                min_cluster_size=int(self.min_cluster_size_field.value),
                min_samples=int(self.min_samples_field.value) if self.min_samples_field.value.strip() != "None" else None,
                metric=self.metric_dropdown.value,
                algorithm=self.algorithm_dropdown.value,
                leaf_size=int(self.leaf_size_field.value),
            )
            labels = model.fit_predict(X_scaled)
            
            # Calculate metrics
            metrics_dict = calculate_clustering_metrics(X_scaled, labels)
            result_text = format_results_markdown(metrics_dict, task_type="clustering")
            
            # Display results dialog with copy button
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "HDBSCAN Clustering Results",
                result_text,
                "HDBSCAN"
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
    
    def _reset_field_to_none(self, field: ft.TextField) -> None:
        field.value = "None"
        self.parent.page.update()
        
    def _field_on_click(self, e: ft.ControlEvent) -> None:
        if e.control.value.strip() == "None":
            e.control.value = ""
            self.parent.page.update()
            
    def _field_on_blur(self, e: ft.ControlEvent) -> None:
        if e.control.value.strip() == "":
            e.control.value = "None"
            self.parent.page.update()

    def _algorithm_on_change(self, e: ft.ControlEvent) -> None:
        algorithm = e.control.value
        if algorithm in ("ball_tree", "kd_tree"):
            self.leaf_size_field.visible = True
        else:
            self.leaf_size_field.visible = False
        self.parent.page.update()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for HDBSCAN hyperparameter configuration."""
        
        self.min_cluster_size_field = ft.TextField(
            label="Minimum Cluster Size",
            value="5",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The minimum number of samples in a group for that group to be considered a cluster; groupings smaller than this size will be left as noise.",
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
                # ft.DropdownOption("minkowski", text_style=ft.TextStyle(font_family="SF regular")), #! '<' not supported between instances of 'NoneType' and 'int'
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
            value="None",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            suffix_icon=ft.IconButton(ft.Icons.RESTART_ALT, on_click=lambda e: self._reset_field_to_none(self.min_samples_field), tooltip="Reset to None"),
            on_click=self._field_on_click,
            on_blur=self._field_on_blur,
            tooltip="The parameter k used to calculate the distance between a point x_p and its k-th nearest neighbor. When None, defaults to min_cluster_size.",
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
            tooltip='Exactly which algorithm to use for computing core distances; By default this is set to "auto" which attempts to use a KDTree tree if possible, otherwise it uses a BallTree tree. Both "kd_tree" and "ball_tree" algorithms use the NearestNeighbors estimator. If the X passed during fit is sparse or metric is invalid for both KDTree and BallTree, then it resolves to use the "brute" algorithm.',
            on_change=self._algorithm_on_change
        )

        self.leaf_size_field = ft.TextField(
            label="Leaf size",
            value="40",
            expand=1,
            visible=False,
            input_filter=ft.NumbersOnlyInputFilter(),
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip='Leaf size for trees responsible for fast nearest neighbour queries when a KDTree or a BallTree are used as core-distance algorithms. A large dataset size and small leaf_size may induce excessive memory usage. If you are running out of memory consider increasing the leaf_size parameter. Ignored for algorithm="brute".',
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
                                ft.Text(
                                    "HDBSCAN Clustering",
                                    font_family="SF thin",
                                    size=24,
                                    text_align="center",
                                    expand=True
                                )
                            ]
                        ),
                        ft.Divider(),
                        ft.Text("Hyperparameters",
                               font_family="SF regular",
                               weight="bold",
                               size=14),
                        self.min_cluster_size_field,
                        ft.Row([self.metric_dropdown, self.min_samples_field]),
                        ft.Row([self.algorithm_dropdown, self.leaf_size_field]),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
