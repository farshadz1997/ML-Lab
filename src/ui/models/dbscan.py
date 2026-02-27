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
from typing import Optional, Tuple
import flet as ft
from dataclasses import dataclass
from sklearn.cluster import DBSCAN

from utils.model_utils import (
    validate_hyperparameters,
    calculate_clustering_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel


@dataclass
class DBSCANModel(BaseModel):
    """DBSCAN clustering model."""

    def _prepare_data(self):
        """Prepare data for clustering."""
        return self._prepare_data_clustering()
    
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
                self._show_snackbar(f"Hyperparameter error: {error_msg}", bgcolor=ft.Colors.RED_500)
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
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)
        
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
            on_click=self._field_on_click,
            on_blur=self._field_on_blur,
            suffix_icon=ft.IconButton(ft.Icons.RESTART_ALT, on_click=lambda e: self._reset_field_to_none(self.p_field), tooltip="Reset to None")
        )

        self.metric_dropdown = ft.Dropdown(
            label="Distance Metric",
            value="euclidean",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("euclidean"),
                ft.DropdownOption("manhattan"),
                ft.DropdownOption("chebyshev"),
                ft.DropdownOption("cityblock"),
                ft.DropdownOption("cosine"),
                ft.DropdownOption("l1"),
                ft.DropdownOption("l2"),
                ft.DropdownOption("nan_euclidean"),
                ft.DropdownOption("braycurtis"),
                ft.DropdownOption("canberra"),
                ft.DropdownOption("correlation"),
                ft.DropdownOption("dice"),
                ft.DropdownOption("hamming"),
                ft.DropdownOption("jaccard"),
                # ft.DropdownOption("mahalanobis"), #! Must provide either V or VI for Mahalanobis distance
                ft.DropdownOption("minkowski"),
                ft.DropdownOption("rogerstanimoto"),
                ft.DropdownOption("russellrao"),
                # ft.DropdownOption("seuclidean"), #! __init__() takes exactly 1 positional argument (0 given)
                ft.DropdownOption("sokalmichener"),
                ft.DropdownOption("sokalsneath"),
                ft.DropdownOption("sqeuclidean"),
                ft.DropdownOption("yule"),
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
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("auto"),
                ft.DropdownOption("ball_tree"),
                ft.DropdownOption("kd_tree"),
                ft.DropdownOption("brute"),
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

        self._build_train_button()

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
