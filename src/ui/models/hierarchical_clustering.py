"""
Hierarchical Agglomerative Clustering Model

Bottom-up clustering that builds dendrograms by repeatedly merging closest clusters.
Produces dendrograms showing cluster hierarchies.

Configurable hyperparameters:
- n_clusters: Number of clusters to form
- linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
"""

from __future__ import annotations
from typing import Optional, Tuple
import flet as ft
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering

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
class HierarchicalClusteringModel(BaseModel):
    """Hierarchical Agglomerative Clustering model."""

    def _prepare_data(self):
        """Prepare data for clustering."""
        return self._prepare_data_clustering()
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train Hierarchical Clustering model and display evaluation results."""
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
                'n_clusters': int(self.n_clusters_field.value) if self.n_clusters_field.value.strip() != "None" else None,
                'distance_threshold': None if self.distance_threshold.value.strip() == "None" else float(self.distance_threshold.value),
            }
            
            validation_rules = {
                'n_clusters': {'type': (int, type(None)), 'min': 2},
                'distance_threshold': {'type': (float, type(None)), 'min': 0},
            }
            
            is_valid, error_msg = validate_hyperparameters(hyperparams, 'hierarchical', validation_rules)
            if not is_valid:
                self._show_snackbar(f"Hyperparameter error: {error_msg}", bgcolor=ft.Colors.RED_500)
                return
            
            # Train Hierarchical Clustering
            compute_full_tree_value = self.compute_full_tree_radio_group.value
            if compute_full_tree_value == "auto":
                pass
            elif compute_full_tree_value == "true":
                compute_full_tree_value = True
            elif compute_full_tree_value == "false":
                compute_full_tree_value = False
            model = AgglomerativeClustering(
                n_clusters=None if self.n_clusters_field.value.strip() == "None" else int(self.n_clusters_field.value),
                linkage=self.linkage_dropdown.value,
                metric=self.metric_dropdown.value,
                compute_full_tree=compute_full_tree_value,
                distance_threshold=None if self.distance_threshold.value.strip() == "None" else float(self.distance_threshold.value),
            )
            labels = model.fit_predict(X_scaled)
            
            # Calculate metrics
            metrics_dict = calculate_clustering_metrics(X_scaled, labels)
            result_text = format_results_markdown(metrics_dict, task_type="clustering")
            
            # Display results dialog with copy button
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Hierarchical Clustering Results",
                result_text,
                "Hierarchical Clustering"
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

    def _linkage_on_change(self, e: ft.ControlEvent) -> None:
        """Adjust metric dropdown based on linkage selection."""
        if e.control.value == "ward":
            self.metric_dropdown.value = "euclidean"
            self.metric_dropdown.disabled = True
        else:
            self.metric_dropdown.disabled = False
        self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for Hierarchical Clustering hyperparameter configuration."""
        
        self.n_clusters_field = ft.TextField(
            label="Number of Clusters",
            value="3",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            suffix_icon=ft.IconButton(ft.Icons.RESTART_ALT, on_click=lambda e: self._reset_field_to_none(self.n_clusters_field), tooltip="Reset to None"),
            on_click=self._field_on_click,
            on_blur=self._field_on_blur,
            tooltip="Number of clusters to form",
        )

        self.linkage_dropdown = ft.Dropdown(
            label="Linkage Criterion",
            value="ward",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("ward"),
                ft.DropdownOption("complete"),
                ft.DropdownOption("average"),
                ft.DropdownOption("single"),
            ],
            on_change=self._linkage_on_change,
            tooltip="Method for computing distances between cluster. ward=minimum variance, complete=maximum distance, average=average distance, single=minimum distance",
        )
        
        self.metric_dropdown = ft.Dropdown(
            label="Metric",
            value="euclidean",
            expand=1,
            disabled=True,  # Initially disabled because default linkage is 'ward'
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("euclidean"),
                ft.DropdownOption("l1"),
                ft.DropdownOption("l2"),
                ft.DropdownOption("manhattan"),
                ft.DropdownOption("cosine"),
                ft.DropdownOption("yule"),
                ft.DropdownOption("chebyshev"),
                ft.DropdownOption("rogerstanimoto"),
                ft.DropdownOption("braycurtis"),
                ft.DropdownOption("sokalmichener"),
                ft.DropdownOption("cityblock"),
                ft.DropdownOption("minkowski"),
                ft.DropdownOption("jaccard"),
                # ft.DropdownOption("precomputed"), #! Matrix must be square
                ft.DropdownOption("canberra"),
                ft.DropdownOption("russellrao"),
                ft.DropdownOption("dice"),
                ft.DropdownOption("correlation"),
                ft.DropdownOption("matching"),
                ft.DropdownOption("hamming"),
                ft.DropdownOption("sokalsneath"),
                ft.DropdownOption("seuclidean"),
                ft.DropdownOption("mahalanobis"),
                ft.DropdownOption("sqeuclidean"),
            ],
            tooltip='Metric used to compute the linkage. Can be "euclidean", "l1", "l2", "manhattan", "cosine", or "precomputed". If linkage is "ward", only "euclidean" is accepted. If "precomputed", a distance matrix is needed as input for the fit method. If connectivity is None, linkage is "single" and affinity is not "precomputed" any valid pairwise distance metric can be assigned.',
        )

        self.distance_threshold = ft.TextField(
            label="Distance Threshold",
            value="None",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            suffix_icon=ft.IconButton(ft.Icons.RESTART_ALT, on_click=lambda e: self._reset_field_to_none(self.distance_threshold), tooltip="Reset to None"),
            on_click=self._field_on_click,
            on_blur=self._field_on_blur,
            tooltip="The linkage distance threshold at or above which clusters will not be merged. If not None, n_clusters must be None and compute_full_tree must be True.",
        )

        self.compute_full_tree_radio_group = ft.RadioGroup(
            value="auto",
            content=ft.Row(
                controls=[
                    ft.Radio(
                        label="Auto",
                        value="auto",
                        label_style=ft.TextStyle(
                            font_family="SF regular"
                        )
                    ),
                    ft.Radio(
                        label="True",
                        value="true",
                        label_style=ft.TextStyle(
                            font_family="SF regular"
                        )
                    ),
                    ft.Radio(
                        label="False",
                        value="false",
                        label_style=ft.TextStyle(
                            font_family="SF regular"
                        )
                    ),
                ]
            )
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
                                ft.Text("Hierarchical Agglomerative Clustering",
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
                        ft.Row([self.n_clusters_field, self.distance_threshold]),
                        self.linkage_dropdown,
                        self.metric_dropdown,
                        ft.Row([ft.Text("Compute full tree:", font_family="SF regular"), self.compute_full_tree_radio_group]),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
