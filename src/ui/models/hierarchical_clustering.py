"""
Hierarchical Agglomerative Clustering Model

Bottom-up clustering that builds dendrograms by repeatedly merging closest clusters.
Produces dendrograms showing cluster hierarchies.

Configurable hyperparameters:
- n_clusters: Number of clusters to form
- linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering
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
class HierarchicalClusteringModel:
    """Hierarchical Agglomerative Clustering model."""
    
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
                self.parent.page.open(ft.SnackBar(
                    ft.Text(f"Hyperparameter error: {error_msg}", font_family="SF regular")
                ))
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
            options=[
                ft.DropdownOption("ward", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("complete", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("average", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("single", text_style=ft.TextStyle(font_family="SF regular")),
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
            options=[
                ft.DropdownOption("euclidean", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("l1", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("l2", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("manhattan", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("cosine", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("yule", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("chebyshev", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("rogerstanimoto", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("braycurtis", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("sokalmichener", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("cityblock", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("minkowski", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("jaccard", text_style=ft.TextStyle(font_family="SF regular")),
                # ft.DropdownOption("precomputed", text_style=ft.TextStyle(font_family="SF regular")), #! Matrix must be square
                ft.DropdownOption("canberra", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("russellrao", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("dice", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("correlation", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("matching", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("hamming", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("sokalsneath", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("seuclidean", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("mahalanobis", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("sqeuclidean", text_style=ft.TextStyle(font_family="SF regular")),
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
