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
        Prepare data for clustering (no train/test split, use full dataset).
        
        Returns:
            Tuple[X_scaled, feature_cols] or None if error
        """
        try:
            df = self.df.copy()
            X = df.copy()
            
            # Apply scaling to full dataset
            if self.parent.scaler_dropdown.value == "standard_scaler":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            elif self.parent.scaler_dropdown.value == "minmax_scaler":
                scaler = MinMaxScaler(feature_range=(0, 1))
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.values
            
            feature_cols = X.columns.tolist()
            return X_scaled, feature_cols
        
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
                'n_clusters': int(self.n_clusters_field.value),
            }
            
            validation_rules = {
                'n_clusters': {'type': int, 'min': 2, 'max': 100},
            }
            
            is_valid, error_msg = validate_hyperparameters(hyperparams, 'hierarchical', validation_rules)
            if not is_valid:
                self.parent.page.open(ft.SnackBar(
                    ft.Text(f"Hyperparameter error: {error_msg}", font_family="SF regular")
                ))
                return
            
            # Train Hierarchical Clustering
            model = AgglomerativeClustering(
                n_clusters=int(self.n_clusters_field.value),
                linkage=self.linkage_dropdown.value,
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
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for Hierarchical Clustering hyperparameter configuration."""
        
        self.n_clusters_field = ft.TextField(
            label="Number of Clusters",
            value="3",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Number of clusters to form. Range: 2-100",
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
            tooltip="Method for computing distances between cluster. ward=minimum variance, complete=maximum distance, average=average distance, single=minimum distance",
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
                        self.n_clusters_field,
                        self.linkage_dropdown,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
