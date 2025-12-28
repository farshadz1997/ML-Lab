"""
MiniBatch K-Means Clustering Model

Memory-efficient variant of K-Means using mini-batch sampling.
Faster training with slightly lower quality clusters, good for large datasets.

Configurable hyperparameters:
- n_clusters: Number of clusters
- batch_size: Size of each mini-batch
- n_init: Number of initializations
- random_state: Random seed for reproducibility
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.cluster import MiniBatchKMeans
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
class MiniBatchKMeansModel:
    """MiniBatch K-Means clustering model."""
    
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
        """Train MiniBatch K-Means model and display evaluation results."""
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
                'batch_size': int(self.batch_size_field.value),
                'n_init': int(self.n_init_field.value),
            }
            
            validation_rules = {
                'n_clusters': {'type': int, 'min': 2, 'max': 100},
                'batch_size': {'type': int, 'min': 10, 'max': 1000},
                'n_init': {'type': int, 'min': 1, 'max': 100},
            }
            
            is_valid, error_msg = validate_hyperparameters(hyperparams, 'minibatch_kmeans', validation_rules)
            if not is_valid:
                self.parent.page.open(ft.SnackBar(
                    ft.Text(f"Hyperparameter error: {error_msg}", font_family="SF regular")
                ))
                return
            
            # Train MiniBatch K-Means
            model = MiniBatchKMeans(
                n_clusters=int(self.n_clusters_field.value),
                batch_size=int(self.batch_size_field.value),
                n_init=int(self.n_init_field.value),
                random_state=42,
            )
            labels = model.fit_predict(X_scaled)
            
            # Calculate metrics
            metrics_dict = calculate_clustering_metrics(X_scaled, labels, inertia=model.inertia_)
            result_text = format_results_markdown(metrics_dict, task_type="clustering")
            
            # Display results dialog with copy button
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "MiniBatch K-Means Clustering Results",
                result_text,
                "MiniBatch K-Means"
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
        """Build Flet UI card for MiniBatch K-Means hyperparameter configuration."""
        
        self.n_clusters_field = ft.TextField(
            label="Number of Clusters",
            value="3",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Number of clusters to partition the data into. Range: 2-100",
        )
        
        self.batch_size_field = ft.TextField(
            label="Batch Size",
            value="128",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Number of samples per batch. Higher=more computation per batch. Range: 10-1000",
        )
        
        self.n_init_field = ft.TextField(
            label="Number of Initializations",
            value="10",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Number of times the algorithm runs with different centroid seeds. Range: 1-100",
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
                                ft.Text("MiniBatch K-Means Clustering",
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
                        self.batch_size_field,
                        self.n_init_field,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
