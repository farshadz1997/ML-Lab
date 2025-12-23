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
        """Train DBSCAN model and display evaluation results."""
        try:
            e.control.disabled = True
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
            }
            
            validation_rules = {
                'eps': {'type': float, 'min': 0.01, 'max': 100.0},
                'min_samples': {'type': int, 'min': 1, 'max': 100},
            }
            
            is_valid, error_msg = validate_hyperparameters(hyperparams, 'dbscan', validation_rules)
            if not is_valid:
                self.parent.page.open(ft.SnackBar(
                    ft.Text(f"Hyperparameter error: {error_msg}", font_family="SF regular")
                ))
                return
            
            # Train DBSCAN
            model = DBSCAN(
                eps=float(self.eps_field.value),
                min_samples=int(self.min_samples_field.value),
                metric=self.metric_dropdown.value,
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
            self.train_btn.disabled = False
            self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for DBSCAN hyperparameter configuration."""
        
        self.eps_field = ft.TextField(
            label="Epsilon (eps)",
            value="0.5",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Maximum distance between samples in neighborhood. Smaller=tighter clusters. Range: 0.01-100.0",
        )
        
        self.min_samples_field = ft.TextField(
            label="Minimum Samples",
            value="5",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Minimum samples to form a core point. Higher=stricter clustering. Range: 1-100",
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
                ft.DropdownOption("minkowski", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="Distance metric for nearest neighbor search. euclidean=straight-line distance, manhattan=grid distance",
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
                        self.eps_field,
                        self.min_samples_field,
                        self.metric_dropdown,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
