"""
Mean Shift Clustering Model

Non-parametric clustering algorithm that discovers modes in the probability
density function. Automatically determines number of clusters.

Configurable hyperparameters:
- bandwidth: Size of the kernel. Estimated if not provided.
- cluster_all: Whether to label all points or only cluster centers
- max_iter: Maximum iterations for algorithm convergence
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.model_utils import (
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
class MeanShiftModel:
    """Mean Shift clustering model."""
    
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
                        color=ft.Colors.ORANGE
                    )
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
    
    def _validate_hyperparameters(self) -> tuple[dict, bool]:
        """
        Validate hyperparameter inputs with defaults for invalid values.
        
        Returns:
            Tuple of (hyperparams_dict, is_valid)
        """
        is_valid = True
        params = {}
        
        # Validate bandwidth
        try:
            bandwidth_value = float(self.bandwidth_field.value)
            if bandwidth_value <= 0 or bandwidth_value > 10.0:
                bandwidth_value = None  # Will use estimate
                is_valid = False
            params['bandwidth'] = bandwidth_value
        except (ValueError, TypeError):
            params['bandwidth'] = None
            is_valid = False
        
        # Validate cluster_all
        cluster_all_value = self.cluster_all_toggle.value
        params['cluster_all'] = cluster_all_value
        
        # Validate max_iter
        try:
            max_iter_value = int(self.max_iter_field.value)
            if max_iter_value < 100 or max_iter_value > 10000:
                max_iter_value = 300
                is_valid = False
            params['max_iter'] = max_iter_value
        except (ValueError, TypeError):
            params['max_iter'] = 300
            is_valid = False
        
        return params, is_valid
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train Mean Shift model and display evaluation results."""
        try:
            e.control.disabled = True
            self.parent.disable_model_selection()
            disable_navigation_bar(self.parent.page)
            self.parent.page.update()
            
            # Prepare data
            data = self._prepare_data()
            if data is None:
                enable_navigation_bar(self.parent.page)
                return
            
            X_scaled, feature_cols = data
            
            # Validate and get hyperparameters
            hyperparams, params_valid = self._validate_hyperparameters()
            
            if not params_valid:
                self.parent.page.open(ft.SnackBar(
                    ft.Text(
                        "Some hyperparameters were invalid. Using defaults.",
                        font_family="SF regular",
                        color=ft.Colors.ORANGE
                    )
                ))
            
            # Estimate bandwidth if not provided
            bandwidth = hyperparams['bandwidth']
            if bandwidth is None:
                bandwidth = estimate_bandwidth(X_scaled, quantile=0.3)
            
            # Create and train model
            model = MeanShift(
                bandwidth=bandwidth,
                cluster_all=hyperparams['cluster_all'],
                max_iter=hyperparams['max_iter'],
            )
            cluster_labels = model.fit_predict(X_scaled)
            n_clusters = len(set(cluster_labels))
            
            # Calculate metrics using centralized utility
            metrics_dict = calculate_clustering_metrics(X_scaled, cluster_labels)
            
            # Add Mean Shift specific metrics
            metrics_dict['n_clusters'] = n_clusters
            metrics_dict['bandwidth_used'] = float(bandwidth)
            
            result_text = format_results_markdown(metrics_dict, task_type="clustering")
            
            # Display results dialog
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Mean Shift Clustering Results",
                result_text,
                "Mean Shift"
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
        """Build Flet UI card for Mean Shift hyperparameter configuration."""
        
        self.bandwidth_field = ft.TextField(
            label="Bandwidth (0=auto-estimate)",
            value="0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Kernel bandwidth. 0 = auto-estimate from data. Higher values = larger kernels = fewer clusters. Range: 0.01 to 10.0",
        )
        
        self.cluster_all_toggle = ft.Checkbox(
            label="Cluster All Points",
            value=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="If true, all points are assigned to nearest cluster center. If false, only cluster centers are kept",
        )
        
        self.max_iter_field = ft.TextField(
            label="Max Iterations",
            value="300",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Maximum iterations for convergence. Range: 100 to 10000",
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
                                ft.Text("Mean Shift",
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
                        self.bandwidth_field,
                        self.max_iter_field,
                        self.cluster_all_toggle,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
