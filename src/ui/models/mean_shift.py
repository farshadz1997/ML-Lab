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
from typing import Optional, Tuple
import flet as ft
from dataclasses import dataclass
from sklearn.cluster import MeanShift, estimate_bandwidth

from utils.model_utils import (
    calculate_clustering_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel


@dataclass
class MeanShiftModel(BaseModel):
    """Mean Shift clustering model."""

    def _prepare_data(self):
        """Prepare data for clustering."""
        return self._prepare_data_clustering()
    
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
            if self.bandwidth_field.value.strip() == "None":
                bandwidth_value = None
            else:
                bandwidth_value = float(self.bandwidth_field.value)
                if bandwidth_value <= 0:
                    bandwidth_value = None  # Will use estimate
                    is_valid = False
            params['bandwidth'] = bandwidth_value
        except (ValueError, TypeError):
            params['bandwidth'] = None
            is_valid = False
        
        # Validate cluster_all
        params['cluster_all'] = self.cluster_all_toggle.value
        # Validate bin_seeding
        params['bin_seeding'] = self.bin_seeding_toggle.value
        
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
                self._show_snackbar("Some hyperparameters were invalid. Using defaults.", bgcolor=ft.Colors.AMBER_ACCENT_200)
            
            # Estimate bandwidth if not provided
            bandwidth = hyperparams['bandwidth']
            if bandwidth is None:
                bandwidth = estimate_bandwidth(X_scaled, quantile=0.3)
            
            # Create and train model
            model = MeanShift(
                bandwidth=bandwidth,
                cluster_all=hyperparams['cluster_all'],
                max_iter=hyperparams['max_iter'],
                bin_seeding=hyperparams['bin_seeding'],
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
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)
        
        finally:
            self.parent.enable_model_selection()
            self.train_btn.disabled = False
            self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for Mean Shift hyperparameter configuration."""
        
        self.bandwidth_field = ft.TextField(
            label="Bandwidth",
            value="None",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            suffix_icon=ft.IconButton(ft.Icons.RESTART_ALT, on_click=lambda e: self._reset_field_to_none(self.bandwidth_field), tooltip="Reset to None"),
            on_click=self._field_on_click,
            on_blur=self._field_on_blur,
            tooltip="Bandwidth used in the flat kernel. If not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth. Range: (0.0, inf) or None.",
        )
        
        self.max_iter_field = ft.TextField(
            label="Max Iterations",
            value="300",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Maximum number of iterations, per seed point before the clustering operation terminates (for that seed point), if has not converged yet.",
        )

        self.cluster_all_toggle = ft.Checkbox(
            label="Cluster All Points",
            value=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="If true, then all points are clustered, even those orphans that are not within any kernel. Orphans are assigned to the nearest kernel. If false, then orphans are given cluster label -1.",
        )

        self.bin_seeding_toggle = ft.Checkbox(
            label="Use Bin Seeding",
            value=False,
            label_style=ft.TextStyle(font_family="SF regular"),
            label_position=ft.LabelPosition.LEFT,
            tooltip="If true, initial kernel locations are not locations of all points, but rather the location of the discretized version of points, where points are binned onto a grid whose coarseness corresponds to the bandwidth. Setting this option to True will speed up the algorithm because fewer seeds will be initialized. The default value is False. Ignored if seeds argument is not None.",
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
                        ft.Row([self.cluster_all_toggle, self.bin_seeding_toggle], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
