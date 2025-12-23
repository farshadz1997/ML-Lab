"""
Affinity Propagation Clustering Model

Message-passing algorithm that discovers exemplars (representative points)
in the data. Automatically determines number of clusters based on similarities.

Configurable hyperparameters:
- damping: Damping factor for convergence [0.5, 1.0)
- max_iter: Maximum iterations for convergence
- convergence_iter: Number of iterations with no change to declare convergence
- preference: Preference for each point to be exemplar (similarity base)
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
import numpy as np
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

from utils.model_utils import (
    calculate_clustering_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)

if TYPE_CHECKING:
    from ..model_factory import ModelFactory


@dataclass
class AffinityPropagationModel:
    """Affinity Propagation clustering model."""
    
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
    
    def _validate_hyperparameters(self) -> tuple[dict, bool]:
        """
        Validate hyperparameter inputs with defaults for invalid values.
        
        Returns:
            Tuple of (hyperparams_dict, is_valid)
        """
        is_valid = True
        params = {}
        
        # Validate damping
        try:
            damping_value = float(self.damping_field.value)
            if damping_value <= 0.5 or damping_value >= 1.0:
                damping_value = 0.9
                is_valid = False
            params['damping'] = damping_value
        except (ValueError, TypeError):
            params['damping'] = 0.9
            is_valid = False
        
        # Validate max_iter
        try:
            max_iter_value = int(self.max_iter_field.value)
            if max_iter_value < 100 or max_iter_value > 10000:
                max_iter_value = 200
                is_valid = False
            params['max_iter'] = max_iter_value
        except (ValueError, TypeError):
            params['max_iter'] = 200
            is_valid = False
        
        # Validate convergence_iter
        try:
            convergence_iter_value = int(self.convergence_iter_field.value)
            if convergence_iter_value < 10 or convergence_iter_value > 1000:
                convergence_iter_value = 15
                is_valid = False
            params['convergence_iter'] = convergence_iter_value
        except (ValueError, TypeError):
            params['convergence_iter'] = 15
            is_valid = False
        
        # Preference (use median of distances if not specified)
        preference_input = self.preference_field.value.strip()
        if preference_input == "" or preference_input.lower() == "auto":
            params['preference'] = None  # Will use default
        else:
            try:
                params['preference'] = float(preference_input)
            except (ValueError, TypeError):
                params['preference'] = None
                is_valid = False
        
        return params, is_valid
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train Affinity Propagation model and display evaluation results."""
        try:
            e.control.disabled = True
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
            
            # Create and train model
            model = AffinityPropagation(
                damping=hyperparams['damping'],
                max_iter=hyperparams['max_iter'],
                convergence_iter=hyperparams['convergence_iter'],
                preference=hyperparams['preference'],
                random_state=42,
            )
            cluster_labels = model.fit_predict(X_scaled)
            n_clusters = len(set(cluster_labels))
            n_exemplars = len(model.cluster_centers_indices_)
            
            # Calculate metrics using centralized utility
            metrics_dict = calculate_clustering_metrics(X_scaled, cluster_labels)
            
            # Add Affinity Propagation specific metrics
            metrics_dict['n_clusters'] = n_clusters
            metrics_dict['n_exemplars'] = n_exemplars
            metrics_dict['damping'] = hyperparams['damping']
            
            result_text = format_results_markdown(metrics_dict, task_type="clustering")
            
            # Display results dialog
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Affinity Propagation Clustering Results",
                result_text,
                "Affinity Propagation"
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
        """Build Flet UI card for Affinity Propagation hyperparameter configuration."""
        
        self.damping_field = ft.TextField(
            label="Damping Factor",
            value="0.9",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Damping factor for convergence. Prevents oscillations. Range: 0.5 to 0.99. Higher values = slower convergence but more stable",
        )
        
        self.max_iter_field = ft.TextField(
            label="Max Iterations",
            value="200",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Maximum iterations for algorithm. Range: 100 to 10000",
        )
        
        self.convergence_iter_field = ft.TextField(
            label="Convergence Iterations",
            value="15",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Iterations with no change needed to declare convergence. Range: 10 to 1000",
        )
        
        self.preference_field = ft.TextField(
            label="Preference (auto=default)",
            value="auto",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Preference for each point to be exemplar. 'auto' uses median of distances. Or specify numeric value",
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
                                ft.Text("Affinity Propagation",
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
                        ft.Row([self.damping_field, self.max_iter_field]),
                        self.convergence_iter_field,
                        self.preference_field,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
