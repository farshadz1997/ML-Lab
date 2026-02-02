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
from core.data_preparation import prepare_data_for_training_no_split

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
    
    def _validate_hyperparameters(self) -> tuple[dict, bool]:
        """
        Validate hyperparameter inputs with defaults for invalid values.
        
        Returns:
            Tuple of (hyperparams_dict, is_valid)
        """
        is_valid = True
        params = {}
        
        # Validate max_iter
        try:
            max_iter_value = int(self.max_iter_field.value)
            if max_iter_value < 0:
                max_iter_value = 200
                is_valid = False
            params['max_iter'] = max_iter_value
        except (ValueError, TypeError):
            params['max_iter'] = 200
            is_valid = False
        
        # Validate convergence_iter
        try:
            convergence_iter_value = int(self.convergence_iter_field.value)
            if convergence_iter_value < 0:
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
                symbol = "+" if self.preference_prefix_btn.icon == ft.Icons.ADD else "-"
                params['preference'] = float(f"{symbol}{preference_input}")
            except (ValueError, TypeError):
                params['preference'] = None
                is_valid = False
        
        return params, is_valid
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train Affinity Propagation model and display evaluation results."""
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
                    ),
                    bgcolor="#FF9800"
                ))
            
            # Create and train model
            model = AffinityPropagation(
                damping=self.damping_field.value / 100,
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
            metrics_dict['damping'] = self.damping_field.value / 100
            
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
            self.parent.enable_model_selection()
            self.train_btn.disabled = False
            self.parent.page.update()
    
    def _preference_field_prefix_icon_on_click(self, e: ft.ControlEvent) -> None:
        if e.control.icon == ft.Icons.ADD:
            e.control.icon = ft.Icons.REMOVE
        else:
            e.control.icon = ft.Icons.ADD
        self.parent.page.update()
        
    def _reset_preference_field_to_auto(self, e: ft.ControlEvent) -> None:
        self.preference_field.value = "auto"
        self.parent.page.update()
        
    def _preference_field_on_click(self, e: ft.ControlEvent) -> None:
        if e.control.value.strip() == "auto":
            e.control.value = ""
            self.parent.page.update()
            
    def _preference_field_on_blur(self, e: ft.ControlEvent) -> None:
        if e.control.value.strip() == "":
            e.control.value = "auto"
            self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for Affinity Propagation hyperparameter configuration."""
        
        self.damping_field = ft.Slider(
            label="0.{value}",
            value=50,
            min=50,
            max=99,
            divisions=49,
            expand=4,
            tooltip="Damping factor for convergence. Prevents oscillations. Range: 0.5 to 0.99. Higher values = slower convergence but more stable",
        )
        
        self.max_iter_field = ft.TextField(
            label="Max Iterations",
            value="200",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Maximum number of iterations.",
        )
        
        self.convergence_iter_field = ft.TextField(
            label="Convergence Iterations",
            value="15",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Number of iterations with no change in the number of estimated clusters that stops the convergence.",
        )
        
        self.preference_prefix_btn = ft.IconButton(ft.Icons.ADD, on_click=self._preference_field_prefix_icon_on_click, scale=0.8)
        self.preference_field = ft.TextField(
            label="Preference (auto=default)",
            value="auto",
            expand=1,
            dense=True,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            shift_enter=False,
            multiline=False,
            tooltip="Preference for each point to be exemplar. 'auto' uses median of distances. Or specify numeric value",
            input_filter=ft.NumbersOnlyInputFilter(),
            prefix=self.preference_prefix_btn,
            suffix=ft.IconButton(ft.Icons.RESTART_ALT, tooltip="Reset to 'auto'", on_click=self._reset_preference_field_to_auto, scale=0.8),
            on_click=self._preference_field_on_click,
            on_blur=self._preference_field_on_blur
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
                        ft.Row([ft.Text("Damping", font_family="SF regular", expand=1), self.damping_field]),
                        ft.Row([self.convergence_iter_field, self.max_iter_field]),
                        self.preference_field,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
