"""
Gaussian Mixture Model (GMM) Clustering

Probabilistic clustering model that assumes data points come from a mixture
of Gaussian distributions. Each cluster has soft membership probabilities.

Configurable hyperparameters:
- n_components: Number of mixture components (clusters)
- covariance_type: Type of covariance matrix ('full', 'tied', 'diag', 'spherical')
- max_iter: Maximum iterations for EM algorithm
- random_state: Random seed for reproducibility
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer

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
class GaussianMixtureModel:
    """Gaussian Mixture clustering model."""
    
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
        
        # Validate n_components
        try:
            n_components_value = int(self.n_components_field.value)
            if n_components_value < 2 or n_components_value > 100:
                n_components_value = 3
                is_valid = False
            params['n_components'] = n_components_value
        except (ValueError, TypeError):
            params['n_components'] = 3
            is_valid = False
        
        # Validate covariance_type
        valid_covariance_types = ['full', 'tied', 'diag', 'spherical']
        covariance_type_value = self.covariance_type_dropdown.value
        if covariance_type_value not in valid_covariance_types:
            covariance_type_value = 'full'
            is_valid = False
        params['covariance_type'] = covariance_type_value
        
        # Validate max_iter
        try:
            max_iter_value = int(self.max_iter_field.value)
            if max_iter_value < 1 or max_iter_value > 1000:
                max_iter_value = 100
                is_valid = False
            params['max_iter'] = max_iter_value
        except (ValueError, TypeError):
            params['max_iter'] = 100
            is_valid = False
        
        return params, is_valid
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train Gaussian Mixture model and display evaluation results."""
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
            model = GaussianMixture(
                n_components=hyperparams['n_components'],
                covariance_type=hyperparams['covariance_type'],
                max_iter=hyperparams['max_iter'],
                random_state=42,
            )
            cluster_labels = model.fit_predict(X_scaled)
            
            # Calculate metrics using centralized utility
            metrics_dict = calculate_clustering_metrics(X_scaled, cluster_labels)
            
            # Add GMM-specific metrics
            metrics_dict['bic_score'] = float(model.bic(X_scaled))
            metrics_dict['aic_score'] = float(model.aic(X_scaled))
            metrics_dict['n_components'] = hyperparams['n_components']
            
            result_text = format_results_markdown(metrics_dict, task_type="clustering")
            
            # Display results dialog
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Gaussian Mixture Clustering Results",
                result_text,
                "Gaussian Mixture"
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
        """Build Flet UI card for Gaussian Mixture hyperparameter configuration."""
        
        self.n_components_field = ft.TextField(
            label="Number of Components",
            value="3",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Number of Gaussian components (clusters). Range: 2 to 100",
        )
        
        self.covariance_type_dropdown = ft.Dropdown(
            label="Covariance Type",
            value="full",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("full", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("tied", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("diag", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("spherical", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="'full'=most flexible, 'tied'=same covariance, 'diag'=diagonal, 'spherical'=fastest",
        )
        
        self.max_iter_field = ft.TextField(
            label="Max Iterations",
            value="100",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Maximum iterations for EM algorithm. Range: 1 to 1000",
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
                                ft.Text("Gaussian Mixture",
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
                        self.n_components_field,
                        self.covariance_type_dropdown,
                        self.max_iter_field,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
