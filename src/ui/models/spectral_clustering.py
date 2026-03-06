"""
Spectral Clustering Model

Applies clustering to a projection of the normalized Laplacian
of the affinity matrix. Good for finding clusters with non-convex
shapes that other methods might struggle with.

Configurable hyperparameters:
- n_clusters: Number of clusters
- affinity: How to construct affinity matrix
- n_neighbors: Number of neighbors for nearest_neighbors affinity
- assign_labels: Strategy for assigning labels
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
from dataclasses import dataclass
from sklearn.cluster import SpectralClustering

from utils.model_utils import (
    calculate_clustering_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel


@dataclass
class SpectralClusteringModel(BaseModel):
    """Spectral Clustering model."""

    def _prepare_data(self):
        return self._prepare_data_clustering()

    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        is_valid = True
        params = {}

        try:
            n_clusters = int(self.n_clusters_field.value)
            if n_clusters < 2:
                n_clusters = 3
                is_valid = False
            params['n_clusters'] = n_clusters
        except (ValueError, TypeError):
            params['n_clusters'] = 3
            is_valid = False

        try:
            n_neighbors = int(self.n_neighbors_field.value)
            if n_neighbors < 1:
                n_neighbors = 10
                is_valid = False
            params['n_neighbors'] = n_neighbors
        except (ValueError, TypeError):
            params['n_neighbors'] = 10
            is_valid = False

        try:
            gamma = float(self.gamma_field.value)
            if gamma <= 0:
                gamma = 1.0
                is_valid = False
            params['gamma'] = gamma
        except (ValueError, TypeError):
            params['gamma'] = 1.0
            is_valid = False

        return params, is_valid

    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train Spectral Clustering model and display evaluation results."""
        try:
            self._disable_training_controls()

            data = self._prepare_data()
            if data is None:
                return

            X_scaled, feature_cols = data

            hyperparams, params_valid = self._validate_hyperparameters()

            if not params_valid:
                self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)

            affinity = self.affinity_dropdown.value
            model_params = dict(
                n_clusters=hyperparams['n_clusters'],
                affinity=affinity,
                assign_labels=self.assign_labels_dropdown.value,
                random_state=42,
            )
            if affinity == 'nearest_neighbors':
                model_params['n_neighbors'] = hyperparams['n_neighbors']
            if affinity == 'rbf':
                model_params['gamma'] = hyperparams['gamma']

            model = SpectralClustering(**model_params)
            labels = model.fit_predict(X_scaled)

            metrics_dict = calculate_clustering_metrics(X_scaled, labels, inertia=None)
            result_text = format_results_markdown(metrics_dict, task_type="clustering")

            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Spectral Clustering Results",
                result_text,
                "Spectral Clustering"
            )
            self.parent.page.open(evaluation_dialog)

        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self._enable_training_controls()

    def _affinity_on_change(self, e: ft.ControlEvent) -> None:
        affinity = self.affinity_dropdown.value
        self.n_neighbors_field.visible = affinity == 'nearest_neighbors'
        self.gamma_field.visible = affinity == 'rbf'
        self.parent.page.update()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for Spectral Clustering hyperparameter configuration."""

        self.n_clusters_field = ft.TextField(
            label="Number of Clusters",
            value="3",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The dimension of the projection subspace.",
        )

        self.affinity_dropdown = ft.Dropdown(
            label="Affinity",
            value="rbf",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("rbf"),
                ft.DropdownOption("nearest_neighbors"),
                ft.DropdownOption("poly"),
                ft.DropdownOption("sigmoid"),
                ft.DropdownOption("cosine"),
            ],
            on_change=self._affinity_on_change,
            tooltip="How to construct the affinity matrix. 'rbf' uses Gaussian kernel, 'nearest_neighbors' uses k-nearest neighbors.",
        )

        self.n_neighbors_field = ft.TextField(
            label="N Neighbors",
            value="10",
            expand=1,
            visible=False,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Number of neighbors for nearest_neighbors affinity. Only used when affinity='nearest_neighbors'.",
        )

        self.gamma_field = ft.TextField(
            label="Gamma",
            value="1.0",
            expand=1,
            visible=True,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Kernel coefficient for 'rbf', 'poly' and 'sigmoid' affinities.",
        )

        self.assign_labels_dropdown = ft.Dropdown(
            label="Assign Labels",
            value="kmeans",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("kmeans"),
                ft.DropdownOption("discretize"),
                ft.DropdownOption("cluster_qr"),
            ],
            tooltip="The strategy for assigning labels in the embedding space.",
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
                                ft.Text("Spectral Clustering",
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
                        ft.Row([self.n_clusters_field, self.affinity_dropdown]),
                        ft.Row([self.n_neighbors_field, self.gamma_field]),
                        self.assign_labels_dropdown,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
