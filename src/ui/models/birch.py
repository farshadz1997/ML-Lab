"""
Birch Clustering Model

Balanced Iterative Reducing and Clustering using Hierarchies.
Efficient memory usage for large datasets. Builds a tree structure
(Clustering Feature Tree) that summarizes cluster information.

Configurable hyperparameters:
- n_clusters: Number of clusters after final clustering step
- threshold: Maximum radius of a subcluster in the leaf node
- branching_factor: Maximum number of CF subclusters in each node
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
from dataclasses import dataclass
from sklearn.cluster import Birch

from utils.model_utils import (
    calculate_clustering_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel


@dataclass
class BirchModel(BaseModel):
    """Birch clustering model."""

    def _prepare_data(self):
        return self._prepare_data_clustering()

    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        is_valid = True
        params = {}

        try:
            n_clusters_val = self.n_clusters_field.value.strip()
            if n_clusters_val == 'None' or n_clusters_val == '':
                params['n_clusters'] = None
            else:
                n_clusters = int(n_clusters_val)
                if n_clusters < 2:
                    n_clusters = 3
                    is_valid = False
                params['n_clusters'] = n_clusters
        except (ValueError, TypeError):
            params['n_clusters'] = 3
            is_valid = False

        try:
            threshold = float(self.threshold_field.value)
            if threshold <= 0:
                threshold = 0.5
                is_valid = False
            params['threshold'] = threshold
        except (ValueError, TypeError):
            params['threshold'] = 0.5
            is_valid = False

        try:
            branching = int(self.branching_factor_field.value)
            if branching < 2:
                branching = 50
                is_valid = False
            params['branching_factor'] = branching
        except (ValueError, TypeError):
            params['branching_factor'] = 50
            is_valid = False

        return params, is_valid

    def _create_model(self) -> Birch:
        hyperparams, params_valid = self._validate_hyperparameters()
        if not params_valid:
            self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)
        model = Birch(
            n_clusters=hyperparams['n_clusters'],
            threshold=hyperparams['threshold'],
            branching_factor=hyperparams['branching_factor'],
        )
        return model
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train Birch model and display evaluation results."""
        try:
            self._disable_training_controls()

            data = self._prepare_data()
            if data is None:
                return

            X_scaled, feature_cols = data

            model = self._create_model()
            model.fit(X_scaled)
            labels = model.labels_

            metrics_dict = calculate_clustering_metrics(X_scaled, labels, inertia=None)
            result_text = format_results_markdown(metrics_dict, task_type="clustering")

            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Birch Clustering Results",
                result_text,
                "Birch"
            )
            self.parent.page.open(evaluation_dialog)

        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self._enable_training_controls()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for Birch hyperparameter configuration."""

        self.n_clusters_field = ft.TextField(
            label="Number of Clusters",
            value="3",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            on_click=self._field_on_click,
            on_blur=self._field_on_blur,
            suffix_icon=ft.IconButton(ft.Icons.RESTART_ALT, on_click=lambda _: self._reset_field_to_none(self.n_clusters_field), tooltip="Reset to None (auto)"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Number of clusters after the final clustering step. None = the final clustering step is not performed and the subclusters are returned as-is.",
        )

        self.threshold_field = ft.TextField(
            label="Threshold",
            value="0.5",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold. A large threshold results in fewer subclusters.",
        )

        self.branching_factor_field = ft.TextField(
            label="Branching Factor",
            value="50",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Maximum number of CF subclusters in each node. If a new samples enters such that the number of subclusters exceed the branching_factor then that node is split into two nodes.",
        )

        self._build_train_button()
        self._build_predict_new_data_button()

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
                                ft.Text("Birch Clustering",
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
                        ft.Row([self.n_clusters_field, self.threshold_field]),
                        self.branching_factor_field,
                        ft.Row([self.train_btn, self.test_data_btn])
                    ]
                )
            )
        )
