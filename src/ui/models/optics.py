"""
OPTICS Clustering Model

Ordering Points To Identify the Clustering Structure.
Similar to DBSCAN but does not require a fixed eps parameter.
Instead, it produces a reachability plot from which clusters
can be extracted at varying density levels.

Configurable hyperparameters:
- min_samples: Minimum samples for core point
- max_eps: Maximum eps distance
- metric: Distance metric
- cluster_method: Method to extract clusters
- xi: Steepness for xi method
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import OPTICS

from utils.model_utils import (
    calculate_clustering_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel


@dataclass
class OPTICSModel(BaseModel):
    """OPTICS clustering model."""

    def _prepare_data(self):
        return self._prepare_data_clustering()

    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        is_valid = True
        params = {}

        try:
            min_samples = int(self.min_samples_field.value)
            if min_samples < 2:
                min_samples = 5
                is_valid = False
            params['min_samples'] = min_samples
        except (ValueError, TypeError):
            params['min_samples'] = 5
            is_valid = False

        try:
            max_eps_val = self.max_eps_field.value.strip()
            if max_eps_val == 'inf' or max_eps_val == '':
                params['max_eps'] = np.inf
            else:
                max_eps = float(max_eps_val)
                if max_eps <= 0:
                    max_eps = np.inf
                    is_valid = False
                params['max_eps'] = max_eps
        except (ValueError, TypeError):
            params['max_eps'] = np.inf
            is_valid = False

        try:
            xi = float(self.xi_field.value)
            if xi <= 0 or xi >= 1:
                xi = 0.05
                is_valid = False
            params['xi'] = xi
        except (ValueError, TypeError):
            params['xi'] = 0.05
            is_valid = False

        return params, is_valid

    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train OPTICS model and display evaluation results."""
        try:
            self._disable_training_controls()

            data = self._prepare_data()
            if data is None:
                return

            X_scaled, feature_cols = data

            hyperparams, params_valid = self._validate_hyperparameters()

            if not params_valid:
                self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)

            model = OPTICS(
                min_samples=hyperparams['min_samples'],
                max_eps=hyperparams['max_eps'],
                metric=self.metric_dropdown.value,
                cluster_method=self.cluster_method_dropdown.value,
                xi=hyperparams['xi'],
            )
            model.fit(X_scaled)
            labels = model.labels_

            metrics_dict = calculate_clustering_metrics(X_scaled, labels, inertia=None)
            result_text = format_results_markdown(metrics_dict, task_type="clustering")

            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "OPTICS Clustering Results",
                result_text,
                "OPTICS"
            )
            self.parent.page.open(evaluation_dialog)

        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self._enable_training_controls()

    def _reset_max_eps_to_inf(self, e: ft.ControlEvent) -> None:
        self.max_eps_field.value = "inf"
        self.parent.page.update()

    def _max_eps_on_click(self, e: ft.ControlEvent) -> None:
        if e.control.value.strip() == "inf":
            e.control.value = ""
            self.parent.page.update()

    def _max_eps_on_blur(self, e: ft.ControlEvent) -> None:
        if e.control.value.strip() == "":
            e.control.value = "inf"
            self.parent.page.update()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for OPTICS hyperparameter configuration."""

        self.min_samples_field = ft.TextField(
            label="Min Samples",
            value="5",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The number of samples in a neighborhood for a point to be considered as a core point.",
        )

        self.max_eps_field = ft.TextField(
            label="Max Eps",
            value="inf",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            on_click=self._max_eps_on_click,
            on_blur=self._max_eps_on_blur,
            suffix_icon=ft.TextButton('Inf', on_click=self._reset_max_eps_to_inf),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="The maximum distance between two samples for one to be considered in the neighborhood. Default inf uses all points.",
        )

        self.xi_field = ft.TextField(
            label="Xi",
            value="0.05",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Determines the minimum steepness on the reachability plot that constitutes a cluster boundary. Range: (0, 1).",
        )

        self.metric_dropdown = ft.Dropdown(
            label="Metric",
            value="minkowski",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("minkowski"),
                ft.DropdownOption("euclidean"),
                ft.DropdownOption("manhattan"),
                ft.DropdownOption("cosine"),
                ft.DropdownOption("chebyshev"),
            ],
            tooltip="Metric to use for distance computation.",
        )

        self.cluster_method_dropdown = ft.Dropdown(
            label="Cluster Method",
            value="xi",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("xi"),
                ft.DropdownOption("dbscan"),
            ],
            tooltip="The extraction method used to extract clusters. 'xi' uses the xi steep method, 'dbscan' uses a fixed eps like DBSCAN.",
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
                                ft.Text("OPTICS Clustering",
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
                        ft.Row([self.min_samples_field, self.max_eps_field]),
                        ft.Row([self.metric_dropdown, self.cluster_method_dropdown]),
                        self.xi_field,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
