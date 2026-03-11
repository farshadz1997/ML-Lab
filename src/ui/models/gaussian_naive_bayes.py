"""
Naive Bayes Classification Model

Probabilistic classifier based on Bayes' theorem with strong
independence assumptions between features.

Supports:
- GaussianNB: For continuous features (assumes Gaussian distribution)

Configurable hyperparameters:
- var_smoothing: Portion of the largest variance added for stability
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
from functools import partial
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB

from utils.model_utils import (
    calculate_classification_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel, CLASSIFICATION_THRESHOLD


@dataclass
class GaussianNBModel(BaseModel):
    """Gaussian Naive Bayes classification model."""

    def _prepare_data(self):
        """Prepare data for training."""
        return self._prepare_data_supervised()

    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        is_valid = True
        params = {}

        # Validate var_smoothing
        try:
            var_smoothing = float(self.var_smoothing_field.value)
            if var_smoothing < 0:
                var_smoothing = 1e-9
                is_valid = False
            params['var_smoothing'] = var_smoothing
        except (ValueError, TypeError):
            params['var_smoothing'] = 1e-9
            is_valid = False

        return params, is_valid

    def _train_and_evaluate_model(self, e: ft.ControlEvent | None = None, force: bool = False) -> None:
        """Train Naive Bayes model and display evaluation results."""
        try:
            self._disable_training_controls()

            data = self._prepare_data()
            if data is None:
                return
            
            ratio = self._target_to_total_rows_ratio()
            if ratio > CLASSIFICATION_THRESHOLD and self._get_task_type() == "Classification" and not force:
                self._show_wrong_task_type_dialog(ratio, 'Classification', partial(self._train_and_evaluate_model, e=e, force=True))
                return
            
            X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols) = data

            hyperparams, params_valid = self._validate_hyperparameters()

            if not params_valid:
                self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)

            model = GaussianNB(
                var_smoothing=hyperparams['var_smoothing'],
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Cross validation
            kf = KFold(
                n_splits=int(self.parent.n_split_slider.value),
                shuffle=self.parent.cross_val_shuffle_switch.value,
                random_state=42 if self.parent.cross_val_shuffle_switch.value else None
            )
            cv_results = cross_val_score(model, X_train, y_train, cv=kf)

            metrics_dict = calculate_classification_metrics(y_test, y_pred)
            metrics_dict["CV"] = cv_results
            result_text = format_results_markdown(metrics_dict, task_type="classification")

            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Gaussian Naive Bayes Classification Results",
                result_text,
                "Naive Bayes"
            )
            self.parent.page.open(evaluation_dialog)

        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self._enable_training_controls()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for Naive Bayes hyperparameter configuration."""

        self.var_smoothing_field = ft.TextField(
            label="Var Smoothing",
            value="1e-9",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$'),
            tooltip="Portion of the largest variance of all features that is added to variances for calculation stability. Must be non-negative.",
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
                                ft.Text("Naive Bayes (Gaussian)",
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
                        self.var_smoothing_field,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
