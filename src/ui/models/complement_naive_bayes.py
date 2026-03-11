"""
Naive Bayes Classification Model

Probabilistic classifier based on Bayes' theorem with strong
independence assumptions between features.

Supports:
- Complement: For discrete non-negative counts/frequencies per feature, for imbalanced datasets

Configurable hyperparameters:
- alpha: Additive (Laplace/Lidstone) smoothing parameter (set alpha=0 and force_alpha=True, for no smoothing).
- force_alpha: If False and alpha is less than 1e-10, it will set alpha to 1e-10. If True, alpha will remain unchanged. This may cause numerical errors if alpha is too close to 0
- fit_prior: Only used in edge case with a single class in the training set
- norm: Whether or not a second normalization of the weights is performed. The default behavior mirrors the implementations found in Mahout and Weka, which do not follow the full algorithm described in Table 9 of the paper.
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
from functools import partial
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import ComplementNB

from utils.model_utils import (
    calculate_classification_metrics,
    format_results_markdown,
    create_results_dialog,
)
from .base_model import BaseModel, CLASSIFICATION_THRESHOLD


@dataclass
class ComplementNBModel(BaseModel):
    """Multinomial Naive Bayes classification model."""

    def _prepare_data(self):
        """Prepare data for training."""
        return self._prepare_data_supervised()

    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        is_valid = True
        params = {}

        # Validate alpha
        try:
            alpha = float(self.alpha.value)
            if alpha < 0:
                alpha = 1
                is_valid = False
            params['alpha'] = alpha
        except (ValueError, TypeError):
            params['alpha'] = 1
            is_valid = False
            
        # validated force_alpha
        force_alpha = self.force_alpha.value
        match force_alpha:
            case "true":
                params['force_alpha'] = True
            case "false":
                params['force_alpha'] = False
            case _:
                params['force_alpha'] = True
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

            model = ComplementNB(
                alpha=hyperparams['alpha'],
                force_alpha=hyperparams['force_alpha'],
                fit_prior=self.fit_prior.value,
                norm=self.norm.value,
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
                "Complement Naive Bayes Classification Results",
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

        self.alpha = ft.TextField(
            label="Alpha",
            value="1",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Additive (Laplace/Lidstone) smoothing parameter (set alpha=0 and force_alpha=True, for no smoothing).",
        )
        
        self.force_alpha = ft.RadioGroup(
            value="true",
            content=ft.Row(
                controls=[
                    ft.Radio(
                        label="True",
                        value="true",
                        label_style=ft.TextStyle(
                            font_family="SF regular"
                        )
                    ),
                    ft.Radio(
                        label="False",
                        value="false",
                        label_style=ft.TextStyle(
                            font_family="SF regular"
                        )
                    ),
                ]
            )
        )

        self.fit_prior = ft.Switch(
            label="Fit prior",
            value=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Only used in edge case with a single class in the training set.",
        )
        
        self.norm = ft.Switch(
            label="Norm",
            value=False,
            label_style=ft.TextStyle(font_family="SF regular"),
            label_position=ft.LabelPosition.LEFT,
            tooltip="Whether or not a second normalization of the weights is performed. The default behavior mirrors the implementations found in Mahout and Weka, which do not follow the full algorithm described in Table 9 of the paper."
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
                                ft.Text("Naive Bayes (Complement)",
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
                        self.alpha,
                        ft.Row([ft.Text("Force Alpha:", font_family="SF regular"), self.force_alpha]),
                        ft.Row([self.fit_prior, self.norm], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
