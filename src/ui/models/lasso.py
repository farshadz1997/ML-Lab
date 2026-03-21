"""
Lasso Regression Model

Linear regression with L1 regularization. Can produce sparse models
by driving some coefficients to exactly zero (feature selection).

Configurable hyperparameters:
- alpha: Regularization strength
- max_iter: Maximum iterations
- selection: Coefficient update strategy
- fit_intercept: Whether to calculate intercept
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso

from utils.model_utils import (
    calculate_regression_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel


@dataclass
class LassoModel(BaseModel):
    """Lasso (L1) regression model."""

    def _prepare_data(self):
        return self._prepare_data_supervised()

    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        is_valid = True
        params = {}

        try:
            alpha = float(self.alpha_field.value)
            if alpha < 0:
                alpha = 1.0
                is_valid = False
            params['alpha'] = alpha
        except (ValueError, TypeError):
            params['alpha'] = 1.0
            is_valid = False

        try:
            max_iter = int(self.max_iter_field.value)
            if max_iter < 1:
                max_iter = 1000
                is_valid = False
            params['max_iter'] = max_iter
        except (ValueError, TypeError):
            params['max_iter'] = 1000
            is_valid = False

        try:
            tol = float(self.tol_field.value)
            if tol <= 0:
                tol = 1e-4
                is_valid = False
            params['tol'] = tol
        except (ValueError, TypeError):
            params['tol'] = 1e-4
            is_valid = False

        return params, is_valid

    def _create_model(self, **kwargs) -> Lasso:
        hyperparams, params_valid = self._validate_hyperparameters()
        if not params_valid:
            self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)
        model = Lasso(
            alpha=hyperparams['alpha'],
            fit_intercept=self.fit_intercept_switch.value,
            max_iter=hyperparams['max_iter'],
            tol=hyperparams['tol'],
            selection=self.selection_dropdown.value,
            random_state=42,
        )
        return model
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train Lasso model and display evaluation results."""
        try:
            self._disable_training_controls()

            data = self._prepare_data()
            if data is None:
                return

            X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols) = data

            model = self._create_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            kf = KFold(
                n_splits=int(self.parent.n_split_slider.value),
                shuffle=self.parent.cross_val_shuffle_switch.value,
                random_state=42 if self.parent.cross_val_shuffle_switch.value else None
            )
            cv_results = cross_val_score(model, X_train, y_train, cv=kf)

            metrics_dict = calculate_regression_metrics(y_test, y_pred)
            metrics_dict["CV"] = cv_results

            result_text = f"**Model Intercept:** {model.intercept_:.4f}\n\n"
            # Show number of non-zero coefficients (sparsity)
            n_nonzero = sum(1 for c in model.coef_ if abs(c) > 1e-10)
            result_text += f"**Non-zero Coefficients:** {n_nonzero}/{len(model.coef_)}\n\n"
            result_text += format_results_markdown(metrics_dict, task_type="regression")
            result_text += self._generate_code_block(
                imports=["from sklearn.linear_model import Lasso"],
                model=model.__class__.__name__,
                model_kwargs=dict(
                    alpha=model.alpha,
                    fit_intercept=model.fit_intercept,
                    max_iter=model.max_iter,
                    tol=model.tol,
                    selection=model.selection,
                    random_state=42,
                )    
            )
            
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Lasso Regression Results",
                result_text,
                "Lasso"
            )
            self.parent.page.open(evaluation_dialog)

        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self._enable_training_controls()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for Lasso hyperparameter configuration."""

        self.alpha_field = ft.TextField(
            label="Alpha (Regularization)",
            value="1.0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Constant that multiplies the L1 term. alpha=0 is equivalent to ordinary least squares. Higher values = more regularization = more sparsity.",
        )

        self.max_iter_field = ft.TextField(
            label="Max Iterations",
            value="1000",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The maximum number of iterations.",
        )

        self.tol_field = ft.TextField(
            label="Tolerance",
            value="1e-4",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$'),
            tooltip="The tolerance for the optimization: if the updates are smaller than tol, the optimization code checks the dual gap for optimality.",
        )

        self.selection_dropdown = ft.Dropdown(
            label="Selection",
            value="cyclic",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("cyclic"),
                ft.DropdownOption("random"),
            ],
            tooltip="If set to 'random', a random coefficient is updated every iteration rather than looping over features sequentially. This can lead to significantly faster convergence.",
        )

        self.fit_intercept_switch = ft.Switch(
            label="Fit Intercept",
            value=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            label_position=ft.LabelPosition.RIGHT,
            tooltip="Whether to fit the intercept for this model.",
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
                                ft.Text("Lasso Regression",
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
                        ft.Row([self.alpha_field, self.max_iter_field]),
                        ft.Row([self.tol_field, self.selection_dropdown]),
                        self.fit_intercept_switch,
                        ft.Row([self.train_btn, self.test_data_btn])
                    ]
                )
            )
        )
