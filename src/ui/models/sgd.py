"""
Stochastic Gradient Descent (SGD) Classification and Regression Model

Linear models fitted by minimizing a regularized loss function
with stochastic gradient descent. Scales well to very large datasets.

Supports both:
- Classification: SGDClassifier
- Regression: SGDRegressor

Configurable hyperparameters:
- loss: Loss function
- penalty: Regularization term
- alpha: Regularization strength
- max_iter: Maximum iterations
- learning_rate: Learning rate schedule
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
from functools import partial
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import SGDClassifier, SGDRegressor

from utils.model_utils import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel, CLASSIFICATION_THRESHOLD


@dataclass
class SGDModel(BaseModel):
    """SGD model supporting both classification and regression."""

    def _prepare_data(self):
        return self._prepare_data_supervised()

    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        is_valid = True
        params = {}

        try:
            alpha = float(self.alpha_field.value)
            if alpha < 0:
                alpha = 0.0001
                is_valid = False
            params['alpha'] = alpha
        except (ValueError, TypeError):
            params['alpha'] = 0.0001
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
                tol = 1e-3
                is_valid = False
            params['tol'] = tol
        except (ValueError, TypeError):
            params['tol'] = 1e-3
            is_valid = False

        try:
            l1_ratio_val = float(self.l1_ratio_field.value)
            if l1_ratio_val < 0 or l1_ratio_val > 1:
                l1_ratio_val = 0.15
                is_valid = False
            params['l1_ratio'] = l1_ratio_val
        except (ValueError, TypeError):
            params['l1_ratio'] = 0.15
            is_valid = False
            
        try:
            eta0_val = float(self.eta0_field.value)
            if eta0_val == 0 and self.learning_rate_dropdown.value != "optimal":
                eta0_val = 0.01
                is_valid = False
            params['eta0'] = eta0_val
        except (ValueError, TypeError):
            params['eta0'] = 0
            is_valid = False
            
        try:
            power_t_val = float(self.power_t_field.value)
            params['power_t'] = power_t_val
        except (ValueError, TypeError):
            params['power_t'] = 0.25
            is_valid = False

        return params, is_valid

    def _create_model(self, **kwargs) -> SGDClassifier | SGDRegressor:
        hyperparams, params_valid = self._validate_hyperparameters()
        if not params_valid:
            self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)
        task_type = self._get_task_type()
        if task_type == "Classification":
            model = SGDClassifier(
                loss=self.loss_dropdown.value,
                penalty=self.penalty_dropdown.value,
                alpha=hyperparams['alpha'],
                l1_ratio=hyperparams['l1_ratio'],
                max_iter=hyperparams['max_iter'],
                tol=hyperparams['tol'],
                learning_rate=self.learning_rate_dropdown.value,
                eta0=hyperparams['eta0'],
                power_t=hyperparams['power_t'],
                shuffle=True,
                random_state=42,
            )
        else:
            model = SGDRegressor(
                loss=self.loss_dropdown.value,
                penalty=self.penalty_dropdown.value,
                alpha=hyperparams['alpha'],
                l1_ratio=hyperparams['l1_ratio'],
                max_iter=hyperparams['max_iter'],
                tol=hyperparams['tol'],
                learning_rate=self.learning_rate_dropdown.value,
                eta0=hyperparams['eta0'],
                power_t=hyperparams['power_t'],
                shuffle=True,
                random_state=42,
            )
        return model
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent | None = None, force: bool = False) -> None:
        """Train SGD model and display evaluation results."""
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

            model = self._create_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            kf = KFold(
                n_splits=int(self.parent.n_split_slider.value),
                shuffle=self.parent.cross_val_shuffle_switch.value,
                random_state=42 if self.parent.cross_val_shuffle_switch.value else None
            )
            cv_results = cross_val_score(model, X_train, y_train, cv=kf)

            task_type = self._get_task_type()
            if task_type == "Classification":
                metrics_dict = calculate_classification_metrics(y_test, y_pred)
                metrics_dict["CV"] = cv_results
                result_text = format_results_markdown(metrics_dict, task_type="classification")
            else:
                metrics_dict = calculate_regression_metrics(y_test, y_pred)
                metrics_dict["CV"] = cv_results
                result_text = format_results_markdown(metrics_dict, task_type="regression")
            result_text += self._generate_code_block(
                imports=[
                    "from sklearn.linear_model import SGDClassifier" if
                    task_type == "Classification" else
                    "from sklearn.linear_model import SGDRegressor"
                ],
                model=model.__class__.__name__,
                model_kwargs=dict(
                    loss=model.loss,
                    penalty=model.penalty,
                    alpha=model.alpha,
                    l1_ratio=model.l1_ratio,
                    max_iter=model.max_iter,
                    tol=model.tol,
                    learning_rate=model.learning_rate,
                    eta0=model.eta0,
                    power_t=model.power_t,
                    shuffle=True,
                    random_state=42,
                )
            )
            
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                f"SGD {task_type} Results",
                result_text,
                "SGD"
            )
            self.parent.page.open(evaluation_dialog)

        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self._enable_training_controls()

    def _learning_rate_changed(self, e: ft.ControlEvent):
        lr_value = self.learning_rate_dropdown.value
        if lr_value == "optimal":
            self.eta0_field.value = "0"
            self.eta0_field.disabled = True
            self.power_t_field.disabled = True
        elif lr_value in ["constant", "adaptive"]:
            self.eta0_field.disabled = False
            self.power_t_field.disabled = True
        elif lr_value == "invscaling":
            self.eta0_field.disabled = False
            self.power_t_field.disabled = False
        else:
             self.eta0_field.value = "0"
             self.eta0_field.disabled = True
             self.power_t_field.disabled = True
        self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for SGD hyperparameter configuration."""

        task_type = self._get_task_type()

        if task_type == "Classification":
            loss_options = ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron']
            default_loss = 'hinge'
        else:
            loss_options = ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
            default_loss = 'squared_error'

        self.loss_dropdown = ft.Dropdown(
            label="Loss",
            value=default_loss,
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[ft.DropdownOption(opt) for opt in loss_options],
            tooltip="The loss function to be used.",
        )

        self.penalty_dropdown = ft.Dropdown(
            label="Penalty",
            value="l2",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("l2"),
                ft.DropdownOption("l1"),
                ft.DropdownOption("elasticnet"),
            ],
            tooltip="The penalty (regularization term) to be used.",
        )

        self.alpha_field = ft.TextField(
            label="Alpha",
            value="0.0001",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$'),
            tooltip="Constant that multiplies the regularization term. The higher the value, the stronger the regularization.",
        )

        self.l1_ratio_field = ft.TextField(
            label="L1 Ratio",
            value="0.15",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="The Elastic Net mixing parameter. 0 = L2 penalty, 1 = L1 penalty. Only used if penalty='elasticnet'. Range: [0, 1].",
        )

        self.max_iter_field = ft.TextField(
            label="Max Iterations",
            value="1000",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The maximum number of passes over the training data (epochs).",
        )

        self.tol_field = ft.TextField(
            label="Tolerance",
            value="1e-3",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$'),
            tooltip="The stopping criterion.",
        )

        self.learning_rate_dropdown = ft.Dropdown(
            label="Learning Rate",
            value="optimal",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("constant"),
                ft.DropdownOption("optimal"),
                ft.DropdownOption("invscaling"),
                ft.DropdownOption("adaptive"),
            ],
            on_change=self._learning_rate_changed,
            tooltip="The learning rate schedule. 'optimal' uses a heuristic, 'adaptive' keeps the learning rate constant as long as training loss keeps decreasing.",
        )
        
        self.eta0_field = ft.TextField(
            label="Initial Learning Rate (eta0)",
            value="0",
            expand=1,
            disabled=True,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="The initial learning rate for the 'constant', 'invscaling', or 'adaptive' schedules. Ignored when learning_rate='optimal'.",
        )
        
        self.power_t_field = ft.TextField(
            label="Power T (for 'invscaling')",
            value="0.25",
            expand=1,
            disabled=True,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^[+-]?(\d+(\.\d*)?|\.\d+)$'),
            tooltip="The exponent for inverse scaling learning rate. Only used when learning_rate='invscaling'. Learning rate = eta0 / pow(t, power_t)",
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
                                ft.Text("Stochastic Gradient Descent",
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
                        ft.Row([self.loss_dropdown, self.penalty_dropdown]),
                        ft.Row([self.alpha_field, self.l1_ratio_field]),
                        ft.Row([self.max_iter_field, self.tol_field]),
                        ft.Row([self.learning_rate_dropdown, self.eta0_field]),
                        self.power_t_field,
                        ft.Row([self.train_btn, self.test_data_btn])
                    ]
                )
            )
        )
