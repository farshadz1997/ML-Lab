"""
Multi-Layer Perceptron (MLP) Classification and Regression Model

Neural network model with configurable hidden layers.
Stays within sklearn ecosystem for consistency.

Supports both:
- Classification: MLPClassifier
- Regression: MLPRegressor

Configurable hyperparameters:
- hidden_layer_sizes: Architecture of hidden layers
- activation: Activation function
- solver: Weight optimization solver
- alpha: L2 regularization
- learning_rate: Learning rate schedule
- max_iter: Maximum iterations
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
from functools import partial
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor

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
class MLPModel(BaseModel):
    """Multi-Layer Perceptron model supporting both classification and regression."""

    def _prepare_data(self):
        return self._prepare_data_supervised()

    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        is_valid = True
        params = {}

        # Parse hidden_layer_sizes (e.g., "100,50" -> (100, 50))
        try:
            raw = self.hidden_layers_field.value.strip()
            sizes = tuple(int(x.strip()) for x in raw.split(',') if x.strip())
            if not sizes or any(s < 1 for s in sizes):
                sizes = (100,)
                is_valid = False
            params['hidden_layer_sizes'] = sizes
        except (ValueError, TypeError):
            params['hidden_layer_sizes'] = (100,)
            is_valid = False

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
                max_iter = 200
                is_valid = False
            params['max_iter'] = max_iter
        except (ValueError, TypeError):
            params['max_iter'] = 200
            is_valid = False

        try:
            learning_rate_init = float(self.learning_rate_init_field.value)
            if learning_rate_init <= 0:
                learning_rate_init = 0.001
                is_valid = False
            params['learning_rate_init'] = learning_rate_init
        except (ValueError, TypeError):
            params['learning_rate_init'] = 0.001
            is_valid = False

        return params, is_valid

    def _train_and_evaluate_model(self, e: ft.ControlEvent | None = None, force: bool = False) -> None:
        """Train MLP model and display evaluation results."""
        try:
            self.train_btn.disabled = True
            self.parent.disable_model_selection()
            disable_navigation_bar(self.parent.page)
            self.parent.page.update()

            data = self._prepare_data()
            if data is None:
                enable_navigation_bar(self.parent.page)
                return

            ratio = self._target_to_total_rows_ratio()
            if ratio > CLASSIFICATION_THRESHOLD and self._get_task_type() == "Classification" and not force:
                self._show_wrong_task_type_dialog(ratio, 'Classification', partial(self._train_and_evaluate_model, e=e, force=True))
                return
            
            X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols) = data

            hyperparams, params_valid = self._validate_hyperparameters()

            if not params_valid:
                self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)

            task_type = self._get_task_type()

            if task_type == "Classification":
                model = MLPClassifier(
                    hidden_layer_sizes=hyperparams['hidden_layer_sizes'],
                    activation=self.activation_dropdown.value,
                    solver=self.solver_dropdown.value,
                    alpha=hyperparams['alpha'],
                    learning_rate=self.learning_rate_dropdown.value,
                    learning_rate_init=hyperparams['learning_rate_init'],
                    max_iter=hyperparams['max_iter'],
                    early_stopping=self.early_stopping_switch.value,
                    random_state=42,
                )
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=hyperparams['hidden_layer_sizes'],
                    activation=self.activation_dropdown.value,
                    solver=self.solver_dropdown.value,
                    alpha=hyperparams['alpha'],
                    learning_rate=self.learning_rate_dropdown.value,
                    learning_rate_init=hyperparams['learning_rate_init'],
                    max_iter=hyperparams['max_iter'],
                    early_stopping=self.early_stopping_switch.value,
                    random_state=42,
                )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            kf = KFold(
                n_splits=int(self.parent.n_split_slider.value),
                shuffle=self.parent.cross_val_shuffle_switch.value,
                random_state=42 if self.parent.cross_val_shuffle_switch.value else None
            )
            cv_results = cross_val_score(model, X_train, y_train, cv=kf)

            if task_type == "Classification":
                metrics_dict = calculate_classification_metrics(y_test, y_pred)
                metrics_dict["CV"] = cv_results
                result_text = format_results_markdown(metrics_dict, task_type="classification")
            else:
                metrics_dict = calculate_regression_metrics(y_test, y_pred)
                metrics_dict["CV"] = cv_results
                result_text = format_results_markdown(metrics_dict, task_type="regression")

            result_text += f"\n\n**Training Iterations:** {model.n_iter_}\n"
            result_text += f"**Final Loss:** {model.loss_:.6f}\n"

            evaluation_dialog = create_results_dialog(
                self.parent.page,
                f"MLP {task_type} Results",
                result_text,
                "MLP"
            )
            self.parent.page.open(evaluation_dialog)
            enable_navigation_bar(self.parent.page)

        except Exception as e:
            enable_navigation_bar(self.parent.page)
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self.parent.enable_model_selection()
            self.train_btn.disabled = False
            self.parent.page.update()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for MLP hyperparameter configuration."""

        self.hidden_layers_field = ft.TextField(
            label="Hidden Layer Sizes",
            value="100",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Comma-separated layer sizes. E.g. '100' for one layer, '100,50' for two layers, '128,64,32' for three.",
        )

        self.activation_dropdown = ft.Dropdown(
            label="Activation",
            value="relu",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("relu"),
                ft.DropdownOption("tanh"),
                ft.DropdownOption("logistic"),
                ft.DropdownOption("identity"),
            ],
            tooltip="Activation function for hidden layers. relu=fast/default, tanh=bounded, logistic=sigmoid.",
        )

        self.solver_dropdown = ft.Dropdown(
            label="Solver",
            value="adam",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("adam"),
                ft.DropdownOption("sgd"),
                ft.DropdownOption("lbfgs"),
            ],
            tooltip="The solver for weight optimization. 'adam' works well for large datasets, 'lbfgs' for small datasets.",
        )

        self.alpha_field = ft.TextField(
            label="Alpha (L2 Penalty)",
            value="0.0001",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$'),
            tooltip="Strength of the L2 regularization term.",
        )

        self.learning_rate_dropdown = ft.Dropdown(
            label="Learning Rate Schedule",
            value="constant",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("constant"),
                ft.DropdownOption("invscaling"),
                ft.DropdownOption("adaptive"),
            ],
            tooltip="Learning rate schedule for weight updates. Only used with 'sgd' solver.",
        )

        self.learning_rate_init_field = ft.TextField(
            label="Initial Learning Rate",
            value="0.001",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$'),
            tooltip="The initial learning rate used. Only used when solver='sgd' or 'adam'.",
        )

        self.max_iter_field = ft.TextField(
            label="Max Iterations",
            value="200",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Maximum number of iterations. The solver iterates until convergence or this number of iterations.",
        )

        self.early_stopping_switch = ft.Switch(
            label="Early Stopping",
            value=False,
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Whether to use early stopping to terminate training when validation score is not improving.",
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
                                ft.Text("Multi-Layer Perceptron",
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
                        self.hidden_layers_field,
                        ft.Row([self.activation_dropdown, self.solver_dropdown]),
                        ft.Row([self.alpha_field, self.learning_rate_init_field]),
                        ft.Row([self.learning_rate_dropdown, self.max_iter_field]),
                        self.early_stopping_switch,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
