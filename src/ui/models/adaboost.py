"""
AdaBoost Classification and Regression Model

Adaptive Boosting ensemble that combines weak learners sequentially,
adjusting weights to focus on misclassified samples.

Supports both:
- Classification: AdaBoostClassifier
- Regression: AdaBoostRegressor

Configurable hyperparameters:
- n_estimators: Number of weak learners
- learning_rate: Shrinkage of each learner's contribution
- algorithm: SAMME or SAMME.R (classification only)
- loss: Loss function (regression only)
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
from functools import partial
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from utils.model_utils import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    format_results_markdown,
    get_feature_importance,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel, CLASSIFICATION_THRESHOLD


@dataclass
class AdaBoostModel(BaseModel):
    """AdaBoost model supporting both classification and regression."""

    def _prepare_data(self):
        return self._prepare_data_supervised()

    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        is_valid = True
        params = {}

        try:
            n_est = int(self.n_estimators_field.value)
            if n_est < 1:
                n_est = 50
                is_valid = False
            params['n_estimators'] = n_est
        except (ValueError, TypeError):
            params['n_estimators'] = 50
            is_valid = False

        try:
            lr = float(self.learning_rate_field.value)
            if lr <= 0:
                lr = 1.0
                is_valid = False
            params['learning_rate'] = lr
        except (ValueError, TypeError):
            params['learning_rate'] = 1.0
            is_valid = False

        return params, is_valid
    
    def _create_model(self, **kwargs) -> AdaBoostClassifier | AdaBoostRegressor:
        hyperparams, params_valid = self._validate_hyperparameters()
        if not params_valid:
            self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)
        task_type = self._get_task_type()
        if task_type == "Classification":
            model = AdaBoostClassifier(
                n_estimators=hyperparams['n_estimators'],
                learning_rate=hyperparams['learning_rate'],
                random_state=42,
            )
        else:
            model = AdaBoostRegressor(
                n_estimators=hyperparams['n_estimators'],
                learning_rate=hyperparams['learning_rate'],
                loss=self.loss_dropdown.value,
                random_state=42,
            )
        return model
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent | None = None, force: bool = False) -> None:
        """Train AdaBoost model and display evaluation results."""
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

            importance = get_feature_importance(model, self.df.columns.tolist())
            if importance:
                result_text += "\n\n**Feature Importance:**\n\n"
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feature, imp_value in sorted_importance:
                    result_text += f"- {feature}: {imp_value:.4f}\n"

            evaluation_dialog = create_results_dialog(
                self.parent.page,
                f"AdaBoost {task_type} Results",
                result_text,
                "AdaBoost"
            )
            self.parent.page.open(evaluation_dialog)

        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self._enable_training_controls()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for AdaBoost hyperparameter configuration."""

        self.n_estimators_field = ft.TextField(
            label="Number of Estimators",
            value="50",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.",
        )

        self.learning_rate_field = ft.TextField(
            label="Learning Rate",
            value="1.0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier. Trade-off between learning_rate and n_estimators.",
        )

        task_type = self._get_task_type()

        self.loss_dropdown = ft.Dropdown(
            label="Loss",
            value="linear",
            expand=1,
            visible=task_type == "Regression",
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("linear"),
                ft.DropdownOption("square"),
                ft.DropdownOption("exponential"),
            ],
            tooltip="The loss function to use when updating the weights after each boosting iteration.",
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
                                ft.Text("AdaBoost",
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
                        ft.Row([self.n_estimators_field, self.learning_rate_field]),
                        self.loss_dropdown,
                        ft.Row([self.train_btn, self.test_data_btn])
                    ]
                )
            )
        )
