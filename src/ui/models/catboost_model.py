"""
CatBoost Classification and Regression Model

Gradient boosting on decision trees with native support for
categorical features without preprocessing.

Supports both:
- Classification: CatBoostClassifier
- Regression: CatBoostRegressor

Configurable hyperparameters:
- iterations: Number of boosting iterations
- depth: Tree depth
- learning_rate: Step size shrinkage
- l2_leaf_reg: L2 regularization
- border_count: Number of splits for numerical features
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
from functools import partial
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, KFold

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

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
class CatBoostModel(BaseModel):
    """CatBoost model supporting both classification and regression."""

    def _prepare_data(self):
        return self._prepare_data_supervised()

    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        is_valid = True
        params = {}

        try:
            iterations = int(self.iterations_field.value)
            if iterations < 1:
                iterations = 1000
                is_valid = False
            params['iterations'] = iterations
        except (ValueError, TypeError):
            params['iterations'] = 1000
            is_valid = False

        try:
            depth = int(self.depth_field.value)
            if depth < 1 or depth > 16:
                depth = 6
                is_valid = False
            params['depth'] = depth
        except (ValueError, TypeError):
            params['depth'] = 6
            is_valid = False

        try:
            lr_val = self.learning_rate_field.value
            if lr_val == 'None' or lr_val == '':
                params['learning_rate'] = None
            else:
                lr = float(lr_val)
                if lr <= 0:
                    lr = None
                    is_valid = False
                params['learning_rate'] = lr
        except (ValueError, TypeError):
            params['learning_rate'] = None
            is_valid = False

        try:
            l2 = float(self.l2_leaf_reg_field.value)
            if l2 < 0:
                l2 = 3.0
                is_valid = False
            params['l2_leaf_reg'] = l2
        except (ValueError, TypeError):
            params['l2_leaf_reg'] = 3.0
            is_valid = False

        try:
            border = int(self.border_count_field.value)
            if border < 1:
                border = 254
                is_valid = False
            params['border_count'] = border
        except (ValueError, TypeError):
            params['border_count'] = 254
            is_valid = False

        return params, is_valid

    def _create_model(self, **kwargs) -> CatBoostClassifier | CatBoostRegressor:
        hyperparams, params_valid = self._validate_hyperparameters()
        if not params_valid:
            self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)
        task_type = self._get_task_type()
        model_params = dict(
            iterations=hyperparams['iterations'],
            depth=hyperparams['depth'],
            l2_leaf_reg=hyperparams['l2_leaf_reg'],
            border_count=hyperparams['border_count'],
            random_state=42,
            verbose=0,
        )
        if hyperparams['learning_rate'] is not None:
            model_params['learning_rate'] = hyperparams['learning_rate']
        if task_type == "Classification":
            model = CatBoostClassifier(**model_params)
        else:
            model = CatBoostRegressor(**model_params)
        return model
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent | None = None, force: bool = False) -> None:
        """Train CatBoost model and display evaluation results."""
        try:
            self._disable_training_controls()

            if not CATBOOST_AVAILABLE:
                self._show_snackbar("CatBoost is not installed. Run: pip install catboost", bgcolor=ft.Colors.RED_500)
                return

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
                f"CatBoost {task_type} Results",
                result_text,
                "CatBoost"
            )
            self.parent.page.open(evaluation_dialog)

        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self._enable_training_controls()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for CatBoost hyperparameter configuration."""

        self.iterations_field = ft.TextField(
            label="Iterations",
            value="1000",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The maximum number of trees that can be built.",
        )

        self.depth_field = ft.TextField(
            label="Depth",
            value="6",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Depth of the tree. Range: [1, 16].",
        )

        self.learning_rate_field = ft.TextField(
            label="Learning Rate",
            value="None",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            on_click=self._field_on_click,
            on_blur=self._field_on_blur,
            suffix_icon=ft.IconButton(ft.Icons.RESTART_ALT, on_click=lambda _: self._reset_field_to_none(self.learning_rate_field), tooltip="Reset to None (auto)"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="The learning rate. Used for reducing the gradient step. None = auto-selected by CatBoost.",
        )

        self.l2_leaf_reg_field = ft.TextField(
            label="L2 Leaf Reg",
            value="3.0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Coefficient at the L2 regularization term of the cost function.",
        )

        self.border_count_field = ft.TextField(
            label="Border Count",
            value="254",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The number of splits for numerical features. More = better quality but slower.",
        )

        self._build_train_button()
        self._build_predict_new_data_button()
        
        if not CATBOOST_AVAILABLE:
            warning = ft.Text(
                "CatBoost not installed. Run: pip install catboost",
                font_family="SF regular",
                color=ft.Colors.RED_500,
                size=12,
            )
        else:
            warning = ft.Container()

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
                                ft.Text("CatBoost",
                                       font_family="SF thin",
                                       size=24,
                                       text_align="center",
                                       expand=True)
                            ]
                        ),
                        warning,
                        ft.Divider(),
                        ft.Text("Hyperparameters",
                               font_family="SF regular",
                               weight="bold",
                               size=14),
                        ft.Row([self.iterations_field, self.depth_field]),
                        ft.Row([self.learning_rate_field, self.l2_leaf_reg_field]),
                        self.border_count_field,
                        ft.Row([self.train_btn, self.test_data_btn])
                    ]
                )
            )
        )
