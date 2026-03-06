"""
XGBoost Classification and Regression Model

Extreme Gradient Boosting - optimized distributed gradient boosting
library. Industry standard for tabular data competitions.

Supports both:
- Classification: XGBClassifier
- Regression: XGBRegressor

Configurable hyperparameters:
- n_estimators: Number of boosting rounds
- max_depth: Maximum tree depth
- learning_rate: Step size shrinkage
- subsample: Row sampling ratio
- colsample_bytree: Column sampling ratio
- reg_alpha: L1 regularization
- reg_lambda: L2 regularization
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
from functools import partial
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, KFold

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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
class XGBoostModel(BaseModel):
    """XGBoost model supporting both classification and regression."""

    def _prepare_data(self):
        return self._prepare_data_supervised()

    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        is_valid = True
        params = {}

        try:
            n_est = int(self.n_estimators_field.value)
            if n_est < 1:
                n_est = 100
                is_valid = False
            params['n_estimators'] = n_est
        except (ValueError, TypeError):
            params['n_estimators'] = 100
            is_valid = False

        try:
            max_depth = int(self.max_depth_field.value)
            if max_depth < 1 or max_depth > 100:
                max_depth = 6
                is_valid = False
            params['max_depth'] = max_depth
        except (ValueError, TypeError):
            params['max_depth'] = 6
            is_valid = False

        try:
            lr = float(self.learning_rate_field.value)
            if lr <= 0:
                lr = 0.3
                is_valid = False
            params['learning_rate'] = lr
        except (ValueError, TypeError):
            params['learning_rate'] = 0.3
            is_valid = False

        try:
            subsample = float(self.subsample_field.value)
            if subsample <= 0 or subsample > 1:
                subsample = 1.0
                is_valid = False
            params['subsample'] = subsample
        except (ValueError, TypeError):
            params['subsample'] = 1.0
            is_valid = False

        try:
            colsample = float(self.colsample_field.value)
            if colsample <= 0 or colsample > 1:
                colsample = 1.0
                is_valid = False
            params['colsample_bytree'] = colsample
        except (ValueError, TypeError):
            params['colsample_bytree'] = 1.0
            is_valid = False

        try:
            reg_alpha = float(self.reg_alpha_field.value)
            if reg_alpha < 0:
                reg_alpha = 0
                is_valid = False
            params['reg_alpha'] = reg_alpha
        except (ValueError, TypeError):
            params['reg_alpha'] = 0
            is_valid = False

        try:
            reg_lambda = float(self.reg_lambda_field.value)
            if reg_lambda < 0:
                reg_lambda = 1
                is_valid = False
            params['reg_lambda'] = reg_lambda
        except (ValueError, TypeError):
            params['reg_lambda'] = 1
            is_valid = False

        try:
            min_child = float(self.min_child_weight_field.value)
            if min_child < 0:
                min_child = 1
                is_valid = False
            params['min_child_weight'] = min_child
        except (ValueError, TypeError):
            params['min_child_weight'] = 1
            is_valid = False

        return params, is_valid

    def _train_and_evaluate_model(self, e: ft.ControlEvent | None = None, force: bool = False) -> None:
        """Train XGBoost model and display evaluation results."""
        try:
            self._disable_training_controls()

            if not XGBOOST_AVAILABLE:
                self._show_snackbar("XGBoost is not installed. Run: pip install xgboost", bgcolor=ft.Colors.RED_500)
                return

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

            task_type = self._get_task_type()

            if task_type == "Classification":
                model = XGBClassifier(
                    n_estimators=hyperparams['n_estimators'],
                    max_depth=hyperparams['max_depth'],
                    learning_rate=hyperparams['learning_rate'],
                    subsample=hyperparams['subsample'],
                    colsample_bytree=hyperparams['colsample_bytree'],
                    reg_alpha=hyperparams['reg_alpha'],
                    reg_lambda=hyperparams['reg_lambda'],
                    min_child_weight=hyperparams['min_child_weight'],
                    booster=self.booster_dropdown.value,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
            else:
                model = XGBRegressor(
                    n_estimators=hyperparams['n_estimators'],
                    max_depth=hyperparams['max_depth'],
                    learning_rate=hyperparams['learning_rate'],
                    subsample=hyperparams['subsample'],
                    colsample_bytree=hyperparams['colsample_bytree'],
                    reg_alpha=hyperparams['reg_alpha'],
                    reg_lambda=hyperparams['reg_lambda'],
                    min_child_weight=hyperparams['min_child_weight'],
                    booster=self.booster_dropdown.value,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
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

            importance = get_feature_importance(model, self.df.columns.tolist())
            if importance:
                result_text += "\n\n**Feature Importance:**\n\n"
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feature, imp_value in sorted_importance:
                    result_text += f"- {feature}: {imp_value:.4f}\n"

            evaluation_dialog = create_results_dialog(
                self.parent.page,
                f"XGBoost {task_type} Results",
                result_text,
                "XGBoost"
            )
            self.parent.page.open(evaluation_dialog)

        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self._enable_training_controls()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for XGBoost hyperparameter configuration."""

        self.n_estimators_field = ft.TextField(
            label="Number of Estimators",
            value="100",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Number of gradient boosted trees. Equivalent to number of boosting rounds.",
        )

        self.max_depth_field = ft.TextField(
            label="Max Depth",
            value="6",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.",
        )

        self.learning_rate_field = ft.TextField(
            label="Learning Rate",
            value="0.3",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Step size shrinkage used to prevent overfitting. Range: (0, 1].",
        )

        self.subsample_field = ft.TextField(
            label="Subsample",
            value="1.0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Subsample ratio of the training instances. Setting to 0.5 means XGBoost randomly samples half the data. Range: (0, 1].",
        )

        self.colsample_field = ft.TextField(
            label="Col Sample by Tree",
            value="1.0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Subsample ratio of columns when constructing each tree. Range: (0, 1].",
        )

        self.reg_alpha_field = ft.TextField(
            label="Reg Alpha (L1)",
            value="0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="L1 regularization term on weights. Increasing this value will make model more conservative.",
        )

        self.reg_lambda_field = ft.TextField(
            label="Reg Lambda (L2)",
            value="1",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="L2 regularization term on weights. Increasing this value will make model more conservative.",
        )

        self.min_child_weight_field = ft.TextField(
            label="Min Child Weight",
            value="1",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Minimum sum of instance weight needed in a child. The larger, the more conservative the algorithm will be.",
        )

        self.booster_dropdown = ft.Dropdown(
            label="Booster",
            value="gbtree",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("gbtree"),
                ft.DropdownOption("gblinear"),
                ft.DropdownOption("dart"),
            ],
            tooltip="Which booster to use. gbtree=tree-based, gblinear=linear, dart=tree with dropout.",
        )

        self._build_train_button()

        if not XGBOOST_AVAILABLE:
            warning = ft.Text(
                "XGBoost not installed. Run: pip install xgboost",
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
                                ft.Text("XGBoost",
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
                        ft.Row([self.n_estimators_field, self.max_depth_field]),
                        ft.Row([self.learning_rate_field, self.booster_dropdown]),
                        ft.Row([self.subsample_field, self.colsample_field]),
                        ft.Row([self.reg_alpha_field, self.reg_lambda_field]),
                        self.min_child_weight_field,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
