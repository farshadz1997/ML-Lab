"""
LightGBM Classification and Regression Model

Light Gradient Boosting Machine - fast, distributed gradient boosting
framework that uses tree-based learning algorithms.

Supports both:
- Classification: LGBMClassifier
- Regression: LGBMRegressor

Configurable hyperparameters:
- n_estimators: Number of boosting iterations
- max_depth: Maximum tree depth
- learning_rate: Boosting learning rate
- num_leaves: Maximum tree leaves
- subsample: Bagging fraction
- colsample_bytree: Feature fraction
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
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

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
class LightGBMModel(BaseModel):
    """LightGBM model supporting both classification and regression."""

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
            if max_depth < -1:
                max_depth = -1
                is_valid = False
            params['max_depth'] = max_depth
        except (ValueError, TypeError):
            params['max_depth'] = -1
            is_valid = False

        try:
            lr = float(self.learning_rate_field.value)
            if lr <= 0:
                lr = 0.1
                is_valid = False
            params['learning_rate'] = lr
        except (ValueError, TypeError):
            params['learning_rate'] = 0.1
            is_valid = False

        try:
            num_leaves = int(self.num_leaves_field.value)
            if num_leaves < 2:
                num_leaves = 31
                is_valid = False
            params['num_leaves'] = num_leaves
        except (ValueError, TypeError):
            params['num_leaves'] = 31
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
                reg_lambda = 0
                is_valid = False
            params['reg_lambda'] = reg_lambda
        except (ValueError, TypeError):
            params['reg_lambda'] = 0
            is_valid = False

        try:
            min_child = int(self.min_child_samples_field.value)
            if min_child < 1:
                min_child = 20
                is_valid = False
            params['min_child_samples'] = min_child
        except (ValueError, TypeError):
            params['min_child_samples'] = 20
            is_valid = False

        return params, is_valid

    def _create_model(self, **kwargs) -> LGBMClassifier | LGBMRegressor:
        hyperparams, params_valid = self._validate_hyperparameters()
        if not params_valid:
            self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)
        task_type = self._get_task_type()
        if task_type == "Classification":
            model = LGBMClassifier(
                n_estimators=hyperparams['n_estimators'],
                max_depth=hyperparams['max_depth'],
                learning_rate=hyperparams['learning_rate'],
                num_leaves=hyperparams['num_leaves'],
                subsample=hyperparams['subsample'],
                colsample_bytree=hyperparams['colsample_bytree'],
                reg_alpha=hyperparams['reg_alpha'],
                reg_lambda=hyperparams['reg_lambda'],
                min_child_samples=hyperparams['min_child_samples'],
                boosting_type=self.boosting_dropdown.value,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            model = LGBMRegressor(
                n_estimators=hyperparams['n_estimators'],
                max_depth=hyperparams['max_depth'],
                learning_rate=hyperparams['learning_rate'],
                num_leaves=hyperparams['num_leaves'],
                subsample=hyperparams['subsample'],
                colsample_bytree=hyperparams['colsample_bytree'],
                reg_alpha=hyperparams['reg_alpha'],
                reg_lambda=hyperparams['reg_lambda'],
                min_child_samples=hyperparams['min_child_samples'],
                boosting_type=self.boosting_dropdown.value,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        return model
        
    def _train_and_evaluate_model(self, e: ft.ControlEvent | None = None, force: bool = False) -> None:
        """Train LightGBM model and display evaluation results."""
        try:
            self._disable_training_controls()

            if not LIGHTGBM_AVAILABLE:
                self._show_snackbar("LightGBM is not installed. Run: pip install lightgbm", bgcolor=ft.Colors.RED_500)
                enable_navigation_bar(self.parent.page)
                return

            data = self._prepare_data()
            if data is None:
                enable_navigation_bar(self.parent.page)
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
            result_text += self._generate_code_block(
                imports=[
                    "from lightgbm import LGBMClassifier" if
                    task_type == "Classification" else
                    "from lightgbm import LGBMRegressor"
                ],
                model=model.__class__.__name__,
                model_kwargs=dict(
                    n_estimators=model.n_estimators,
                    max_depth=model.max_depth,
                    learning_rate=model.learning_rate,
                    num_leaves=model.num_leaves,
                    subsample=model.subsample,
                    colsample_bytree=model.colsample_bytree,
                    reg_alpha=model.reg_alpha,
                    reg_lambda=model.reg_lambda,
                    min_child_samples=model.min_child_samples,
                    boosting_type=model.boosting_type,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            )
            
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                f"LightGBM {task_type} Results",
                result_text,
                "LightGBM"
            )
            self.parent.page.open(evaluation_dialog)

        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self._enable_training_controls()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for LightGBM hyperparameter configuration."""

        self.n_estimators_field = ft.TextField(
            label="Number of Estimators",
            value="100",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Number of boosting iterations.",
        )

        self.max_depth_field = ft.TextField(
            label="Max Depth",
            value="-1",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^-?\d*$'),
            tooltip="Maximum tree depth for base learners. -1 means no limit.",
        )

        self.learning_rate_field = ft.TextField(
            label="Learning Rate",
            value="0.1",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Boosting learning rate.",
        )

        self.num_leaves_field = ft.TextField(
            label="Num Leaves",
            value="31",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Maximum tree leaves for base learners. Main parameter to control complexity.",
        )

        self.subsample_field = ft.TextField(
            label="Subsample",
            value="1.0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Subsample ratio of the training instance. Range: (0, 1].",
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
            tooltip="L1 regularization term on weights.",
        )

        self.reg_lambda_field = ft.TextField(
            label="Reg Lambda (L2)",
            value="0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="L2 regularization term on weights.",
        )

        self.min_child_samples_field = ft.TextField(
            label="Min Child Samples",
            value="20",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Minimum number of data needed in a child (leaf).",
        )

        self.boosting_dropdown = ft.Dropdown(
            label="Boosting Type",
            value="gbdt",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("gbdt"),
                ft.DropdownOption("dart"),
                ft.DropdownOption("rf"),
            ],
            tooltip="gbdt=traditional Gradient Boosting, dart=Dropouts meet Multiple Additive Regression Trees, rf=Random Forest.",
        )

        self._build_train_button()
        self._build_predict_new_data_button()
        
        if not LIGHTGBM_AVAILABLE:
            warning = ft.Text(
                "LightGBM not installed. Run: pip install lightgbm",
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
                                ft.Text("LightGBM",
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
                        ft.Row([self.learning_rate_field, self.num_leaves_field]),
                        ft.Row([self.subsample_field, self.colsample_field]),
                        ft.Row([self.reg_alpha_field, self.reg_lambda_field]),
                        ft.Row([self.min_child_samples_field, self.boosting_dropdown]),
                        ft.Row([self.train_btn, self.test_data_btn])
                    ]
                )
            )
        )
