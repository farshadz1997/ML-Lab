"""
Extra Trees Classification and Regression Model

Extremely Randomized Trees ensemble. Similar to Random Forest but
uses random thresholds for each feature rather than searching for
the best thresholds, making it faster.

Supports both:
- Classification: ExtraTreesClassifier
- Regression: ExtraTreesRegressor

Configurable hyperparameters:
- n_estimators: Number of trees
- max_depth: Maximum tree depth
- min_samples_split: Minimum samples to split node
- min_samples_leaf: Minimum samples at leaf node
- max_features: Number of features to consider
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
from functools import partial
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

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
class ExtraTreesModel(BaseModel):
    """Extra Trees model supporting both classification and regression."""

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
            max_depth_val = self.max_depth_field.value
            if max_depth_val == 'None' or max_depth_val == '':
                params['max_depth'] = None
            else:
                max_depth = int(max_depth_val)
                if max_depth < 1 or max_depth > 100:
                    max_depth = 10
                    is_valid = False
                params['max_depth'] = max_depth
        except (ValueError, TypeError):
            params['max_depth'] = 10
            is_valid = False

        try:
            min_samples = int(self.min_samples_split_field.value)
            if min_samples < 2:
                min_samples = 2
                is_valid = False
            params['min_samples_split'] = min_samples
        except (ValueError, TypeError):
            params['min_samples_split'] = 2
            is_valid = False

        try:
            min_leaf = int(self.min_samples_leaf_field.value)
            if min_leaf < 1:
                min_leaf = 1
                is_valid = False
            params['min_samples_leaf'] = min_leaf
        except (ValueError, TypeError):
            params['min_samples_leaf'] = 1
            is_valid = False

        max_features_val = self.max_features_dropdown.value
        if max_features_val in ['sqrt', 'log2']:
            params['max_features'] = max_features_val
        elif max_features_val == 'None':
            params['max_features'] = None
        else:
            params['max_features'] = 'sqrt'
            is_valid = False

        return params, is_valid

    def _create_model(self, **kwargs) -> ExtraTreesClassifier | ExtraTreesRegressor:
        hyperparams, params_valid = self._validate_hyperparameters()
        if not params_valid:
            self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)
        task_type = self._get_task_type()
        if task_type == "Classification":
            model = ExtraTreesClassifier(
                n_estimators=hyperparams['n_estimators'],
                criterion=self.criterion_dropdown.value,
                max_depth=hyperparams['max_depth'],
                min_samples_split=hyperparams['min_samples_split'],
                min_samples_leaf=hyperparams['min_samples_leaf'],
                max_features=hyperparams['max_features'],
                bootstrap=self.bootstrap_switch.value,
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = ExtraTreesRegressor(
                n_estimators=hyperparams['n_estimators'],
                criterion=self.criterion_dropdown.value,
                max_depth=hyperparams['max_depth'],
                min_samples_split=hyperparams['min_samples_split'],
                min_samples_leaf=hyperparams['min_samples_leaf'],
                max_features=hyperparams['max_features'],
                bootstrap=self.bootstrap_switch.value,
                random_state=42,
                n_jobs=-1,
            )
        return model
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent | None = None, force: bool = False) -> None:
        """Train Extra Trees model and display evaluation results."""
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
            result_text += self._generate_code_block(
                imports=[
                    "from sklearn.ensemble import ExtraTreesClassifier" if
                    task_type == "Classification" else
                    "from sklearn.ensemble import ExtraTreesRegressor"
                ],
                model=model.__class__.__name__,
                model_kwargs=dict(
                    n_estimators=model.n_estimators,
                    criterion=model.criterion,
                    max_depth=model.max_depth,
                    min_samples_split=model.min_samples_split,
                    min_samples_leaf=model.min_samples_leaf,
                    max_features=model.max_features,
                    bootstrap=model.bootstrap,
                    random_state=42,
                    n_jobs=-1,
                )
            )
            
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                f"Extra Trees {task_type} Results",
                result_text,
                "Extra Trees"
            )
            self.parent.page.open(evaluation_dialog)

        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self._enable_training_controls()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for Extra Trees hyperparameter configuration."""

        self.n_estimators_field = ft.TextField(
            label="Number of Trees",
            value="100",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The number of trees in the forest.",
        )

        if self._get_task_type() == "Classification":
            criterion_options = ['gini', 'entropy', 'log_loss']
        else:
            criterion_options = ['squared_error', 'absolute_error', 'friedman_mse']
        self.criterion_dropdown = ft.Dropdown(
            label="Criterion",
            value=criterion_options[0],
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[ft.dropdown.Option(key=opt, text=opt) for opt in criterion_options],
            tooltip="The function to measure the quality of a split.",
        )

        self.max_depth_field = ft.TextField(
            label="Max Depth",
            value="None",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            on_click=self._field_on_click,
            on_blur=self._field_on_blur,
            suffix_icon=ft.IconButton(ft.Icons.RESTART_ALT, on_click=lambda _: self._reset_field_to_none(self.max_depth_field), tooltip="Reset to None"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.",
        )

        self.min_samples_split_field = ft.TextField(
            label="Min Samples Split",
            value="2",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The minimum number of samples required to split an internal node.",
        )

        self.min_samples_leaf_field = ft.TextField(
            label="Min Samples Leaf",
            value="1",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The minimum number of samples required to be at a leaf node.",
        )

        self.max_features_dropdown = ft.Dropdown(
            label="Max Features",
            value="sqrt",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.dropdown.Option(key="sqrt"),
                ft.dropdown.Option(key="log2"),
                ft.dropdown.Option(key="None"),
            ],
            tooltip="The number of features to consider when looking for the best split.",
        )

        self.bootstrap_switch = ft.Switch(
            label="Bootstrap",
            value=False,
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Whether bootstrap samples are used. Default is False for Extra Trees (uses whole dataset).",
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
                                ft.Text("Extra Trees",
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
                        ft.Row([self.n_estimators_field, self.max_depth_field]),
                        self.criterion_dropdown,
                        ft.Row([self.min_samples_split_field, self.min_samples_leaf_field]),
                        ft.Row([self.bootstrap_switch, self.max_features_dropdown]),
                        ft.Row([self.train_btn, self.test_data_btn])
                    ]
                )
            )
        )
