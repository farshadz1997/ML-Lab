"""
Random Forest Classification and Regression Model

Ensemble of decision trees supporting both:
- Classification: RandomForestClassifier
- Regression: RandomForestRegressor

Configurable hyperparameters:
- n_estimators: Number of trees
- max_depth: Maximum tree depth
- min_samples_split: Minimum samples to split node
- random_state: Random seed for reproducibility
- n_jobs: Parallel jobs (-1 = all cores)
"""

from __future__ import annotations
from typing import Optional, Tuple
import flet as ft
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from utils.model_utils import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    format_results_markdown,
    get_feature_importance,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel


@dataclass
class RandomForestModel(BaseModel):
    """Random Forest model supporting both classification and regression."""

    def _prepare_data(self):
        """Prepare data for training."""
        return self._prepare_data_supervised()
    
    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        """
        Validate hyperparameters and apply defaults for invalid values.
        
        Returns:
            Tuple[dict, bool]: (validated_params, is_valid)
                - validated_params: dict with valid hyperparameters
                - is_valid: False if any defaults were applied
        """
        is_valid = True
        params = {}
        
        # Validate n_estimators
        try:
            n_est = int(self.n_estimators_field.value)
            if n_est < 1:
                n_est = 100
                is_valid = False
            params['n_estimators'] = n_est
        except (ValueError, TypeError):
            params['n_estimators'] = 100
            is_valid = False
        
        # Validate max_depth
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
        
        # Validate min_samples_split
        try:
            min_samples = int(self.min_samples_split_field.value)
            if min_samples < 2:
                min_samples = 2
                is_valid = False
            params['min_samples_split'] = min_samples
        except (ValueError, TypeError):
            params['min_samples_split'] = 2
            is_valid = False

        # validate min_samples_leaf
        try:
            min_samples_leaf = int(self.min_samples_leaf.value)
            if min_samples_leaf < 1:
                min_samples_leaf = 1
                is_valid = False
            params['min_samples_leaf'] = min_samples_leaf
        except (ValueError, TypeError):
            params['min_samples_leaf'] = 1
            is_valid = False

        # Validate min_weight_fraction_leaf
        try:
            min_weight_fraction_leaf = float(self.min_weight_fraction_leaf.value)
            if min_weight_fraction_leaf < 0 or min_weight_fraction_leaf >= 0.5:
                min_weight_fraction_leaf = 0
                is_valid = False
            params['min_weight_fraction_leaf'] = min_weight_fraction_leaf
        except (ValueError, TypeError):
            params['min_weight_fraction_leaf'] = 0
            is_valid = False

        # validate max_features
        max_features_val = self.max_features_dropdown.value
        if max_features_val in ['sqrt', 'log2']:
            params['max_features'] = max_features_val
        elif max_features_val == 'None':
            params['max_features'] = None
        else:
            params['max_features'] = 'sqrt' # default
            is_valid = False
        
        # validate max_leaf_nodes
        try:
            max_leaf_nodes_val = self.max_leaf_nodes_field.value
            if max_leaf_nodes_val == 'None' or max_leaf_nodes_val == '':
                params['max_leaf_nodes'] = None
            else:
                max_leaf_nodes = int(max_leaf_nodes_val)
                if max_leaf_nodes < 2:
                    max_leaf_nodes = None
                    is_valid = False
                params['max_leaf_nodes'] = max_leaf_nodes
        except (ValueError, TypeError):
            params['max_leaf_nodes'] = None
            is_valid = False
        
        # validate min_impurity_decrease
        try:
            min_impurity_decrease = float(self.min_impurity_decrease_field.value)
            if min_impurity_decrease < 0:
                min_impurity_decrease = 0
                is_valid = False
            params['min_impurity_decrease'] = min_impurity_decrease
        except (ValueError, TypeError):
            params['min_impurity_decrease'] = 0
            is_valid = False

        return params, is_valid
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train random forest model and display evaluation results."""
        try:
            e.control.disabled = True
            self.parent.disable_model_selection()
            disable_navigation_bar(self.parent.page)
            self.parent.page.update()
            
            # Prepare data
            data = self._prepare_data()
            if data is None:
                enable_navigation_bar(self.parent.page)
                return
            
            X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols) = data
            
            # Validate hyperparameters
            hyperparams, params_valid = self._validate_hyperparameters()
            
            if not params_valid:
                self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)
            
            # Create model based on task type
            task_type = self._get_task_type()
            
            if task_type == "Classification":
                model = RandomForestClassifier(
                    n_estimators=hyperparams['n_estimators'],
                    criterion=self.criterion_dropdown.value,
                    max_depth=hyperparams['max_depth'],
                    min_samples_split=hyperparams['min_samples_split'],
                    min_samples_leaf=hyperparams['min_samples_leaf'],
                    min_weight_fraction_leaf=hyperparams['min_weight_fraction_leaf'],
                    max_features=hyperparams['max_features'],
                    max_leaf_nodes=hyperparams['max_leaf_nodes'],
                    min_impurity_decrease=hyperparams['min_impurity_decrease'],
                    bootstrap=self.bootstrap_switch.value,
                    oob_score=self.oob_score_switch.value,
                    random_state=42,
                    n_jobs=-1,
                )
            else:  # Regression
                model = RandomForestRegressor(
                    n_estimators=hyperparams['n_estimators'],
                    criterion=self.criterion_dropdown.value,
                    max_depth=hyperparams['max_depth'],
                    min_samples_split=hyperparams['min_samples_split'],
                    min_samples_leaf=hyperparams['min_samples_leaf'],
                    min_weight_fraction_leaf=hyperparams['min_weight_fraction_leaf'],
                    max_features=hyperparams['max_features'],
                    max_leaf_nodes=hyperparams['max_leaf_nodes'],
                    min_impurity_decrease=hyperparams['min_impurity_decrease'],
                    bootstrap=self.bootstrap_switch.value,
                    oob_score=self.oob_score_switch.value,
                    random_state=42,
                    n_jobs=-1,
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
            
            # Calculate metrics using centralized utility
            if task_type == "Classification":
                metrics_dict = calculate_classification_metrics(y_test, y_pred)
                metrics_dict["CV"] = cv_results
                result_text = format_results_markdown(metrics_dict, task_type="classification")
            else:
                metrics_dict = calculate_regression_metrics(y_test, y_pred)
                metrics_dict["CV"] = cv_results
                result_text = format_results_markdown(metrics_dict, task_type="regression")
            
            # Add feature importance if available
            importance = get_feature_importance(model, self.df.columns.tolist())
            if importance:
                result_text += "\n\n**Feature Importance:**\n\n"
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feature, imp_value in sorted_importance:
                    result_text += f"- {feature}: {imp_value:.4f}\n"
            
            # Display results dialog with copy button
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                f"Random Forest {task_type} Results",
                result_text,
                "Random Forest"
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
    
    def _bootstrap_switch_on_change(self, e: ft.ControlEvent) -> None:
        if self.bootstrap_switch.value:
            self.oob_score_switch.disabled = False
        else:
            self.oob_score_switch.value = False
            self.oob_score_switch.disabled = True
        self.parent.page.update()

    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for random forest hyperparameter configuration."""
        
        self.n_estimators_field = ft.TextField(
            label="Number of Trees",
            value="100",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The number of trees in the forest. More trees can improve performance but increase computation time.",
        )
        
        if self._get_task_type() == "Classification":
            criterion_options = ['gini', 'entropy', 'log_loss'] # default: gini
        else:
            criterion_options = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'] # default: squared_error
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
            tooltip="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.",
        )
        
        self.min_samples_split_field = ft.TextField(
            label="Min Samples to Split",
            value="2",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The minimum number of samples required to split an internal node",
        )
        
        self.min_samples_leaf = ft.TextField(
            label="Min Samples in Leaf",
            value="1",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The minimum number of samples required to be at a leaf node.",
        )

        self.min_weight_fraction_leaf = ft.TextField(
            label="Min Weight Fraction Leaf",
            value="0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.",
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

        self.max_leaf_nodes_field = ft.TextField(
            label="Max Leaf Nodes",
            value="None",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            on_click=self._field_on_click,
            on_blur=self._field_on_blur,
            suffix_icon=ft.IconButton(ft.Icons.RESTART_ALT, on_click=lambda _: self._reset_field_to_none(self.max_leaf_nodes_field), tooltip="Reset to None"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.",
        )

        self.min_impurity_decrease_field = ft.TextField(
            label="Min Impurity Decrease",
            value="0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
        )

        self.bootstrap_switch = ft.Switch(
            label="Bootstrap",
            value=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            on_change=self._bootstrap_switch_on_change,
            tooltip="Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.",
        )

        self.oob_score_switch = ft.Switch(
            label="OOB Score",
            value=False,
            label_style=ft.TextStyle(font_family="SF regular"),
            label_position=ft.LabelPosition.LEFT,
            tooltip="Whether to use out-of-bag samples to estimate the generalization score. By default, accuracy_score is used. Provide a callable with signature metric(y_true, y_pred) to use a custom metric. Only available if bootstrap=True.",
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
                                ft.Text("Random Forest",
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
                        ft.Row([self.min_samples_split_field, self.min_samples_leaf]),
                        ft.Row([self.min_weight_fraction_leaf, self.max_features_dropdown]),
                        ft.Row([self.max_leaf_nodes_field, self.min_impurity_decrease_field]),
                        ft.Row([self.bootstrap_switch, self.oob_score_switch], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
