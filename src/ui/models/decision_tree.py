"""
Decision Tree Classification Model

Tree-based classifier that recursively partitions data based on feature values
to minimize impurity at each node.

Configurable hyperparameters:
- max_depth: Maximum depth of tree (controls complexity)
- min_samples_split: Minimum samples required to split internal node
- min_samples_leaf: Minimum samples required at leaf node
- criterion: Function to measure split quality ('gini', 'entropy', 'log_loss')
- splitter: Strategy for selecting splits ('best' or 'random')
- max_features: Number of features to consider when looking for best split
"""

from __future__ import annotations
from typing import Optional, Tuple
import flet as ft
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.model_utils import (
    check_data_quality,
    calculate_classification_metrics,
    calculate_regression_metrics,
    format_results_markdown,
    create_results_dialog,
    get_feature_importance,
)
from .base_model import BaseModel


@dataclass
class DecisionTreeModel(BaseModel):
    """Decision Tree classifier with configurable hyperparameters."""

    def _prepare_data(self):
        """Prepare data for training."""
        return self._prepare_data_supervised()
    
    def _validate_hyperparameters(self) -> tuple[dict, bool]:
        """
        Validate hyperparameter inputs with defaults for invalid values.
        
        Returns:
            Tuple of (hyperparams_dict, is_valid)
            - is_valid=True if all params are valid
            - is_valid=False if any defaults were used
        """
        is_valid = True
        params = {}
        
        # Validate max_depth
        try:
            max_depth_value = int(self.max_depth_field.value)
            if max_depth_value < 1 or max_depth_value > 50:
                max_depth_value = 10
                is_valid = False
            params['max_depth'] = max_depth_value
        except (ValueError, TypeError):
            params['max_depth'] = 10
            is_valid = False
        
        # Validate min_samples_split
        try:
            min_split_value = int(self.min_samples_split_field.value)
            if min_split_value < 2:
                min_split_value = 2
                is_valid = False
            params['min_samples_split'] = min_split_value
        except (ValueError, TypeError):
            params['min_samples_split'] = 2
            is_valid = False
        
        # Validate min_samples_leaf
        try:
            min_leaf_value = int(self.min_samples_leaf_field.value)
            if min_leaf_value < 1:
                min_leaf_value = 1
                is_valid = False
            params['min_samples_leaf'] = min_leaf_value
        except (ValueError, TypeError):
            params['min_samples_leaf'] = 1
            is_valid = False
        
        # Validate criterion
        task_type = self._get_task_type()
        if task_type == "Classification":
            valid_criteria = ['gini', 'entropy', 'log_loss']
        else:
            valid_criteria = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        criterion_value = self.criterion_dropdown.value
        if criterion_value not in valid_criteria:
            criterion_value = valid_criteria[0]  # Default to first valid criterion
            is_valid = False
        params['criterion'] = criterion_value
        
        # Validate splitter
        valid_splitters = ['best', 'random']
        splitter_value = self.splitter_dropdown.value
        if splitter_value not in valid_splitters:
            splitter_value = 'best'
            is_valid = False
        params['splitter'] = splitter_value
        
        # Validate max_features
        max_features_value = self.max_features_dropdown.value
        if max_features_value not in ['sqrt', 'log2', 'None']:
            max_features_value = 'None'
            is_valid = False
        params['max_features'] = None if max_features_value == 'None' else max_features_value
        
        return params, is_valid
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train decision tree model and display evaluation results."""
        try:
            e.control.disabled = True
            self.parent.disable_model_selection()
            self.parent.page.update()
            
            # Check data quality first
            is_valid, error_msg = check_data_quality(self.df, self.parent.target_column_dropdown.value)
            if not is_valid:
                self._show_snackbar(f"Data error: {error_msg}", bgcolor=ft.Colors.RED_500)
                return
            
            # Prepare data
            data = self._prepare_data()
            if data is None:
                return
            
            X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols) = data
            
            # Validate and get hyperparameters with defaults for invalid inputs
            hyperparams, params_valid = self._validate_hyperparameters()
            
            # If invalid params were detected, inform user
            if not params_valid:
                self._show_snackbar("Some hyperparameters were invalid. Using defaults.", bgcolor=ft.Colors.AMBER_ACCENT_200)
            
            # Create and train model with validated parameters
            task_type = self._get_task_type()
            if task_type == "Classification":
                model = DecisionTreeClassifier(
                    max_depth=hyperparams['max_depth'],
                    min_samples_split=hyperparams['min_samples_split'],
                    min_samples_leaf=hyperparams['min_samples_leaf'],
                    criterion=hyperparams['criterion'],
                    splitter=hyperparams['splitter'],
                    max_features=hyperparams['max_features'],
                    random_state=42,
                )
            else:
                model = DecisionTreeRegressor(
                    max_depth=hyperparams['max_depth'],
                    min_samples_split=hyperparams['min_samples_split'],
                    min_samples_leaf=hyperparams['min_samples_leaf'],
                    criterion=hyperparams['criterion'],
                    splitter=hyperparams['splitter'],
                    max_features=hyperparams['max_features'],
                    random_state=42,
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
            else:
                metrics_dict = calculate_regression_metrics(y_test, y_pred)
            metrics_dict["CV"] = cv_results
            
            # Get feature importance for decision tree
            feature_importance = get_feature_importance(model, self.df.columns.tolist())
            if feature_importance:
                metrics_dict['feature_importance'] = feature_importance
            
            result_text = format_results_markdown(metrics_dict, task_type=task_type.lower())
            
            # Display results dialog with copy button
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                f"Decision Tree {task_type} Results",
                result_text,
                "Decision Tree"
            )
            self.parent.page.open(evaluation_dialog)
        
        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)
        
        finally:
            self.parent.enable_model_selection()
            self.train_btn.disabled = False
            self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for decision tree hyperparameter configuration."""
        
        # Create hyperparameter controls
        self.max_depth_field = ft.TextField(
            label="Max Depth",
            value="5",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Maximum depth of tree. Range: 1 to 50. Lower values prevent overfitting",
        )
        
        self.min_samples_split_field = ft.TextField(
            label="Min Samples Split",
            value="2",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Minimum samples required to split node. Range: 2 to 20. Higher values reduce tree complexity",
        )
        
        self.min_samples_leaf_field = ft.TextField(
            label="Min Samples Leaf",
            value="1",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Minimum samples at leaf node. Range: 1 to 20. Higher values create smoother decision boundaries",
        )
        
        self.criterion_dropdown = ft.Dropdown(
            label="Criterion",
            value="gini",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("gini"),
                ft.DropdownOption("entropy"),
                ft.DropdownOption("log_loss"),
            ],
            tooltip="Function to measure split quality. gini=default, entropy=information gain based",
        )
        
        if self._get_task_type() == "Regression":
            self.criterion_dropdown.options = [
                ft.DropdownOption("squared_error"),
                ft.DropdownOption("friedman_mse"),
                ft.DropdownOption("absolute_error"),
                ft.DropdownOption("poisson"),
            ]
            self.criterion_dropdown.value = "squared_error"
            self.criterion_dropdown.tooltip = "Function to measure split quality. squared_error=default for regression"
        
        self.splitter_dropdown = ft.Dropdown(
            label="Splitter",
            value="best",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("best"),
                ft.DropdownOption("random"),
            ],
            tooltip="'best' searches all splits, 'random' uses random thresholds. best=slower but more accurate",
        )
        
        self.max_features_dropdown = ft.Dropdown(
            label="Max Features",
            value="None",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None"),
                ft.DropdownOption("sqrt"),
                ft.DropdownOption("log2"),
            ],
            tooltip="Number of features to consider for splits. sqrt/log2 reduce overfitting on high-dimensional data",
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
                                ft.Text("Decision Tree Classifier",
                                    font_family="SF thin",
                                    size=24,
                                    text_align="center",
                                    expand=True
                                )
                            ]
                        ),
                        ft.Divider(),
                        ft.Text("Hyperparameters",
                               font_family="SF regular",
                               weight="bold",
                               size=14),
                        ft.Row([self.max_depth_field, self.min_samples_split_field]),
                        ft.Row([self.min_samples_leaf_field, self.criterion_dropdown]),
                        ft.Row([self.splitter_dropdown, self.max_features_dropdown]),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
