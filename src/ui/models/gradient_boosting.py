"""
Gradient Boosting Classification and Regression Model

Boosted ensemble of weak learners supporting both:
- Classification: GradientBoostingClassifier
- Regression: GradientBoostingRegressor

Configurable hyperparameters:
- n_estimators: Number of boosting stages
- learning_rate: Step shrinkage (eta)
- max_depth: Maximum tree depth
- min_samples_split: Minimum samples to split node
- subsample: Fraction of samples for fitting
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from utils.model_utils import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    format_results_markdown,
    get_feature_importance,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from core.data_preparation import prepare_data_for_training

if TYPE_CHECKING:
    from ..model_factory import ModelFactory


@dataclass
class GradientBoostingModel:
    """Gradient Boosting model supporting both classification and regression."""
    
    parent: ModelFactory
    df: DataFrame
    
    def __post_init__(self):
        """Ensure dataset is copied to avoid mutations."""
        self.df = self.df.copy()
    
    def _get_task_type(self) -> str:
        """Determine if this is classification or regression task."""
        task_type = self.parent.task_type_dropdown.value
        return task_type if task_type in ["Classification", "Regression"] else "Classification"
    
    def _prepare_data(self) -> Optional[Tuple]:
        """
        Prepare data for training using spec-compliant categorical encoding.
        
        Uses prepare_data_for_training() which:
        - Performs train-test split BEFORE encoding (prevents data leakage)
        - Fits encoders ONLY on training data
        - Applies encoders to test data
        - Returns encoding metadata and cardinality warnings
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, categorical_cols, numeric_cols,
                      encoders, warnings) or None if error
        """
        try:
            target_name = self.parent.target_column_dropdown.value
            test_size = self._validate_test_size()
            
            # Call spec-compliant data preparation
            (
                X_train,
                X_test,
                y_train,
                y_test,
                categorical_cols,
                numeric_cols,
                encoders,
                cardinality_warnings,
            ) = prepare_data_for_training(
                self.df.copy(),
                target_col=target_name,
                test_size=test_size,
                random_state=42,
                raise_on_unseen=True,
            )
            
            # Store encoding metadata for later use
            self.categorical_cols = categorical_cols
            self.numeric_cols = numeric_cols
            self.encoders = encoders
            self.cardinality_warnings = cardinality_warnings
            
            # Warn user about high-cardinality columns if any
            if cardinality_warnings:
                warning_msgs = [
                    f"{col}: {w.message}"
                    for col, w in cardinality_warnings.items()
                ]
                self.parent.page.open(ft.SnackBar(
                    ft.Text(
                        "Cardinality warnings: " + "; ".join(warning_msgs),
                        font_family="SF regular",
                        color=ft.Colors.ORANGE
                    )
                ))
            
            # Return tuple for backward compatibility with train method
            return X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols)
        
        except Exception as e:
            self.parent.page.open(ft.SnackBar(
                ft.Text(f"Data preparation error: {str(e)}", font_family="SF regular")
            ))
            return None
    
    def _build_preprocessor(self, categorical_cols: list, numeric_cols: list):
        """Build preprocessing pipeline for features."""
        preprocessors = []
        
        if numeric_cols:
            if self.parent.scaler_dropdown.value == "standard_scaler":
                scaler = StandardScaler()
            elif self.parent.scaler_dropdown.value == "minmax_scaler":
                scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                scaler = FunctionTransformer(validate=False)
            
            preprocessors.append(('numeric', scaler, numeric_cols))
        
        if categorical_cols:
            preprocessors.append((
                'categorical',
                OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                categorical_cols
            ))
        
        if preprocessors:
            return ColumnTransformer(preprocessors, remainder='passthrough')
        else:
            return FunctionTransformer(validate=False)
    
    def _validate_test_size(self) -> float:
        """
        Validate and return test_size value.
        
        Returns:
            float: Valid test_size between 0.1 and 0.5, default 0.2 if invalid
        """
        try:
            test_size = float(self.parent.test_size_field.value)
            if test_size < 0.1 or test_size > 0.5:
                return 0.2
            return test_size
        except (ValueError, TypeError, AttributeError):
            return 0.2
    
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
            if n_est < 1 or n_est > 1000:
                n_est = 100
                is_valid = False
            params['n_estimators'] = n_est
        except (ValueError, TypeError):
            params['n_estimators'] = 100
            is_valid = False
        
        # Validate learning_rate
        try:
            lr = float(self.learning_rate_field.value)
            if lr <= 0 or lr > 1:
                lr = 0.1
                is_valid = False
            params['learning_rate'] = lr
        except (ValueError, TypeError):
            params['learning_rate'] = 0.1
            is_valid = False
        
        # Validate max_depth
        try:
            max_depth_val = self.max_depth_field.value
            if max_depth_val == 'None' or max_depth_val == '':
                params['max_depth'] = None
            else:
                max_depth = int(max_depth_val)
                if max_depth < 1 or max_depth > 100:
                    max_depth = 3
                    is_valid = False
                params['max_depth'] = max_depth
        except (ValueError, TypeError):
            params['max_depth'] = 3
            is_valid = False
        
        # Validate min_samples_split
        try:
            min_samples = int(self.min_samples_split_field.value)
            if min_samples < 2 or min_samples > 50:
                min_samples = 2
                is_valid = False
            params['min_samples_split'] = min_samples
        except (ValueError, TypeError):
            params['min_samples_split'] = 2
            is_valid = False
        
        # Validate subsample
        try:
            subsample = float(self.subsample_field.value)
            if subsample <= 0 or subsample > 1:
                subsample = 1.0
                is_valid = False
            params['subsample'] = subsample
        except (ValueError, TypeError):
            params['subsample'] = 1.0
            is_valid = False
        
        return params, is_valid
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train gradient boosting model and display evaluation results."""
        try:
            e.control.disabled = True
            self.parent.disable_model_selection()
            disable_navigation_bar(self.parent.page)
            self.parent.page.update()
            
            data = self._prepare_data()
            if data is None:
                enable_navigation_bar(self.parent.page)
                return
            
            X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols) = data
            
            # Validate hyperparameters
            hyperparams, params_valid = self._validate_hyperparameters()
            
            if not params_valid:
                self.parent.page.open(ft.SnackBar(
                    ft.Text("Invalid hyperparameters. Using default values.", font_family="SF regular"),
                    bgcolor="#FF9800"  # Orange for warning
                ))
            
            # Create model based on task type
            task_type = self._get_task_type()
            
            if task_type == "Classification":
                model = GradientBoostingClassifier(
                    n_estimators=hyperparams['n_estimators'],
                    learning_rate=hyperparams['learning_rate'],
                    max_depth=hyperparams['max_depth'],
                    min_samples_split=hyperparams['min_samples_split'],
                    subsample=hyperparams['subsample'],
                    random_state=42,
                )
            else:  # Regression
                model = GradientBoostingRegressor(
                    n_estimators=hyperparams['n_estimators'],
                    learning_rate=hyperparams['learning_rate'],
                    max_depth=hyperparams['max_depth'],
                    min_samples_split=hyperparams['min_samples_split'],
                    subsample=hyperparams['subsample'],
                    random_state=42,
                )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if task_type == "Classification":
                metrics_dict = calculate_classification_metrics(y_test, y_pred)
                result_text = format_results_markdown(metrics_dict, task_type="classification")
            else:
                metrics_dict = calculate_regression_metrics(y_test, y_pred)
                result_text = format_results_markdown(metrics_dict, task_type="regression")
            
            # Add feature importance
            importance = get_feature_importance(model, feature_cols)
            if importance:
                result_text += "\n\n**Feature Importance:**\n\n"
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feature, imp_value in sorted_importance:
                    result_text += f"- {feature}: {imp_value:.4f}\n"
            
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                f"Gradient Boosting {task_type} Results",
                result_text,
                "Gradient Boosting"
            )
            self.parent.page.open(evaluation_dialog)
            enable_navigation_bar(self.parent.page)
        
        except Exception as e:
            enable_navigation_bar(self.parent.page)
            self.parent.page.open(ft.SnackBar(
                ft.Text(f"Training failed: {str(e)}", font_family="SF regular")
            ))
        
        finally:
            self.parent.enable_model_selection()
            self.train_btn.disabled = False
            self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for gradient boosting hyperparameter configuration."""
        
        self.n_estimators_field = ft.TextField(
            label="Number of Stages",
            value="100",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Number of boosting stages (iterations). Range: 1-500",
        )
        
        self.learning_rate_field = ft.TextField(
            label="Learning Rate",
            value="0.1",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Step shrinkage (eta). Lower values require more iterations but often better generalization. Range: 0.0001-1.0",
        )
        
        self.max_depth_field = ft.TextField(
            label="Max Depth",
            value="3",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Maximum depth of weak learners. Typical values 3-10. Range: 1-50",
        )
        
        self.min_samples_split_field = ft.TextField(
            label="Min Samples to Split",
            value="2",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Minimum samples required to split a node. Range: 2-100",
        )
        
        self.subsample_field = ft.TextField(
            label="Subsample",
            value="1.0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Fraction of samples used for fitting base learners. <1.0 improves generalization. Range: 0.1-1.0",
        )
        
        self.train_btn = ft.FilledButton(
            text="Train and evaluate model",
            icon=ft.Icons.PSYCHOLOGY,
            on_click=self._train_and_evaluate_model,
            expand=1,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        
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
                                ft.Text("Gradient Boosting",
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
                            size=14
                        ),
                        self.n_estimators_field,
                        ft.Row([self.learning_rate_field, self.max_depth_field]),
                        ft.Row([self.min_samples_split_field, self.subsample_field]),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
