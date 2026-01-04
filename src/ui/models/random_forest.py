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
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
class RandomForestModel:
    """Random Forest model supporting both classification and regression."""
    
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
                    ),
                    color="#FF9800"
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
            if min_samples < 2 or min_samples > 50:
                min_samples = 2
                is_valid = False
            params['min_samples_split'] = min_samples
        except (ValueError, TypeError):
            params['min_samples_split'] = 2
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
                self.parent.page.open(ft.SnackBar(
                    ft.Text("Invalid hyperparameters. Using default values.", font_family="SF regular"),
                    bgcolor="#FF9800"
                ))
            
            # Create model based on task type
            task_type = self._get_task_type()
            
            if task_type == "Classification":
                model = RandomForestClassifier(
                    n_estimators=hyperparams['n_estimators'],
                    max_depth=hyperparams['max_depth'],
                    min_samples_split=hyperparams['min_samples_split'],
                    random_state=42,
                    n_jobs=-1,
                )
            else:  # Regression
                model = RandomForestRegressor(
                    n_estimators=hyperparams['n_estimators'],
                    max_depth=hyperparams['max_depth'],
                    min_samples_split=hyperparams['min_samples_split'],
                    random_state=42,
                    n_jobs=-1,
                )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics using centralized utility
            if task_type == "Classification":
                metrics_dict = calculate_classification_metrics(y_test, y_pred)
                result_text = format_results_markdown(metrics_dict, task_type="classification")
            else:
                metrics_dict = calculate_regression_metrics(y_test, y_pred)
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
            self.parent.page.open(ft.SnackBar(
                ft.Text(f"Training failed: {str(e)}", font_family="SF regular")
            ))
        
        finally:
            self.parent.enable_model_selection()
            self.train_btn.disabled = False
            self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for random forest hyperparameter configuration."""
        
        self.n_estimators_field = ft.TextField(
            label="Number of Trees",
            value="100",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Number of trees in the forest. Higher values increase computational cost but often improve accuracy. Range: 1-1000",
        )
        
        self.max_depth_field = ft.TextField(
            label="Max Depth",
            value="None",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Maximum depth of trees. Unlimited if 'None'. Lower values prevent overfitting. Range: 1-100 or None",
        )
        
        self.min_samples_split_field = ft.TextField(
            label="Min Samples to Split",
            value="2",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Minimum samples required to split a node. Higher values prevent overfitting. Range: 2-100",
        )
        
        self.n_jobs_field = ft.TextField(
            label="Number of Jobs",
            value="-1",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Number of processors to use. -1 uses all processors. 1 disables parallelism",
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
                        self.n_estimators_field,
                        self.max_depth_field,
                        ft.Row([self.min_samples_split_field, self.n_jobs_field]),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
