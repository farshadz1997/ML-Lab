"""
Logistic Regression Classification Model

Multinomial/binary logistic regression classifier with configurable hyperparameters:
- C: Inverse of regularization strength
- max_iter: Maximum iterations for solver
- solver: Optimization algorithm
- class_weight: Strategy for handling class imbalance
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from utils.model_utils import (
    check_data_quality,
    calculate_classification_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from core.data_preparation import prepare_data_for_training

if TYPE_CHECKING:
    from ..model_factory import ModelFactory


@dataclass
class LogisticRegressionModel:
    """Logistic Regression classifier with configurable hyperparameters."""
    
    parent: ModelFactory
    df: DataFrame
    
    def __post_init__(self):
        """Ensure dataset is copied to avoid mutations."""
        self.df = self.df.copy()
    
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
        
        # Handle numeric columns with scaling
        if numeric_cols:
            if self.parent.scaler_dropdown.value == "standard_scaler":
                scaler = StandardScaler()
            elif self.parent.scaler_dropdown.value == "minmax_scaler":
                scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                scaler = FunctionTransformer(validate=False)  # No scaling
            
            preprocessors.append(('numeric', scaler, numeric_cols))
        
        # Handle categorical columns with one-hot encoding
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
        Validate test_size input and return safe default if invalid.
        
        Returns:
            float: Valid test_size between 0.1 and 0.5
        """
        try:
            test_size = float(self.parent.test_size_field.value)
            # Clamp to safe range
            if test_size < 0.1:
                return 0.2  # Default
            elif test_size > 0.5:
                return 0.2  # Default
            return test_size
        except (ValueError, TypeError):
            return 0.2  # Default if parsing fails
    
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
        
        # Validate C
        try:
            c_value = float(self.C_field.value)
            if c_value <= 0 or c_value > 1000:
                c_value = 1.0
                is_valid = False
            params['C'] = c_value
        except (ValueError, TypeError):
            params['C'] = 1.0
            is_valid = False
        
        # Validate max_iter
        try:
            max_iter_value = int(self.max_iter_field.value)
            if max_iter_value < 1 or max_iter_value > 10000:
                max_iter_value = 100
                is_valid = False
            params['max_iter'] = max_iter_value
        except (ValueError, TypeError):
            params['max_iter'] = 100
            is_valid = False
        
        # Validate solver
        valid_solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
        solver_value = self.solver_dropdown.value
        if solver_value not in valid_solvers:
            solver_value = 'lbfgs'
            is_valid = False
        params['solver'] = solver_value
        
        # Validate class_weight
        class_weight_value = self.class_weight_dropdown.value
        if class_weight_value == 'None':
            class_weight_value = None
        elif class_weight_value not in ['balanced', None]:
            class_weight_value = None
            is_valid = False
        params['class_weight'] = class_weight_value
        
        return params, is_valid
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train logistic regression model and display evaluation results."""
        try:
            e.control.disabled = True
            self.parent.disable_model_selection()
            disable_navigation_bar(self.parent.page)
            self.parent.page.update()
            
            # Check data quality first
            is_valid, error_msg = check_data_quality(self.df, self.parent.target_column_dropdown.value)
            if not is_valid:
                self.parent.page.open(ft.SnackBar(
                    ft.Text(f"Data error: {error_msg}", font_family="SF regular")
                ))
                enable_navigation_bar(self.parent.page)
                return
            
            # Prepare data
            data = self._prepare_data()
            if data is None:
                enable_navigation_bar(self.parent.page)
                return
            
            X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols) = data
            
            # Validate and get hyperparameters with defaults for invalid inputs
            hyperparams, params_valid = self._validate_hyperparameters()
            
            # If invalid params were detected, inform user
            if not params_valid:
                self.parent.page.open(ft.SnackBar(
                    ft.Text(
                        "Some hyperparameters were invalid. Using defaults.",
                        font_family="SF regular",
                        color=ft.Colors.ORANGE
                    )
                ))
            
            # Create and train model with validated parameters
            model = LogisticRegression(
                C=hyperparams['C'],
                max_iter=hyperparams['max_iter'],
                solver=hyperparams['solver'],
                class_weight=hyperparams['class_weight'],
                random_state=42,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics using centralized utility
            metrics_dict = calculate_classification_metrics(y_test, y_pred)
            result_text = format_results_markdown(metrics_dict, task_type="classification")
            
            # Display results dialog with copy button
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Logistic Regression Classification Results",
                result_text,
                "Logistic Regression"
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
        """Build Flet UI card for logistic regression hyperparameter configuration."""
        
        # Create hyperparameter controls
        self.C_field = ft.TextField(
            label="C (Regularization)",
            value="1.0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Inverse of regularization strength; lower values indicate stronger regularization. Range: 0.001 to 100",
        )
        
        self.max_iter_field = ft.TextField(
            label="Max Iterations",
            value="100",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Maximum number of iterations for solver convergence. Range: 50 to 5000",
        )
        
        self.solver_dropdown = ft.Dropdown(
            label="Solver",
            value="lbfgs",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("lbfgs", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("liblinear", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("saga", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("newton-cg", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="Algorithm to use in the optimization problem. lbfgs=default for multiclass, liblinear=faster for binary",
        )
        
        self.class_weight_dropdown = ft.Dropdown(
            label="Class Weight",
            value="None",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("balanced", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="'balanced' automatically adjusts weights inversely proportional to class frequency. Use for imbalanced datasets",
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
                                ft.Text("Logistic Regression",
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
                        ft.Row([self.C_field, self.max_iter_field]),
                        self.solver_dropdown,
                        self.class_weight_dropdown,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
