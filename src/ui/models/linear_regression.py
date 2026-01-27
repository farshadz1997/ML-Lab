from __future__ import annotations
from typing import List, Literal, Tuple, TYPE_CHECKING, Optional
import flet as ft
from dataclasses import dataclass, field
from pandas import DataFrame
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

from utils.model_utils import (
    calculate_regression_metrics, format_results_markdown, create_results_dialog,
    disable_navigation_bar, enable_navigation_bar
)
from core.data_preparation import prepare_data_for_training, prepare_data_for_training_no_split

if TYPE_CHECKING:
    from ..model_factory import ModelFactory
    from ..layout import AppLayout
    

@dataclass
class LinearRegressionModel:
    """Linear Regression model with sklearn Pipeline for reproducible preprocessing."""
    
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
            test_size = self.parent.test_size_field.value / 100
            
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
                    bgcolor="#FF9800"
                ))
            
            # Return tuple for backward compatibility with train method
            return X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols)
        
        except Exception as e:
            self.parent.page.open(ft.SnackBar(ft.Text(f"Data preparation error: {str(e)}", font_family="SF regular")))
            return None
    
    def _build_pipeline(self, categorical_cols: list, numeric_cols: list) -> Pipeline:
        """
        Build sklearn Pipeline for data preprocessing.
        
        Handles both categorical and numeric features with appropriate transformers.
        """
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
            # Return identity transformer if no columns
            return FunctionTransformer(validate=False)
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train linear regression model and display evaluation results."""
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
            
            # Create and train model with current hyperparameters
            model = LinearRegression(
                fit_intercept=self.fit_intercept_switch.value,
                positive=self.positive_switch.value
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
            metrics_dict = calculate_regression_metrics(y_test, y_pred)
            metrics_dict["CV"] = cv_results
            
            # Add intercept to results
            result_text = f"**Model Intercept:** {model.intercept_:.4f}\n\n"
            result_text += format_results_markdown(metrics_dict, task_type="regression")
            
            # Display results dialog
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Linear Regression Results",
                result_text,
                "Linear Regression"
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
        self.fit_intercept_switch = ft.Switch(
            label="Fit intercept",
            tooltip="Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).",
            label_position=ft.LabelPosition.RIGHT,
            label_style=ft.TextStyle(font_family="SF regular"),
            value=True,
        )
        self.positive_switch = ft.Switch(
            label="Positive",
            tooltip="When set to True, forces the coefficients to be positive. This option is only supported for dense arrays",
            label_position=ft.LabelPosition.LEFT,
            label_style=ft.TextStyle(font_family="SF regular"),
            value=False,
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
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Row(
                            controls=[ft.Text("Linear regression", font_family="SF thin", size=24, text_align="center", expand=True)]
                        ),
                        ft.Divider(),
                        ft.Row([self.fit_intercept_switch, self.positive_switch], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
        