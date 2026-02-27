from __future__ import annotations
from typing import List, Literal, Tuple, Optional
import flet as ft
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.pipeline import Pipeline, make_pipeline

from utils.model_utils import (
    calculate_regression_metrics, format_results_markdown, create_results_dialog,
    disable_navigation_bar, enable_navigation_bar,
)
from .base_model import BaseModel
    

@dataclass
class LinearRegressionModel(BaseModel):
    """Linear Regression model with sklearn Pipeline for reproducible preprocessing."""

    def _prepare_data(self):
        """Prepare data for training."""
        return self._prepare_data_supervised()
    
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
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)
        
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
        self._build_train_button()
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
        