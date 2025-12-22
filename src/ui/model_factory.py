from __future__ import annotations
import logging
from pathlib import Path
import flet as ft
import pandas as pd
import numpy as np
from typing import List, Literal, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from helpers import resource_path
from .models import LinearRegressionModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import json

if TYPE_CHECKING:
    from .layout import AppLayout

MODEL_TYPE = Literal["Classification", "Regression"]
CLASSIFIER = Literal["logistic_regression", "random_forest", "gradient_boosting", "svm"]
REGRESSOR = Literal["linear_regression", "random_forest", "gradient_boosting", "svm"]
MODELS = Literal["linear_regression", "logistic_regression", "random_forest", "gradient_boosting", "svm", "knn"]

CLASSIFICATION_MODELS_OPTIONS = [
    ft.DropdownOption("logistic_regression", text="Logistic regression", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("random_forest", text="Random forest", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("gradient_boosting", text="Gradient boosting", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("svm", text="SVM", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("knn", text="KNN", text_style=ft.TextStyle(font_family="SF regular")),
]
REGRESSION_MODELS_OPTIONS = [
    ft.DropdownOption("linear_regression", text="Linear regression", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("random_forest", text="Random forest", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("gradient_boosting", text="Gradient boosting", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("svm", text="SVM", text_style=ft.TextStyle(font_family="SF regular")),
]
CLUSTERING_MODELS_OPTIONS = [
    ft.DropdownOption("kmeans", text="K-Means", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("minibatch-kmeans", text="MiniBatch K-Means", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("hierarchical", text="Hierarchical Clustering", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("dbscan", text="DBSCAN", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("hdbscan", text="HDBSCAN", text_style=ft.TextStyle(font_family="SF regular")),
]

@dataclass
class ModelFactory:
    parent: AppLayout
    page: ft.Page
    column: ft.Column | None = None
    model = None
    scaler = None
    label_encoders = {}
    
    def _learning_type_on_change(self, e: ft.ControlEvent) -> None:
        learning_type = e.control.value
        if learning_type == "Supervised":
            self.task_type_dropdown.options = [
                ft.DropdownOption("Classification", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("Regression", text_style=ft.TextStyle(font_family="SF regular")),
            ]
            self.task_type_dropdown.value = "Classification"
            self.task_type_dropdown.disabled = False
            self.model_dropdown.options = CLASSIFICATION_MODELS_OPTIONS
            self.target_column_dropdown.disabled = False
            self.test_size_field.disabled = False
        else:
            self.task_type_dropdown.options = [ft.DropdownOption("Clustering", text_style=ft.TextStyle(font_family="SF regular"))]
            self.task_type_dropdown.value = "Clustering"
            self.task_type_dropdown.disabled = True
            self.model_dropdown.options = CLUSTERING_MODELS_OPTIONS
            self.target_column_dropdown.disabled = True
            self.test_size_field.disabled = True
        self.model_dropdown.value = self.model_dropdown.options[0].key
        self.page.update()
    
    def _on_task_change(self, e: ft.ControlEvent) -> None:
        """Handle task type change (classification/regression)"""
        task_type = e.control.value
        self._update_model_options(task_type)
    
    def _update_model_options(self, task_type: MODEL_TYPE) -> None:
        """Update available models based on task type"""
        if task_type == "Classification":
            self.model_dropdown.options = CLASSIFICATION_MODELS_OPTIONS
        else:
            self.model_dropdown.options = REGRESSION_MODELS_OPTIONS
        self.model_dropdown.value = self.model_dropdown.options[0].key
        self.page.update()
        
    def _model_on_change(self, model_name: MODELS):
        if self.config_card in self.column.controls[2].controls:
            self.column.controls[2].controls.remove(self.config_card)
        if model_name == "linear_regression":
            model = LinearRegressionModel(self, self.parent.dataset.df)
            self.config_card = model.build_model_control()
        elif model_name == "logistic_regression":
            pass
        elif model_name == "gradient_boosting":
            pass
        elif model_name == "random_forest":
            pass
        elif model_name == "svm":
            pass
        elif model_name == "knn":
            pass
        self.column.controls[2].controls.append(self.config_card)
        self.page.update()
    
    def build_controls(self) -> ft.Column:
        numeric_cols = self.parent.dataset.df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = self.parent.dataset.df.columns.tolist()
        
        self.learning_type_dropdown = ft.Dropdown(
            value="Supervised",
            label="Learning type",
            label_style=ft.TextStyle(font_family="SF regular"),
            expand=1,
            options=[
                ft.DropdownOption("Supervised", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("Unsupervised", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            on_change=self._learning_type_on_change,
        )
        
        self.task_type_dropdown = ft.Dropdown(
            value="Classification",
            label="Task Type",
            label_style=ft.TextStyle(font_family="SF regular"),
            expand=1,
            options=[
                ft.DropdownOption("Classification", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("Regression", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            on_change=self._on_task_change,
        )
        
        self.model_dropdown = ft.Dropdown(
            value="logistic_regression",
            label="Model Type",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=CLASSIFICATION_MODELS_OPTIONS,
            on_change=lambda e: self._model_on_change(e.control.value)
        )
        
        self.target_column_dropdown = ft.Dropdown(
            label="Target Column",
            label_style=ft.TextStyle(font_family="SF regular"),
            expand=1,
            options=[
                ft.DropdownOption(col, text_style=ft.TextStyle(font_family="SF regular"))
                for col in all_cols
            ]
        )
        self.target_column_dropdown.value = self.target_column_dropdown.options[0].key
        
        self.test_size_field = ft.TextField(
            label="Test Size (0.0-1.0)",
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            value="0.2",
            expand=1
        )
        
        self.scaler_dropdown = ft.Dropdown(
            value="standard_scaler",
            label="Feature Scaler",
            label_style=ft.TextStyle(font_family="SF regular"),
            expand=1,
            options=[
                ft.DropdownOption("standard_scaler", text="Standard Scaler", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("minmax_scaler", text="Min-Max Scaler", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("none", text="No Scaling", text_style=ft.TextStyle(font_family="SF regular")),
            ]
        )
        
        self.config_card = LinearRegressionModel(self, self.parent.dataset.df).build_model_control()
        self.column = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.START,
            alignment=ft.MainAxisAlignment.START,
            controls=[
                ft.Row([ft.Text("Model Factory", expand=False, size=30, font_family="SF thin", text_align="center")], alignment=ft.MainAxisAlignment.CENTER),
                ft.Divider(),
                ft.Row(
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        ft.Card(
                            expand=2,
                            content=ft.Container(
                                margin=ft.margin.all(15),
                                content=ft.Column(
                                    spacing=15,
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                    alignment=ft.MainAxisAlignment.START,
                                    controls=[
                                        ft.Text("Model Configuration", font_family="SF thin", size=24),
                                        ft.Divider(),
                                        ft.Row(
                                            controls=[
                                                ft.Column(
                                                    expand=1,
                                                    controls=[
                                                        ft.Text("ML Pipeline", font_family="SF regular", weight="bold", size=14),
                                                        self.learning_type_dropdown,
                                                        self.task_type_dropdown,
                                                        self.model_dropdown,
                                                    ]
                                                ),
                                                ft.VerticalDivider(),
                                                ft.Column(
                                                    expand=1,
                                                    controls=[
                                                        ft.Text("Data Configuration", font_family="SF regular", weight="bold", size=14),
                                                        self.target_column_dropdown,
                                                        self.test_size_field,
                                                        self.scaler_dropdown,
                                                    ]
                                                ),
                                            ]
                                        ),
                                        ft.Row(
                                            alignment=ft.MainAxisAlignment.START,
                                            controls=[
                                                ft.Text(
                                                    "Note: Categorical features will be automatically encoded. Data is normalized using selected scaler.",
                                                    font_family="SF light",
                                                    size=11,
                                                    color=ft.Colors.GREY_700
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            )
                        ),
                        self.config_card
                    ]
                ),
            ]
        )
        return self.column