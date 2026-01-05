from __future__ import annotations
import logging
from pathlib import Path
import flet as ft
import pandas as pd
import numpy as np
from typing import List, Literal, Any, TYPE_CHECKING, Type, Dict
from dataclasses import dataclass, field
from helpers import resource_path
from .models import (
    LinearRegressionModel,
    LogisticRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    SVMModel,
    KNNModel,
    DecisionTreeModel,
    DecisionTreeRegressorModel,
    KMeansModel,
    MiniBatchKMeansModel,
    HierarchicalClusteringModel,
    DBSCANModel,
    HDBSCANModel,
    GaussianMixtureModel,
    MeanShiftModel,
    AffinityPropagationModel,
    ElbowLocatorModel,
)
import json

if TYPE_CHECKING:
    from .layout import AppLayout

MODEL_TYPE = Literal["Classification", "Regression"]
CLASSIFIER = Literal["logistic_regression", "random_forest", "gradient_boosting", "svm", "knn", "decision_tree"]
REGRESSOR = Literal["linear_regression", "random_forest", "gradient_boosting", "svm", "decision_tree_regressor"]
CLUSTERER = Literal["kmeans", "minibatch_kmeans", "hierarchical", "dbscan", "hdbscan", "gaussian_mixture", "mean_shift", "affinity_propagation", "elbow_locator"]
MODELS = Literal["linear_regression", "logistic_regression", "random_forest", "gradient_boosting", "svm", "knn", "decision_tree", "decision_tree_regressor", "kmeans", "minibatch_kmeans", "hierarchical", "dbscan", "hdbscan", "gaussian_mixture", "mean_shift", "affinity_propagation", "elbow_locator"]

# Model registry for dynamic instantiation (Factory Pattern)
# Maps model name to model class; populated as new models are implemented
MODEL_REGISTRY: Dict[str, Type] = {
    'linear_regression': LinearRegressionModel,
    # Classification models
    'logistic_regression': LogisticRegressionModel,
    'random_forest': RandomForestModel,
    'gradient_boosting': GradientBoostingModel,
    'svm': SVMModel,
    'knn': KNNModel,
    'decision_tree': DecisionTreeModel,
    # Regression models
    'decision_tree_regressor': DecisionTreeRegressorModel,
    # Clustering models
    'kmeans': KMeansModel,
    'minibatch_kmeans': MiniBatchKMeansModel,
    'hierarchical': HierarchicalClusteringModel,
    'dbscan': DBSCANModel,
    'hdbscan': HDBSCANModel,
    'gaussian_mixture': GaussianMixtureModel,
    'mean_shift': MeanShiftModel,
    'affinity_propagation': AffinityPropagationModel,
    'elbow_locator': ElbowLocatorModel,
}

CLASSIFICATION_MODELS_OPTIONS = [
    ft.DropdownOption("logistic_regression", text="Logistic regression", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("random_forest", text="Random forest", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("gradient_boosting", text="Gradient boosting", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("svm", text="SVM", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("knn", text="KNN", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("decision_tree", text="Decision Tree", text_style=ft.TextStyle(font_family="SF regular")),
]
REGRESSION_MODELS_OPTIONS = [
    ft.DropdownOption("linear_regression", text="Linear regression", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("random_forest", text="Random forest", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("gradient_boosting", text="Gradient boosting", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("svm", text="SVM", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("decision_tree_regressor", text="Decision Tree", text_style=ft.TextStyle(font_family="SF regular")),
]
CLUSTERING_MODELS_OPTIONS = [
    ft.DropdownOption("kmeans", text="K-Means", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("minibatch_kmeans", text="MiniBatch K-Means", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("elbow_locator", text="Elbow Locator", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("hierarchical", text="Hierarchical Clustering", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("dbscan", text="DBSCAN", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("hdbscan", text="HDBSCAN", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("gaussian_mixture", text="Gaussian Mixture", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("mean_shift", text="Mean Shift", text_style=ft.TextStyle(font_family="SF regular")),
    ft.DropdownOption("affinity_propagation", text="Affinity Propagation", text_style=ft.TextStyle(font_family="SF regular")),
]

@dataclass
class ModelFactory:
    parent: AppLayout
    page: ft.Page
    column: ft.Column | None = None
    
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
            self._model_on_change("logistic_regression")
        else:
            self.task_type_dropdown.options = [ft.DropdownOption("Clustering", text_style=ft.TextStyle(font_family="SF regular"))]
            self.task_type_dropdown.value = "Clustering"
            self.task_type_dropdown.disabled = True
            self.model_dropdown.options = CLUSTERING_MODELS_OPTIONS
            self.target_column_dropdown.disabled = True
            self.test_size_field.disabled = True
            self._model_on_change("kmeans")
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
            self._model_on_change("logistic_regression")
        else:
            self.model_dropdown.options = REGRESSION_MODELS_OPTIONS
            self._model_on_change("linear_regression")
        self.model_dropdown.value = self.model_dropdown.options[0].key
        self.page.update()
        
    def _model_on_change(self, model_name: MODELS) -> None:
        """
        Load and display model-specific configuration card.
        
        Uses factory pattern to instantiate correct model class dynamically.
        Replaces previous config card with new one.
        
        Args:
            model_name: Name of selected model (e.g., "logistic_regression")
        """
        try:
            # Remove previous config card if present
            if (len(self.column.controls) > 2 and 
                self.config_card in self.column.controls[2].controls):
                self.column.controls[2].controls.remove(self.config_card)
            
            # Get model class from registry
            model_class = MODEL_REGISTRY.get(model_name)
            
            if model_class is None:
                # Model not yet implemented; show placeholder
                placeholder = ft.Card(
                    expand=1,
                    content=ft.Container(
                        expand=True,
                        margin=ft.margin.all(15),
                        content=ft.Column(
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            mainaxis_alignment=ft.MainAxisAlignment.CENTER,
                            controls=[
                                ft.Text(
                                    f"{model_name.replace('_', ' ').title()} - Coming Soon",
                                    font_family="SF thin",
                                    size=20,
                                    text_align="center"
                                ),
                                ft.Text(
                                    "This model is not yet implemented",
                                    font_family="SF regular",
                                    color=ft.Colors.GREY_700
                                )
                            ]
                        )
                    )
                )
                self.config_card = placeholder
            else:
                # Instantiate model and get configuration card
                model_instance = model_class(self, self.parent.dataset.df)
                self.config_card = model_instance.build_model_control()
            
            # Add new config card to column
            if len(self.column.controls) > 2:
                self.column.controls[2].controls.append(self.config_card)
            
            self.page.update()
        
        except Exception as e:
            self.page.open(ft.SnackBar(
                ft.Text(f"Error loading model: {str(e)}", font_family="SF regular")
            ))
            
    def disable_model_selection(self) -> None:
        self.learning_type_dropdown.disabled = True
        self.task_type_dropdown.disabled = True
        self.model_dropdown.disabled = True
        self.page.update()
            
    def enable_model_selection(self) -> None:
        self.learning_type_dropdown.disabled = False
        if self.task_type_dropdown.value != "Clustering":
            self.task_type_dropdown.disabled = False
        self.model_dropdown.disabled = False
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
                ft.DropdownOption(key=col, text=f"{col} ({self.parent.dataset.number_of_unique_values(col)})", text_style=ft.TextStyle(font_family="SF regular"))
                for col in all_cols
            ],
            tooltip="Number of unique values shown in parentheses",
        )
        self.target_column_dropdown.value = self.target_column_dropdown.options[0].key
        
        self.test_size_field = ft.Slider(
            value=20,
            min=10,
            max=50,
            divisions=40,
            label="{value}% Test Size",
            expand=4
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
        
        self.config_card = LogisticRegressionModel(self, self.parent.dataset.df).build_model_control()
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
                            expand=3,
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
                                                        ft.Row([ft.Text("Test Size", font_family="SF regular", expand=2), self.test_size_field]),
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