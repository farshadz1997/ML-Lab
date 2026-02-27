"""
K-Nearest Neighbors (KNN) Classification Model

Instance-based classifier that stores training instances and classifies
based on nearest neighbors in feature space.

Configurable hyperparameters:
- n_neighbors: Number of neighbors to consider
- weights: Weight function ('uniform' or 'distance')
- algorithm: Algorithm to compute nearest neighbors
- metric: Distance metric to use
"""

from __future__ import annotations
from typing import Optional, Tuple
import flet as ft
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier

from utils.model_utils import (
    calculate_classification_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel


@dataclass
class KNNModel(BaseModel):
    """K-Nearest Neighbors classification model."""

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
        
        # Validate n_neighbors
        try:
            n_neighbors = int(self.n_neighbors_field.value)
            if n_neighbors < 1 or n_neighbors > 100:
                n_neighbors = 5
                is_valid = False
            params['n_neighbors'] = n_neighbors
        except (ValueError, TypeError):
            params['n_neighbors'] = 5
            is_valid = False
        
        # Validate weights
        try:
            weights = self.weights_dropdown.value
            valid_weights = ['uniform', 'distance']
            if weights not in valid_weights:
                weights = 'uniform'
                is_valid = False
            params['weights'] = weights
        except (ValueError, TypeError):
            params['weights'] = 'uniform'
            is_valid = False
        
        # Validate algorithm
        try:
            algorithm = self.algorithm_dropdown.value
            valid_algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
            if algorithm not in valid_algorithms:
                algorithm = 'auto'
                is_valid = False
            params['algorithm'] = algorithm
        except (ValueError, TypeError):
            params['algorithm'] = 'auto'
            is_valid = False
            
        # Validate leaf size
        try:
            leaf_size = int(self.leaf_size_field.value)
            if leaf_size < 1:
                leaf_size = 30
                is_valid = False
            params['leaf_size'] = leaf_size
        except (ValueError, TypeError):
            params['leaf_size'] = 30
            is_valid = False
        
        return params, is_valid
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train KNN model and display evaluation results."""
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
                self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)
            
            model = KNeighborsClassifier(
                n_neighbors=hyperparams['n_neighbors'],
                weights=hyperparams['weights'],
                algorithm=hyperparams['algorithm'],
                metric=self.metric_dropdown.value,
                leaf_size=hyperparams['leaf_size']
            )
            
            model.fit(X_train, y_train.to_numpy())
            y_pred = model.predict(X_test)
            
            # Cross validation
            kf = KFold(
                n_splits=int(self.parent.n_split_slider.value),
                shuffle=self.parent.cross_val_shuffle_switch.value,
                random_state=42 if self.parent.cross_val_shuffle_switch.value else None
            )
            cv_results = cross_val_score(model, X_train, y_train, cv=kf)
            
            metrics_dict = calculate_classification_metrics(y_test, y_pred)
            metrics_dict["CV"] = cv_results
            result_text = format_results_markdown(metrics_dict, task_type="classification")
            
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "KNN Classification Results",
                result_text,
                "KNN"
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
    
    def _algorithm_change(self, e: ft.ControlEvent) -> None:
        """Show/hide leaf size field based on selected algorithm."""
        selected_algorithm = self.algorithm_dropdown.value
        if selected_algorithm in ['ball_tree', 'kd_tree']:
            self.leaf_size_field.visible = True
        else:
            self.leaf_size_field.visible = False
        self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for KNN hyperparameter configuration."""
        
        self.n_neighbors_field = ft.TextField(
            label="Number of Neighbors",
            value="5",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Number of nearest neighbors to consider. Lower=faster/more local, Higher=smoother. Range: 1-100",
        )
        
        self.weights_dropdown = ft.Dropdown(
            label="Weights",
            value="uniform",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("uniform"),
                ft.DropdownOption("distance"),
            ],
            tooltip="Weight function. uniform=all neighbors equally, distance=closer neighbors weighted higher",
        )
        
        self.algorithm_dropdown = ft.Dropdown(
            label="Algorithm",
            value="auto",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("auto"),
                ft.DropdownOption("ball_tree"),
                ft.DropdownOption("kd_tree"),
                ft.DropdownOption("brute"),
            ],
            tooltip="Algorithm to compute neighbors. auto=auto-select, ball_tree=scalable, kd_tree=fast, brute=accurate",
            on_change=self._algorithm_change
        )
        
        self.leaf_size_field = ft.TextField(
            label="Leaf Size",
            value="30",
            expand=1,
            visible=False,  # Hidden by default since it's only relevant for certain algorithms
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.",
        )
        
        self.metric_dropdown = ft.Dropdown(
            label="Distance Metric",
            value="euclidean",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("euclidean"),
                ft.DropdownOption("manhattan"),
                ft.DropdownOption("minkowski"),
                ft.DropdownOption("cityblock"),
                ft.DropdownOption("cosine"),
                ft.DropdownOption("haversine"),
                ft.DropdownOption("l1"),
                ft.DropdownOption("l2"),
                ft.DropdownOption("nan_euclidean"),
            ],
            tooltip="Distance metric. euclidean=straight-line, manhattan=grid-based, minkowski=general",
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
                                ft.Text("K-Nearest Neighbors",
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
                        self.n_neighbors_field,
                        ft.Row([self.weights_dropdown, self.metric_dropdown]),
                        ft.Row([self.algorithm_dropdown, self.leaf_size_field]),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
