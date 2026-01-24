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
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from utils.model_utils import (
    calculate_classification_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from core.data_preparation import prepare_data_for_training, prepare_data_for_training_no_split

if TYPE_CHECKING:
    from ..model_factory import ModelFactory


@dataclass
class KNNModel:
    """K-Nearest Neighbors classification model."""
    
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
        
        # Validate metric
        try:
            metric = self.metric_dropdown.value
            valid_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
            if metric not in valid_metrics:
                metric = 'euclidean'
                is_valid = False
            params['metric'] = metric
        except (ValueError, TypeError):
            params['metric'] = 'euclidean'
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
                self.parent.page.open(ft.SnackBar(
                    ft.Text("Invalid hyperparameters. Using default values.", font_family="SF regular"),
                    bgcolor="#FF9800"
                ))
            
            model = KNeighborsClassifier(
                n_neighbors=hyperparams['n_neighbors'],
                weights=hyperparams['weights'],
                algorithm=hyperparams['algorithm'],
                metric=hyperparams['metric'],
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
            self.parent.page.open(ft.SnackBar(
                ft.Text(f"Training failed: {str(e)}", font_family="SF regular")
            ))
        
        finally:
            self.parent.enable_model_selection()
            self.train_btn.disabled = False
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
            options=[
                ft.DropdownOption("uniform", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("distance", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="Weight function. uniform=all neighbors equally, distance=closer neighbors weighted higher",
        )
        
        self.algorithm_dropdown = ft.Dropdown(
            label="Algorithm",
            value="auto",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("auto", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("ball_tree", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("kd_tree", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("brute", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="Algorithm to compute neighbors. auto=auto-select, ball_tree=scalable, kd_tree=fast, brute=accurate",
        )
        
        self.metric_dropdown = ft.Dropdown(
            label="Distance Metric",
            value="euclidean",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("euclidean", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("manhattan", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("minkowski", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="Distance metric. euclidean=straight-line, manhattan=grid-based, minkowski=general",
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
                        ft.Row([self.weights_dropdown, self.algorithm_dropdown]),
                        self.metric_dropdown,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
