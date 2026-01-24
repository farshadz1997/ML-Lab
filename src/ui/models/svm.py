"""
Support Vector Machine (SVM) Classification and Regression Model

Powerful kernel-based model supporting both:
- Classification: SVC (Support Vector Classifier)
- Regression: SVR (Support Vector Regressor)

Configurable hyperparameters:
- C: Regularization parameter
- kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
- gamma: Kernel coefficient
- degree: Polynomial degree (for poly kernel)
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from utils.model_utils import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from core.data_preparation import prepare_data_for_training, prepare_data_for_training_no_split

if TYPE_CHECKING:
    from ..model_factory import ModelFactory


@dataclass
class SVMModel:
    """Support Vector Machine model supporting both classification and regression."""
    
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
        
        # Validate kernel
        try:
            kernel = self.kernel_dropdown.value
            valid_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            if kernel not in valid_kernels:
                kernel = 'rbf'
                is_valid = False
            params['kernel'] = kernel
        except (ValueError, TypeError):
            params['kernel'] = 'rbf'
            is_valid = False
        
        # Validate gamma
        try:
            gamma = self.gamma_field.value
            if gamma == 'scale' or gamma == 'auto':
                params['gamma'] = gamma
            else:
                gamma_float = float(gamma)
                if gamma_float <= 0 or gamma_float > 1:
                    gamma_float = 0.1
                    is_valid = False
                params['gamma'] = gamma_float
        except (ValueError, TypeError):
            params['gamma'] = 'scale'
            is_valid = False
        
        # Validate degree (for polynomial kernel)
        try:
            degree = int(self.degree_field.value)
            if degree < 1 or degree > 10:
                degree = 3
                is_valid = False
            params['degree'] = degree
        except (ValueError, TypeError):
            params['degree'] = 3
            is_valid = False
        
        return params, is_valid
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train SVM model and display evaluation results."""
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
            
            task_type = self._get_task_type()
            
            if task_type == "Classification":
                model = SVC(
                    C=hyperparams['C'],
                    kernel=hyperparams['kernel'],
                    gamma=hyperparams['gamma'],
                    degree=hyperparams['degree'],
                    random_state=42,
                )
            else:  # Regression
                model = SVR(
                    C=hyperparams['C'],
                    kernel=hyperparams['kernel'],
                    gamma=hyperparams['gamma'],
                    degree=hyperparams['degree'],
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
            
            if task_type == "Classification":
                metrics_dict = calculate_classification_metrics(y_test, y_pred)
                metrics_dict["CV"] = cv_results
                result_text = format_results_markdown(metrics_dict, task_type="classification")
            else:
                metrics_dict = calculate_regression_metrics(y_test, y_pred)
                metrics_dict["CV"] = cv_results
                result_text = format_results_markdown(metrics_dict, task_type="regression")
            
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                f"SVM {task_type} Results",
                result_text,
                "SVM"
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
        """Build Flet UI card for SVM hyperparameter configuration."""
        
        self.C_field = ft.TextField(
            label="C (Regularization)",
            value="1.0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^\d+(\.\d*)?|\.\d+$'),
            tooltip="Regularization parameter. Higher C → less regularization. Range: 0.0001-1000",
        )
        
        self.kernel_dropdown = ft.Dropdown(
            label="Kernel Type",
            value="rbf",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("linear", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("rbf", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("poly", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("sigmoid", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="Kernel type. linear=fast/interpretable, rbf=flexible, poly=custom power, sigmoid=neural-like",
        )
        
        self.gamma_field = ft.TextField(
            label="Gamma",
            value="scale",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Kernel coefficient. 'scale'=1/(n_features*X.var()), 'auto'=1/n_features, or numeric value (0.0001-1.0)",
        )
        
        self.degree_field = ft.TextField(
            label="Degree (poly kernel)",
            value="3",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Degree of the polynomial kernel function ('poly'). Must be non-negative. Ignored by all other kernels.",
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
                                ft.Text("Support Vector Machine",
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
                        ft.Row([self.C_field, self.kernel_dropdown]),
                        self.gamma_field,
                        self.degree_field,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
