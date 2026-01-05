"""
Decision Tree Classification Model

Tree-based classifier that recursively partitions data based on feature values
to minimize impurity at each node.

Configurable hyperparameters:
- max_depth: Maximum depth of tree (controls complexity)
- min_samples_split: Minimum samples required to split internal node
- min_samples_leaf: Minimum samples required at leaf node
- criterion: Function to measure split quality ('gini', 'entropy', 'log_loss')
- splitter: Strategy for selecting splits ('best' or 'random')
- max_features: Number of features to consider when looking for best split
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from utils.model_utils import (
    check_data_quality,
    calculate_classification_metrics,
    format_results_markdown,
    create_results_dialog,
    get_feature_importance,
    disable_navigation_bar,
    enable_navigation_bar,
)
from core.data_preparation import prepare_data_for_training

if TYPE_CHECKING:
    from ..model_factory import ModelFactory


@dataclass
class DecisionTreeModel:
    """Decision Tree classifier with configurable hyperparameters."""
    
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
        
        # Validate max_depth
        try:
            max_depth_value = int(self.max_depth_field.value)
            if max_depth_value < 1 or max_depth_value > 50:
                max_depth_value = 10
                is_valid = False
            params['max_depth'] = max_depth_value
        except (ValueError, TypeError):
            params['max_depth'] = 10
            is_valid = False
        
        # Validate min_samples_split
        try:
            min_split_value = int(self.min_samples_split_field.value)
            if min_split_value < 2 or min_split_value > 20:
                min_split_value = 2
                is_valid = False
            params['min_samples_split'] = min_split_value
        except (ValueError, TypeError):
            params['min_samples_split'] = 2
            is_valid = False
        
        # Validate min_samples_leaf
        try:
            min_leaf_value = int(self.min_samples_leaf_field.value)
            if min_leaf_value < 1 or min_leaf_value > 20:
                min_leaf_value = 1
                is_valid = False
            params['min_samples_leaf'] = min_leaf_value
        except (ValueError, TypeError):
            params['min_samples_leaf'] = 1
            is_valid = False
        
        # Validate criterion
        valid_criteria = ['gini', 'entropy', 'log_loss']
        criterion_value = self.criterion_dropdown.value
        if criterion_value not in valid_criteria:
            criterion_value = 'gini'
            is_valid = False
        params['criterion'] = criterion_value
        
        # Validate splitter
        valid_splitters = ['best', 'random']
        splitter_value = self.splitter_dropdown.value
        if splitter_value not in valid_splitters:
            splitter_value = 'best'
            is_valid = False
        params['splitter'] = splitter_value
        
        # Validate max_features
        max_features_value = self.max_features_dropdown.value
        if max_features_value not in ['sqrt', 'log2', 'None']:
            max_features_value = 'None'
            is_valid = False
        params['max_features'] = None if max_features_value == 'None' else max_features_value
        
        return params, is_valid
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train decision tree model and display evaluation results."""
        try:
            e.control.disabled = True
            self.parent.disable_model_selection()
            self.parent.page.update()
            
            # Check data quality first
            is_valid, error_msg = check_data_quality(self.df, self.parent.target_column_dropdown.value)
            if not is_valid:
                self.parent.page.open(ft.SnackBar(
                    ft.Text(f"Data error: {error_msg}", font_family="SF regular")
                ))
                return
            
            # Prepare data
            data = self._prepare_data()
            if data is None:
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
                    ),
                    bgcolor="#FF9800"
                ))
            
            # Create and train model with validated parameters
            model = DecisionTreeClassifier(
                max_depth=hyperparams['max_depth'],
                min_samples_split=hyperparams['min_samples_split'],
                min_samples_leaf=hyperparams['min_samples_leaf'],
                criterion=hyperparams['criterion'],
                splitter=hyperparams['splitter'],
                max_features=hyperparams['max_features'],
                random_state=42,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics using centralized utility
            metrics_dict = calculate_classification_metrics(y_test, y_pred)
            
            # Get feature importance for decision tree
            feature_importance = get_feature_importance(model, self.df.columns.tolist())
            if feature_importance:
                metrics_dict['feature_importance'] = feature_importance
            
            result_text = format_results_markdown(metrics_dict, task_type="classification")
            
            # Display results dialog with copy button
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Decision Tree Classification Results",
                result_text,
                "Decision Tree"
            )
            self.parent.page.open(evaluation_dialog)
        
        except Exception as e:
            self.parent.page.open(ft.SnackBar(
                ft.Text(f"Training failed: {str(e)}", font_family="SF regular")
            ))
        
        finally:
            self.parent.enable_model_selection()
            self.train_btn.disabled = False
            self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for decision tree hyperparameter configuration."""
        
        # Create hyperparameter controls
        self.max_depth_field = ft.TextField(
            label="Max Depth",
            value="10",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Maximum depth of tree. Range: 1 to 50. Lower values prevent overfitting",
        )
        
        self.min_samples_split_field = ft.TextField(
            label="Min Samples Split",
            value="2",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Minimum samples required to split node. Range: 2 to 20. Higher values reduce tree complexity",
        )
        
        self.min_samples_leaf_field = ft.TextField(
            label="Min Samples Leaf",
            value="1",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Minimum samples at leaf node. Range: 1 to 20. Higher values create smoother decision boundaries",
        )
        
        self.criterion_dropdown = ft.Dropdown(
            label="Criterion",
            value="gini",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("gini", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("entropy", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("log_loss", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="Function to measure split quality. gini=default, entropy=information gain based",
        )
        
        self.splitter_dropdown = ft.Dropdown(
            label="Splitter",
            value="best",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("best", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("random", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="'best' searches all splits, 'random' uses random thresholds. best=slower but more accurate",
        )
        
        self.max_features_dropdown = ft.Dropdown(
            label="Max Features",
            value="None",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("sqrt", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("log2", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="Number of features to consider for splits. sqrt/log2 reduce overfitting on high-dimensional data",
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
                                ft.Text("Decision Tree Classifier",
                                    font_family="SF thin",
                                    size=24,
                                    text_align="center",
                                    expand=True
                                )
                            ]
                        ),
                        ft.Divider(),
                        ft.Text("Hyperparameters",
                               font_family="SF regular",
                               weight="bold",
                               size=14),
                        ft.Row([self.max_depth_field, self.min_samples_split_field]),
                        ft.Row([self.min_samples_leaf_field, self.criterion_dropdown]),
                        ft.Row([self.splitter_dropdown, self.max_features_dropdown]),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
