"""
Base Model

Abstract base class providing shared functionality for all ML model implementations.

Common utilities:
- Data preparation (supervised and clustering variants)
- Task type detection (classification vs regression)
- Field event handlers (on_click, on_blur, reset to None)
- UI helpers (snackbar notifications, train button builder)
- Training lifecycle (disable/enable controls)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, TYPE_CHECKING, Literal, Callable, List, Dict
import flet as ft
from dataclasses import dataclass, field
from pandas import DataFrame, to_numeric
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.base import BaseEstimator

from utils.model_utils import (
    disable_navigation_bar,
    enable_navigation_bar,
    CardinalityWarning
)
from core.data_preparation import prepare_data_for_training, prepare_data_for_training_no_split

if TYPE_CHECKING:
    from ..model_factory import ModelFactory


CLASSIFICATION_THRESHOLD = 50


@dataclass
class BaseModel(ABC):
    """Abstract base class for all ML models.

    Attributes:
        parent: Reference to the ModelFactory UI parent.
        df: The dataset to train on.
    """

    parent: ModelFactory
    df: DataFrame
    model: BaseEstimator | None = field(default=None, init=False) 

    def __post_init__(self):
        """Ensure dataset is copied to avoid mutations."""
        self.df = self.df.copy()

    # ── Task Type ──────────────────────────────────────────────

    def _get_task_type(self) -> Literal["Classification", "Regression"]:
        """Determine if this is a classification or regression task."""
        task_type = self.parent.task_type_dropdown.value
        return task_type if task_type in ["Classification", "Regression"] else "Classification"

    # ── Field Event Handlers ───────────────────────────────────

    def _field_on_click(self, e: ft.ControlEvent) -> None:
        """Clear 'None' placeholder when field is clicked."""
        if e.control.value.strip() == "None":
            e.control.value = ""
            self.parent.page.update()

    def _field_on_blur(self, e: ft.ControlEvent) -> None:
        """Restore 'None' placeholder when field loses focus and is empty."""
        if e.control.value.strip() == "":
            e.control.value = "None"
            self.parent.page.update()

    def _reset_field_to_none(self, control: ft.TextField) -> None:
        """Reset a text field value back to 'None'."""
        control.value = "None"
        self.parent.page.update()

    # ── Data Preparation ───────────────────────────────────────

    def _prepare_data_supervised(self) -> Optional[Tuple]:
        """
        Prepare data for supervised learning (classification/regression).

        Uses prepare_data_for_training() which:
        - Performs train-test split BEFORE encoding (prevents data leakage)
        - Fits encoders ONLY on training data
        - Applies encoders to test data
        - Returns encoding metadata and cardinality warnings

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols))
            or None if error
        """
        try:
            target_name = self.parent.target_column_dropdown.value
            test_size = self.parent.test_size_field.value / 100

            (
                X_train,
                X_test,
                y_train,
                y_test,
                categorical_cols,
                numeric_cols,
                encoders,
                scaler,
                cardinality_warnings,
            ) = prepare_data_for_training(
                self.df.copy(),
                target_col=target_name,
                test_size=test_size,
                random_state=42,
                raise_on_unseen=True,
                scaler_mode=self.parent.scaler_dropdown.value
            )

            self._store_encoding_metadata(
                categorical_cols, numeric_cols, encoders, scaler, cardinality_warnings
            )
            self._warn_cardinality(cardinality_warnings)
            
            return X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols)

        except Exception as e:
            self._show_snackbar(f"Data preparation error: {str(e)}")
            return None

    def _prepare_data_clustering(self) -> Optional[Tuple]:
        """
        Prepare data for clustering (unsupervised, no train-test split).

        Uses prepare_data_for_training_no_split() which:
        - Detects and encodes categorical columns
        - Validates cardinality
        - Fits encoders on full dataset
        - Returns encoded features and metadata

        Applies feature scaling based on parent scaler_dropdown selection.

        Returns:
            Tuple of (X_scaled, feature_columns) or None if error
        """
        try:
            (
                X_encoded,
                _,  # y is None for clustering
                categorical_cols,
                numeric_cols,
                encoders,
                scaler,
                cardinality_warnings,
            ) = prepare_data_for_training_no_split(
                self.df.copy(),
                target_col=None,
                raise_on_unseen=True,
                scaler_mode=self.parent.scaler_dropdown.value
            )

            self._store_encoding_metadata(
                categorical_cols, numeric_cols, encoders, scaler, cardinality_warnings
            )
            self._warn_cardinality(cardinality_warnings)

            return X_encoded, self.df.columns.tolist()

        except Exception as e:
            self._show_snackbar(f"Data preparation error: {str(e)}")
            return None

    def _store_encoding_metadata(
        self,
        categorical_cols: List[str], numeric_cols: List[str],
        encoders: Dict[str, LabelEncoder], scaler: Optional[StandardScaler | MinMaxScaler],
        cardinality_warnings: Dict[str, CardinalityWarning]
    ) -> None:
        """Store encoding metadata from data preparation for later use."""
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.encoders = encoders
        self.scaler = scaler
        self.cardinality_warnings = cardinality_warnings

    def _warn_cardinality(self, cardinality_warnings) -> None:
        """Show snackbar warning if high-cardinality columns were detected."""
        if cardinality_warnings:
            warning_msgs = [
                f"{col}: {w.message}"
                for col, w in cardinality_warnings.items()
            ]
            self._show_snackbar(
                "Cardinality warnings: " + "; ".join(warning_msgs),
                bgcolor=ft.Colors.AMBER_ACCENT_200,
            )
            
    def _target_to_total_rows_ratio(self) -> float | int:
        """Returns percentage of unique values of target column to samples"""
        return round((self.df[self.parent.target_column_dropdown.value].nunique() / self.df.shape[0]) * 100, 2)

    # ── UI Helpers ─────────────────────────────────────────────

    def _show_snackbar(self, message: str, bgcolor: str | None = None) -> None:
        """Display a snackbar notification to the user."""
        snackbar = ft.SnackBar(
            ft.Text(message, font_family="SF regular"), action="Ok"
        )
        if bgcolor:
            snackbar.bgcolor = bgcolor
        self.parent.page.open(snackbar)
    
    def _show_wrong_task_type_dialog(
        self, ratio: int | float,
        task_type: Literal["Classification", "Regression"] = "Classification",
        call_back_func: Callable | None = None,
    ):
        def _continue(callback: Callable):
            nonlocal dialog
            self.parent.page.close(dialog)
            callback()
        
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Column(
                controls=[
                    ft.Row([ft.Text(f'Wrong task type', font_family="SF thin", expand=True, text_align="center")]),
                    ft.Divider()
                ]
            ),
            content=ft.Container(
                width=500,
                content=ft.Text(
                    f"This could be a '{task_type}' problem. Ratio percentage of '{self.parent.target_column_dropdown.value}' unique count to samples is %{ratio}. Would you like to continue?",
                    font_family="SF regular",
                    text_align=ft.TextAlign.CENTER
                ),
            ),
            actions=[ft.Row(
                controls=[
                    ft.OutlinedButton(
                        text="Cancel",
                        expand=1,
                        style=ft.ButtonStyle(text_style=ft.TextStyle(font_family="SF regular")),
                        on_click=lambda _: self.parent.page.close(dialog)
                    ),
                    ft.FilledButton(
                        text="Continue",
                        expand=1,
                        style=ft.ButtonStyle(text_style=ft.TextStyle(font_family="SF regular")),
                        on_click=lambda _: _continue(call_back_func)
                    ),
                ]
            )]
        )
        self.parent.page.open(dialog)

    def _build_train_button(self, on_click: Callable | None = None) -> ft.FilledButton:
        """Create the standard 'Train and evaluate model' button.

        Args:
            on_click: Click handler. Defaults to self._train_and_evaluate_model.

        Returns:
            The created FilledButton (also stored as self.train_btn).
        """
        if on_click is None:
            on_click = self._train_and_evaluate_model

        self.train_btn = ft.FilledButton(
            text="Train & evaluate",
            icon=ft.Icons.PSYCHOLOGY,
            on_click=on_click,
            expand=1,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            ),
        )
        return self.train_btn
    
    def _build_predict_new_data_button(self, on_click: Callable | None = None):
        if on_click is None:
            on_click = self._open_test_data_dialog
        
        self.test_data_btn = ft.OutlinedButton(
            text="Predict new data",
            icon=ft.Icons.ONLINE_PREDICTION,
            disabled=False,
            expand=1,
            on_click=on_click,
            tooltip="Train model on all sample then predict entered data on the model",
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular")
            ),
        )
        return self.test_data_btn
    
    def _open_test_data_dialog(self, e: ft.ControlEvent) -> None:
        def field_changed(e: ft.ControlEvent) -> None:
            if e.control.value is None or e.control.value.strip() == "":
                e.control.error_text = "This field can not be empty"
            else:
                e.control.error_text = ""
            self.parent.page.update()
            
        def predict(e: ft.ControlEvent):
            try:
                nonlocal model, scaler, encoders, categorical_cols
                if (
                    any(control.error_text not in (None, "") for control in controls.values()) or
                    any(control.value.strip() == "" for control in controls.values() if isinstance(control, ft.TextField))
                ):
                    result_message.value = "Fill all empty fields"
                    result_message.color = "red"
                    return
                predict_btn.disabled = True
                cancel_btn.disabled = True
                fill_data_btn.disabled = True
                self.parent.page.update()
                
                is_supervised = True if self.parent.learning_type_dropdown.value == "Supervised" else False
                if model is None:
                    (
                        X_encoded,
                        y,
                        categorical_cols,
                        numeric_cols,
                        encoders,
                        scaler,
                        cardinality_warnings,
                    ) = prepare_data_for_training_no_split( # This is not model test anymore so we use all the data for training
                        self.df.copy(),
                        target_col=self.parent.target_column_dropdown.value if is_supervised else None,
                        raise_on_unseen=True,
                        scaler_mode=self.parent.scaler_dropdown.value
                    )
                
                # Create a new dataframe for inputed data
                array = {}
                for col, control in controls.items():
                    if col in categorical_cols:
                        encoder = encoders.get(col)
                        encoded_data = encoder.transform(np.array([control.value]))[0]
                    else:
                        encoded_data = to_numeric(control.value)
                    array[col] = [encoded_data]
                X_new = DataFrame(array)
                
                # Scale new data if scaler was chosen
                if scaler:
                    X_new = scaler.transform(X_new)
                    # X_new = scaler.transform(X_new)[0]
                    # X_new = DataFrame(dict(zip(list(controls.keys()), map(lambda val: [val], X_new))))
                else:
                    X_new = X_new.values
                
                # Train model based on learning type and predict new data
                if model is None:
                    model = self._create_model()
                    if is_supervised:
                        model.fit(X_encoded, y)
                    else:
                        model.fit(X_encoded)
                predicted_data = model.predict(X_new)[0]
            except Exception as e:
                result_message.value = str(e)
                result_message.color = "red"
            else:
                if is_supervised:
                    result_message.value = f"'{self.parent.target_column_dropdown.value}': {predicted_data}"
                else:
                    result_message.value = f"Cluster: {predicted_data}"
                result_message.color = "green"
            finally:
                predict_btn.disabled = False
                cancel_btn.disabled = False
                fill_data_btn.disabled = False
                self.parent.page.update()
                
        def input_data_from_random_sample(e: ft.ControlEvent):
            sample = next(df.sample().iterrows())[1]
            for col, control in controls.items():
                if isinstance(control, ft.TextField):
                    control.value = str(sample[col])
                    control.error_text = ""
                else: # ft.Dropdown
                    available_options = [option.key for option in control.options]
                    if str(sample[col]) in available_options:
                        control.value = str(sample[col])
                    else:
                        if self.parent.parent.dataset.is_numeric(sample[col]):
                            if str(int(sample[col])) in available_options:
                                control.value = str(int(sample[col]))
            self.parent.page.update()
        
        try:
            # save required attributes on first train to avoid training each test 
            model = None
            scaler = None
            encoders = None
            categorical_cols = None
            
            df = self.df.copy()
            if self.parent.learning_type_dropdown.value == "Supervised":
                df = df.drop(columns=[self.parent.target_column_dropdown.value])
            controls = {}
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for column in df.columns.to_list():
                unique_items_count = df[column].nunique()
                if column in cat_cols or column in bool_cols:
                    control = ft.Dropdown(
                        label=column,
                        enable_search=True,
                        expand=1,
                        col={"xs": 6, "sm": 6, "md": 6, "lg": 6, "xl": 6},
                        label_style=ft.TextStyle(font_family="SF regular"),
                        text_style=ft.TextStyle(font_family="SF regular"),
                        options=[ft.DropdownOption(str(item)) for item in sorted(df[column].unique().tolist())]
                    )
                    control.value = control.options[0].key
                elif column in numeric_cols:
                    if unique_items_count <= 20:
                        control = ft.Dropdown(
                            label=column,
                            enable_search=True,
                            expand=1,
                            col={"xs": 6, "sm": 6, "md": 6, "lg": 6, "xl": 6},
                            label_style=ft.TextStyle(font_family="SF regular"),
                            text_style=ft.TextStyle(font_family="SF regular"),
                            options=[ft.DropdownOption(str(item)) for item in sorted(df[column].unique().tolist())]
                        )
                        control.value = control.options[0].key
                    else:
                        control = ft.TextField(
                            label=column,
                            expand=1,
                            col={"xs": 6, "sm": 6, "md": 6, "lg": 6, "xl": 6},
                            label_style=ft.TextStyle(font_family="SF regular"),
                            text_style=ft.TextStyle(font_family="SF regular"),
                            error_style=ft.TextStyle(font_family="SF regular", color="red"),
                            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
                            on_change=field_changed
                        )
                else:
                    control = ft.TextField(
                        label=column,
                        col={"xs": 6, "sm": 6, "md": 6, "lg": 6, "xl": 6},
                        expand=1,
                        label_style=ft.TextStyle(font_family="SF regular"),
                        text_style=ft.TextStyle(font_family="SF regular"),
                        error_style=ft.TextStyle(font_family="SF regular", color="red"),
                        on_change=field_changed
                    )
                controls[column] = control
            
            dialog = ft.AlertDialog(
                modal=True,
                title=ft.Column(
                    controls=[
                        ft.Row([ft.Text(f'Predict new data', font_family="SF thin", expand=True, text_align="center")]),
                        ft.Divider()
                    ]
                ),
                content=ft.Container(
                    width=500,
                    content=ft.Column(
                        scroll=ft.ScrollMode.AUTO,
                        controls=[
                            ft.Row(
                                controls=[
                                    ft.Text("Result will be appear here:", font_family="SF regular"),
                                    result_message := ft.Text("", font_family="SF regular")
                                ]
                            ),
                            ft.Row(
                                controls=[
                                    fill_data_btn := ft.TextButton(
                                        "Fill data from a sample",
                                        icon=ft.Icons.DATA_OBJECT,
                                        expand=1,
                                        style=ft.ButtonStyle(text_style=ft.TextStyle(font_family="SF regular")),
                                        on_click=input_data_from_random_sample
                                    )
                                ]
                            ),
                            ft.ResponsiveRow([control for control in controls.values()])
                        ]
                    )
                ),
                actions=[ft.Row(
                    controls=[
                        cancel_btn := ft.OutlinedButton(
                            text="Cancel",
                            expand=1,
                            style=ft.ButtonStyle(text_style=ft.TextStyle(font_family="SF regular")),
                            on_click=lambda _: self.parent.page.close(dialog)
                        ),
                        predict_btn := ft.FilledButton(
                            text="Predict",
                            expand=1,
                            style=ft.ButtonStyle(text_style=ft.TextStyle(font_family="SF regular")),
                            on_click=predict
                        ),
                    ]
                )]
            )
            self.parent.page.open(dialog)
                    
        except Exception as e:
            self._show_snackbar(f"Unable to open test data menu: {e}", ft.Colors.RED_500)

    # ── Training Lifecycle ─────────────────────────────────────

    def _disable_training_controls(self) -> None:
        """Disable UI controls at the start of training."""
        if hasattr(self, 'test_data_btn'):
            self.test_data_btn.disabled = True
        self.train_btn.disabled = True
        self.parent.disable_model_selection()
        disable_navigation_bar(self.parent.page)
        self.parent.page.update()

    def _enable_training_controls(self) -> None:
        """Re-enable UI controls after training completes (use in finally block)."""
        if hasattr(self, 'test_data_btn'):
            self.test_data_btn.disabled = False
        self.train_btn.disabled = False
        self.parent.enable_model_selection()
        enable_navigation_bar(self.parent.page)
        self.parent.page.update()

    # ── Abstract Methods ───────────────────────────────────────

    @abstractmethod
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for model hyperparameter configuration."""
        ...

    @abstractmethod
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train model and display evaluation results."""
        ...
    
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create model"""
        ...
        # raise NotImplementedError(f"'{self.__class__.__name__}' does not have '_create_model' method.")
