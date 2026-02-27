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
from typing import Optional, Tuple, TYPE_CHECKING, Literal
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.model_utils import (
    disable_navigation_bar,
    enable_navigation_bar,
)
from core.data_preparation import prepare_data_for_training, prepare_data_for_training_no_split

if TYPE_CHECKING:
    from ..model_factory import ModelFactory


@dataclass
class BaseModel(ABC):
    """Abstract base class for all ML models.

    Attributes:
        parent: Reference to the ModelFactory UI parent.
        df: The dataset to train on.
    """

    parent: ModelFactory
    df: DataFrame

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
                cardinality_warnings,
            ) = prepare_data_for_training(
                self.df.copy(),
                target_col=target_name,
                test_size=test_size,
                random_state=42,
                raise_on_unseen=True,
            )

            self._store_encoding_metadata(
                categorical_cols, numeric_cols, encoders, cardinality_warnings
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
                cardinality_warnings,
            ) = prepare_data_for_training_no_split(
                self.df.copy(),
                target_col=None,
                raise_on_unseen=True,
            )

            self._store_encoding_metadata(
                categorical_cols, numeric_cols, encoders, cardinality_warnings
            )
            self._warn_cardinality(cardinality_warnings)

            # Apply scaling based on user selection
            if self.parent.scaler_dropdown.value == "standard_scaler":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_encoded)
            elif self.parent.scaler_dropdown.value == "minmax_scaler":
                scaler = MinMaxScaler(feature_range=(0, 1))
                X_scaled = scaler.fit_transform(X_encoded)
            else:
                X_scaled = X_encoded.values

            return X_scaled, X_encoded.columns.tolist()

        except Exception as e:
            self._show_snackbar(f"Data preparation error: {str(e)}")
            return None

    def _store_encoding_metadata(
        self, categorical_cols, numeric_cols, encoders, cardinality_warnings
    ) -> None:
        """Store encoding metadata from data preparation for later use."""
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.encoders = encoders
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

    # ── UI Helpers ─────────────────────────────────────────────

    def _show_snackbar(self, message: str, bgcolor: str | None = None) -> None:
        """Display a snackbar notification to the user."""
        snackbar = ft.SnackBar(
            ft.Text(message, font_family="SF regular"), action="Ok"
        )
        if bgcolor:
            snackbar.bgcolor = bgcolor
        self.parent.page.open(snackbar)

    def _build_train_button(self, on_click=None) -> ft.FilledButton:
        """Create the standard 'Train and evaluate model' button.

        Args:
            on_click: Click handler. Defaults to self._train_and_evaluate_model.

        Returns:
            The created FilledButton (also stored as self.train_btn).
        """
        if on_click is None:
            on_click = self._train_and_evaluate_model

        self.train_btn = ft.FilledButton(
            text="Train and evaluate model",
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

    # ── Training Lifecycle ─────────────────────────────────────

    def _disable_training_controls(self, e: ft.ControlEvent) -> None:
        """Disable UI controls at the start of training."""
        e.control.disabled = True
        self.parent.disable_model_selection()
        disable_navigation_bar(self.parent.page)
        self.parent.page.update()

    def _enable_training_controls(self) -> None:
        """Re-enable UI controls after training completes (use in finally block)."""
        enable_navigation_bar(self.parent.page)
        self.parent.enable_model_selection()
        self.train_btn.disabled = False
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
