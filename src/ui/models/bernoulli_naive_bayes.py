"""
Naive Bayes Classification Model

Probabilistic classifier based on Bayes' theorem with strong
independence assumptions between features.

Supports:
- Bernoulli: Like MultinomialNB, this classifier is suitable for discrete data. The difference is that while MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolean features.

Configurable hyperparameters:
- alpha: Additive (Laplace/Lidstone) smoothing parameter (set alpha=0 and force_alpha=True, for no smoothing).
- force_alpha: If False and alpha is less than 1e-10, it will set alpha to 1e-10. If True, alpha will remain unchanged. This may cause numerical errors if alpha is too close to 0
- binarize : Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors
- fit_prior: Whether to learn class prior probabilities or not. If false, a uniform prior will be used
"""

from __future__ import annotations
from typing import Tuple
import flet as ft
from functools import partial
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import BernoulliNB

from utils.model_utils import (
    calculate_classification_metrics,
    format_results_markdown,
    create_results_dialog,
)
from .base_model import BaseModel, CLASSIFICATION_THRESHOLD


@dataclass
class BernoulliNBModel(BaseModel):
    """Multinomial Naive Bayes classification model."""

    def _prepare_data(self):
        """Prepare data for training."""
        return self._prepare_data_supervised()

    def _validate_hyperparameters(self) -> Tuple[dict, bool]:
        is_valid = True
        params = {}

        # Validate alpha
        try:
            alpha = float(self.alpha.value)
            if alpha < 0:
                alpha = 1
                is_valid = False
            params['alpha'] = alpha
        except (ValueError, TypeError):
            params['alpha'] = 1
            is_valid = False

        # validate binarize
        try:
            binarize = self.binarize.value.strip()
            if binarize == "None":
                params['binarize'] = None
            else:
                binarize = float(self.binarize.value.strip())
                if binarize < 0:
                    params['binarize'] = 0
                    is_valid = False
                else:
                    params['binarize'] = binarize
        except (ValueError, TypeError):
                params['binarize'] = 0
                is_valid = False
        
        return params, is_valid

    def _create_model(self) -> BernoulliNB:
        hyperparams, params_valid = self._validate_hyperparameters()
        if not params_valid:
            self._show_snackbar("Invalid hyperparameters. Using default values.", bgcolor=ft.Colors.AMBER_ACCENT_200)
        model = BernoulliNB(
            alpha=hyperparams['alpha'],
            force_alpha=self.force_alpha.value,
            fit_prior=self.fit_prior.value,
            binarize=hyperparams['binarize'],
        )
        return model
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent | None = None, force: bool = False) -> None:
        """Train Naive Bayes model and display evaluation results."""
        try:
            self._disable_training_controls()

            data = self._prepare_data()
            if data is None:
                return
            
            ratio = self._target_to_total_rows_ratio()
            if ratio > CLASSIFICATION_THRESHOLD and self._get_task_type() == "Classification" and not force:
                self._show_wrong_task_type_dialog(ratio, 'Classification', partial(self._train_and_evaluate_model, e=e, force=True))
                return
            
            X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols) = data

            model = self._create_model()

            model.fit(X_train, y_train)
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
                "Bernoulli Naive Bayes Classification Results",
                result_text,
                "Naive Bayes"
            )
            self.parent.page.open(evaluation_dialog)

        except Exception as e:
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)

        finally:
            self._enable_training_controls()

    def _set_binarize_None(self, e: ft.ControlEvent) -> None:
        self.binarize.value = "None"
        self.parent.page.update()
    
    def _binarize_on_click(self, e: ft.ControlEvent) -> None:
        value = e.control.value.strip()
        if value == "None":
            self.binarize.value = ""
            self.parent.page.update()
        
    def _binarize_on_blur(self, e: ft.ControlEvent) -> None:
        value = e.control.value.strip()
        if value == "":
            self.binarize.value = "0"
            self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for Naive Bayes hyperparameter configuration."""

        self.alpha = ft.TextField(
            label="Alpha",
            value="1",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            tooltip="Additive (Laplace/Lidstone) smoothing parameter (set alpha=0 and force_alpha=True, for no smoothing).",
        )
        
        self.binarize = ft.TextField(
            label="Binarize",
            value="0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            suffix_icon=ft.IconButton(ft.Icons.RESTART_ALT, tooltip="Set 'None'", on_click=self._set_binarize_None),
            on_click=self._binarize_on_click,
            on_blur=self._binarize_on_blur,
            tooltip="Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors"
        )
        
        self.force_alpha = ft.Switch(
            label="Force alpha",
            value=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="If False and alpha is less than 1e-10, it will set alpha to 1e-10. If True, alpha will remain unchanged. This may cause numerical errors if alpha is too close to 0.",
        )
        
        self.fit_prior = ft.Switch(
            label="Fit prior",
            value=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            label_position=ft.LabelPosition.LEFT,
            tooltip="Only used in edge case with a single class in the training set.",
        )
        
        self._build_train_button()
        self._build_predict_new_data_button()
        
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
                                ft.Text("Naive Bayes (Bernoulli)",
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
                        ft.Row([self.alpha, self.binarize]),
                        ft.Row([self.force_alpha, self.fit_prior], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        ft.Row([self.train_btn, self.test_data_btn])
                    ]
                )
            )
        )
