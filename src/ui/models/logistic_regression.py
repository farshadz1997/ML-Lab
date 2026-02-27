"""
Logistic Regression Classification Model

Multinomial/binary logistic regression classifier with configurable hyperparameters:
- C: Inverse of regularization strength
- max_iter: Maximum iterations for solver
- solver: Optimization algorithm
- class_weight: Strategy for handling class imbalance
"""

from __future__ import annotations
from typing import Optional, Tuple
import flet as ft
from dataclasses import dataclass
from numpy import inf as infinite
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

from utils.model_utils import (
    check_data_quality,
    calculate_classification_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel


@dataclass
class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier with configurable hyperparameters."""
    
    def _prepare_data(self):
        """Prepare data for training."""
        return self._prepare_data_supervised()
    
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
        
        # Validate C
        try:
            if self.C_field.value.strip() == "Infinite":
                params['C'] = infinite
            else:
                c_value = float(self.C_field.value)
                if c_value <= 0 or c_value > 1000:
                    c_value = 1.0
                    is_valid = False
                params['C'] = c_value
        except (ValueError, TypeError):
            params['C'] = 1.0
            is_valid = False
        
        # Validate max_iter
        try:
            max_iter_value = int(self.max_iter_field.value)
            if max_iter_value < 1:
                max_iter_value = 100
                is_valid = False
            params['max_iter'] = max_iter_value
        except (ValueError, TypeError):
            params['max_iter'] = 100
            is_valid = False
        
        # Validate solver
        valid_solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
        solver_value = self.solver_dropdown.value
        if solver_value not in valid_solvers:
            solver_value = 'lbfgs'
            is_valid = False
        params['solver'] = solver_value

        # Validate penalty
        penalty = self.penalty_dropdown.value
        if penalty == "None":
            penalty = None
        # Enforce compatible penalty-solvers
        if solver_value in ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'] and penalty not in [None, 'l2']:
            penalty = 'l2'
            is_valid = False
        params['penalty'] = penalty

        try:
            if self.l1_ratio_field.value.strip() == "None":
                params['l1_ratio'] = None
            else:
                l1_ratio = float(self.l1_ratio_field.value)
                if not 0 <= l1_ratio <= 1:
                    params['l1_ratio'] = 0
                    is_valid = False
                else:
                    if solver_value in ('lbfgs', 'newton-cg', 'newton-cholesky', 'sag'):
                        params['l1_ratio'] = 0
                    elif solver_value == 'liblinear': # 0 or 1 only
                        if l1_ratio not in (0, 1):
                            params['l1_ratio'] = 0
                            is_valid = False
                        else:
                            params['l1_ratio'] = l1_ratio
                    else:  # saga supports all penalties
                        params['l1_ratio'] = l1_ratio
        except (ValueError, TypeError):
            params['l1_ratio'] = None
            is_valid = False
        
        # Validate class_weight
        class_weight_value = self.class_weight_dropdown.value
        if class_weight_value == 'None':
            class_weight_value = None
        elif class_weight_value not in ['balanced', None]:
            class_weight_value = None
            is_valid = False
        params['class_weight'] = class_weight_value

        # Validate fit_intercept and intercept_scaling for liblinear
        params['fit_intercept'] = self.fit_intercept_switch.value
        try:
            intercept_scaling_value = float(self.intercept_scaling_field.value)
            if intercept_scaling_value <= 0:
                intercept_scaling_value = 1.0
                is_valid = False
            params['intercept_scaling'] = intercept_scaling_value
        except (ValueError, TypeError):
            params['intercept_scaling'] = 1.0
            is_valid = False
        
        return params, is_valid
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train logistic regression model and display evaluation results."""
        try:
            e.control.disabled = True
            self.parent.disable_model_selection()
            disable_navigation_bar(self.parent.page)
            self.parent.page.update()
            
            # Check data quality first
            is_valid, error_msg = check_data_quality(self.df, self.parent.target_column_dropdown.value)
            if not is_valid:
                self._show_snackbar(f"Data error: {error_msg}", bgcolor=ft.Colors.RED_500)
                enable_navigation_bar(self.parent.page)
                return
            
            # Prepare data
            data = self._prepare_data()
            if data is None:
                enable_navigation_bar(self.parent.page)
                return
            
            X_train, X_test, y_train, y_test, (categorical_cols, numeric_cols) = data
            
            # Validate and get hyperparameters with defaults for invalid inputs
            hyperparams, params_valid = self._validate_hyperparameters()
            
            # If invalid params were detected, inform user
            if not params_valid:
                self._show_snackbar("Some hyperparameters were invalid. Using defaults.", bgcolor=ft.Colors.AMBER_ACCENT_200)
            
            # Create and train model with validated parameters
            model = LogisticRegression(
                penalty=hyperparams['penalty'],
                C=hyperparams['C'],
                fit_intercept=hyperparams['fit_intercept'],
                intercept_scaling=hyperparams['intercept_scaling'],
                max_iter=hyperparams['max_iter'],
                solver=hyperparams['solver'],
                class_weight=hyperparams['class_weight'],
                random_state=42,
                l1_ratio=hyperparams['l1_ratio']
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
            metrics_dict = calculate_classification_metrics(y_test, y_pred)
            metrics_dict["CV"] = cv_results
            result_text = format_results_markdown(metrics_dict, task_type="classification")
            
            # Display results dialog with copy button
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Logistic Regression Classification Results",
                result_text,
                "Logistic Regression"
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
    
    def _solver_on_change(self, e: ft.ControlEvent) -> None:
        """Enable/disable penalty options based on selected solver."""
        solver = self.solver_dropdown.value

        if solver == 'liblinear':
            self.fit_intercept_switch.visible = True
            self.intercept_scaling_field.visible = True
        else:
            self.fit_intercept_switch.visible = False
            self.intercept_scaling_field.visible = False
            self.fit_intercept_switch.value = True  # liblinear requires fit_intercept=True
            self.intercept_scaling_field.disabled = False
            self.intercept_scaling_field.value = "1.0"  # Default value

        if solver in ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag']:
            # Only 'l2' or None allowed
            self.penalty_dropdown.options = [
                ft.DropdownOption("l2"),
                ft.DropdownOption("None"),
            ]
            if self.penalty_dropdown.value not in ["l2", "None"]:
                self.penalty_dropdown.value = "l2"
                self._show_snackbar(f"'{solver}' solver only supports 'l2' or None penalty. Defaulting to 'l2'.", bgcolor=ft.Colors.AMBER_ACCENT_200)
        elif solver == 'liblinear':
            # 'l1' and 'l2' allowed, but not elasticnet
            self.penalty_dropdown.options = [
                ft.DropdownOption("l1"),
                ft.DropdownOption("l2"),
                ft.DropdownOption("None"),
            ]
            if self.penalty_dropdown.value == "elasticnet":
                self.penalty_dropdown.value = "l2"
                self._show_snackbar("'liblinear' solver does not support 'elasticnet' penalty. Defaulting to 'l2'.", bgcolor=ft.Colors.AMBER_ACCENT_200)
        else:  # saga supports all penalties
            self.penalty_dropdown.options = [
                ft.DropdownOption("l1"),
                ft.DropdownOption("l2"),
                ft.DropdownOption("elasticnet"),
                ft.DropdownOption("None"),
            ]
        self.parent.page.update()

    def _fit_intercept_switch_on_change(self, e: ft.ControlEvent) -> None:
        if self.fit_intercept_switch.value:
            self.intercept_scaling_field.disabled = False
        else:
            self.intercept_scaling_field.disabled = True
        self.parent.page.update()

    def _infinite_button_on_click(self, e: ft.ControlEvent) -> None:
        self.C_field.value = "Infinite"
        self.parent.page.update()

    def _c_field_on_click(self, e: ft.ControlEvent) -> None:
        if e.control.value.strip().lower() == "infinite":
            e.control.value = ""
            self.parent.page.update()
    
    def _c_field_on_blur(self, e: ft.ControlEvent) -> None:
        if e.control.value.strip() == "":
            e.control.value = "Infinite"
            self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for logistic regression hyperparameter configuration."""
        
        # Create hyperparameter controls
        self.C_field = ft.TextField(
            label="C (Regularization)",
            value="1.0",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            suffix_icon=ft.IconButton(ft.Icons.ALL_INCLUSIVE, on_click=self._infinite_button_on_click, tooltip="Set to Infinite (no regularization)"),
            on_click=self._c_field_on_click,
            tooltip="Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization. C=np.inf results in unpenalized logistic regression.",
        )
        
        self.max_iter_field = ft.TextField(
            label="Max Iterations",
            value="100",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Maximum number of iterations taken for the solvers to converge.",
        )
        
        self.solver_dropdown = ft.Dropdown(
            label="Solver",
            value="lbfgs",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            on_change=self._solver_on_change,
            options=[
                ft.DropdownOption("lbfgs"),
                ft.DropdownOption("liblinear"),
                ft.DropdownOption("newton-cg"),
                ft.DropdownOption("newton-cholesky"),
                ft.DropdownOption("sag"),
                ft.DropdownOption("saga"),
            ],
            tooltip="""Algorithm to use in the optimization problem. Default is ‘lbfgs’. To choose a solver, you might want to consider the following aspects:
‘lbfgs’ is a good default solver because it works reasonably well for a wide class of problems.
For multiclass problems (n_classes >= 3), all solvers except ‘liblinear’ minimize the full multinomial loss, ‘liblinear’ will raise an error.
‘newton-cholesky’ is a good choice for n_samples >> n_features * n_classes, especially with one-hot encoded categorical features with rare categories. Be aware that the memory usage of this solver has a quadratic dependency on n_features * n_classes because it explicitly computes the full Hessian matrix.
For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;
‘liblinear’ can only handle binary classification by default. To apply a one-versus-rest scheme for the multiclass setting one can wrap it with the OneVsRestClassifier."""
        )
        
        self.penalty_dropdown = ft.Dropdown(
            label="Penalty",
            value="l2",
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            expand=1,
            options=[
                ft.DropdownOption("l2"),
                ft.DropdownOption("None"),
                #? We will dynamically adjust options based on solver selection, but we include all for tooltip explanation
                # ft.DropdownOption("l1"),
                # ft.DropdownOption("elasticnet"),
            ],
            tooltip="""Specify the norm of the penalty:
None: no penalty is added;
'l2': add a L2 penalty term and it is the default choice;
'l1': add a L1 penalty term;
'elasticnet': both L1 and L2 penalty terms are added."""
        )

        self.l1_ratio_field = ft.TextField(
            label="L1 ratio",
            value="None",
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            expand=1,
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            suffix_icon=ft.IconButton(ft.Icons.RESTART_ALT, on_click=lambda e: self._reset_field_to_none(self.l1_ratio_field), tooltip="Reset to None"),
            on_click=self._field_on_click,
            on_blur=self._field_on_blur,
            tooltip="The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Setting l1_ratio=1 gives a pure L1-penalty, setting l1_ratio=0 a pure L2-penalty. Any value between 0 and 1 gives an Elastic-Net penalty of the form l1_ratio * L1 + (1 - l1_ratio) * L2.",
        )

        self.class_weight_dropdown = ft.Dropdown(
            label="Class Weight",
            value="None",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None"),
                ft.DropdownOption("balanced"),
            ],
            tooltip="'balanced' automatically adjusts weights inversely proportional to class frequency. Use for imbalanced datasets",
        )

        self.fit_intercept_switch = ft.Switch(
            label="Fit Intercept",
            value=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            label_position=ft.LabelPosition.RIGHT,
            expand=1,
            tooltip="Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.",
            visible=False,  # Hidden for now to simplify UI, only useful for 'liblinear'
            on_change=self._fit_intercept_switch_on_change
        )

        self.intercept_scaling_field = ft.TextField(
            label="Intercept Scaling",
            value="1.0",
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            expand=1,
            input_filter=ft.InputFilter(r'^$|^(\d+(\.\d*)?|\.\d+)$'),
            visible=False,  # Hidden for now to simplify UI, only useful for 'liblinear' solver when fit_intercept=True
            tooltip="Useful only when the solver liblinear is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight."
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
                                ft.Text("Logistic Regression",
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
                        ft.Row([self.C_field, self.l1_ratio_field]),
                        ft.Row([self.solver_dropdown, self.class_weight_dropdown]),
                        ft.Row([self.penalty_dropdown, self.max_iter_field]),
                        ft.Row([self.fit_intercept_switch, self.intercept_scaling_field]),
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
