from __future__ import annotations
from typing import List, Literal, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass, field
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import make_pipeline


if TYPE_CHECKING:
    from ..model_factory import ModelFactory
    from ..layout import AppLayout
    

@dataclass
class LinearRegressionModel:
    parent: ModelFactory
    df: DataFrame
    
    def __post_init__(self):
        self.df = self.df.copy()
    
    def _prepare_data(self):
        try:
            df = self.df.copy()
            target_name = self.parent.target_column_dropdown.value
            # Get feature and target columns
            feature_cols = [col for col in df.columns if col != target_name]
            
            X = df[feature_cols]
            y = df[target_name]
            
            # Encode categorical features
            self.label_encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            
            # Encode target if categorical
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                self.label_encoders['target'] = le
            
            # Split data
            test_size = float(self.parent.test_size_field.value)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y.values.reshape(-1, 1), test_size=test_size, random_state=42
            )
            
            # Scale features
            if self.parent.scaler_dropdown.value == "standard_scaler":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            elif self.parent.scaler_dropdown.value == "minmax_scaler":
                scaler = MinMaxScaler(feature_range=(0, 1))
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            return X_train, X_test, y_train, y_test, feature_cols
        
        except Exception as e:
            self.parent.page.open(ft.SnackBar(ft.Text(f"Data preparation error: {str(e)}", font_family="SF regular")))
            return None
        
    def _train_and_evaluate_model(self, e: ft.ControlEvent):
        try:
            e.control.disabled = True
            self.parent.page.update()
            data = self._prepare_data()
            if data is None:
                return None
            X_train, X_test, y_train, y_test, feature_cols = data
            linear_model = LinearRegression(
                fit_intercept=self.fit_intercept_switch.value,
                positive=self.positive_switch.value
            )
            linear_model.fit(X_train, y_train)
            y_pred = linear_model.predict(X_test)
            # evaluation
            r2_score = metrics.r2_score(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = metrics.root_mean_squared_error(y_test, y_pred)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            result =(
            f"**Intercept:** {linear_model.intercept_}\n\n"
            f"**R2 Score:** {r2_score:.4f}\n\n"
            f"**Mean squared error:** {mse:.4f}\n\n"
            f"**Root mean squared error:** {rmse:.4f}\n\n"
            f"**Mean absolute error:** {mae:.4f}"
            )
            evaluation_dialog = ft.AlertDialog(
                modal=True,
                title=ft.Row([ft.Text("Linear regression result", font_family="SF thin", size=20, expand=1, text_align="center")]),
                content=ft.Markdown(result, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB),
                actions_alignment=ft.MainAxisAlignment.CENTER,
                actions=[
                    ft.Row([ft.FilledButton(
                        text="Ok",
                        expand=True,
                        on_click=lambda _: self.parent.page.close(evaluation_dialog)
                    )])
                ]
            )
            self.parent.page.open(evaluation_dialog)
        except Exception as e:
            self.parent.page.open(ft.SnackBar(ft.Text(f"Train and model evaluation failed: {str(e)}", font_family="SF regular")))
            return None
        finally:
            e.control.disabled = False
            self.parent.page.update()
        
    def build_model_control(self) -> ft.Card:
        self.target_col_dropdown = ft.Dropdown(
            label="Target",
            value=self.df.columns.tolist()[0],
            label_style=ft.TextStyle(font_family="SF regular"),
            expand=1,
            options=[ft.DropdownOption(col, text_style=ft.TextStyle(font_family="SF regular")) for col in self.df.columns.tolist()]
        )
        self.fit_intercept_switch = ft.Switch(
            label="Fit intercept",
            tooltip="Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).",
            label_position=ft.LabelPosition.RIGHT,
            label_style=ft.TextStyle(font_family="SF regular"),
            value=True,
        )
        self.positive_switch = ft.Switch(
            label="Positive",
            tooltip="When set to True, forces the coefficients to be positive. This option is only supported for dense arrays",
            label_position=ft.LabelPosition.RIGHT,
            label_style=ft.TextStyle(font_family="SF regular"),
            value=False,
        )
        train_btn = ft.FilledButton(
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
            expand=1,
            content=ft.Container(
                expand=True,
                margin=ft.margin.all(15),
                alignment=ft.alignment.center,
                content=ft.Column(
                    scroll=ft.ScrollMode.AUTO,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        ft.Row(
                            controls=[ft.Text("Linear regression", font_family="SF thin", size=24, text_align="center", expand=True)]
                        ),
                        ft.Divider(),
                        self.fit_intercept_switch,
                        self.positive_switch,
                        ft.Row([train_btn])
                    ]
                )
            )
        )
        