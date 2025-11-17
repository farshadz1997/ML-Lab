from __future__ import annotations
import logging
from pathlib import Path
import flet as ft
import pandas as pd
import numpy as np
from typing import List, Literal, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from helpers import resource_path

if TYPE_CHECKING:
    from pandas._typing import Dtype
    

@dataclass
class DataSet:
    dataset_path: str

    def __post_init__(self):
        self.df: pd.DataFrame = pd.read_csv(self.dataset_path)
        
    @property
    def shape(self) -> tuple[int, int]:
        return self.df.shape
    
    def describe(
        self,
        percentiles: list[float] | None = None,
        include: Literal["all"] | list[Dtype] | None = None,
        exclude: list[Dtype] | None = None
    ) -> pd.DataFrame:
        return self.df.describe(percentiles, include, exclude)
    
    def custom_describe(self) -> pd.DataFrame:
        variables = []
        dtypes = []
        count = []
        unique = []
        missing = []
        for item in self.df.columns:
            variables.append(item)
            dtypes.append(self.df[item].dtype)
            count.append(len(self.df[item]))
            unique.append(len(self.df[item].unique()))
            missing.append(self.df[item].isna().sum())
        output = pd.DataFrame({
            'Variable': variables, 
            'Dtype': dtypes,
            'Count': count,
            'Unique': unique,
            'Missing value': missing
        })    
        return output

    def calculate_missing_percent(self) -> pd.DataFrame:
        percent_missing = self.df.isnull().sum() * 100 / len(self.df)
        missing_value_df = pd.DataFrame(
            {
                'column_name': self.df.columns,
                'percent_missing': percent_missing
            }
        )
        percent = pd.DataFrame(missing_value_df)
        return percent


@dataclass
class AppLayout:
    page: ft.Page
    dataset: DataSet | None = None
    
    def __post_init__(self):
        self.page.title = "Data sceience and ML helper"
        self.page.fonts = {
            "SF thin": "fonts/SFUIDisplay-Thin.otf",
            "SF regular": "fonts/SF-Pro-Display-Regular.otf",
            "SF light": "fonts/SFUIText-Light.otf"
        }
        self.page.on_view_pop = self.view_pop
        self.page.on_route_change = self.on_route_change
        self.page.window.center()
        self.page.window.min_height = 1000
        self.page.window.min_width = 1000
        self.page.window.width = 1000
        self.page.window.height = 1000
        self.home = Home(self, self.page)
        self.data_science = DataScience(self, self.page)
        self.model_factory = ModelFactory(self, self.page)
        self.page.navigation_bar = ft.NavigationBar(
            on_change=self.on_navigation_change,
            destinations=[
                ft.NavigationBarDestination(icon=ft.Icons.DATA_OBJECT, label="Dataset overview"),
                ft.NavigationBarDestination(icon=ft.Icons.ANALYTICS, label="Data Science", disabled=True),
                ft.NavigationBarDestination(
                    icon=ft.Icons.MODEL_TRAINING,
                    label="Model Factory",
                    disabled=True
                ),
            ]
        )
        self.page.controls = [self.home.build_controls()]
        self.page.update()
    
    def on_navigation_change(self, e: ft.ControlEvent):
        nav_index = int(e.data)
        if nav_index == 0:
            column = self.home.build_controls()
        elif nav_index == 1:
            column = self.data_science.build_controls()
        elif nav_index == 2:
            column = self.model_factory.build_controls()
        else:
            column = self.home.build_controls()
        self.page.controls.clear()
        self.page.controls = [column]
        self.page.update()
    
    def on_route_change(self, e: ft.RouteChangeEvent):
        print(e.route)
        
    def view_pop(self, view):
        self.page.views.pop()
        top_view = self.page.views[-1]
        self.page.go(top_view.route)
        
        
@dataclass
class Home:
    parent: AppLayout
    page: ft.Page
    column: ft.Column | None = None
    
    def _pick_dataset_file_result(self, e: ft.FilePickerResultEvent) -> None:
        if e.files:
            self.dataset_path_field.value = e.files[0].path
            self.parent.dataset = DataSet(e.files[0].path)
            self.display_tables_options_row.visible = True
            self.close_dataset_btn.visible = True
            self.open_dataset_button.height = 65
            self.page.navigation_bar.destinations[1].disabled = False
            self.page.navigation_bar.destinations[2].disabled = False
            self.page.update()
            self._display_datatable(self.parent.dataset.describe(include="all"), "Pandas describe")
            
    def _display_datatable(self, df: pd.DataFrame, title: str = "Describe") -> None:
        df = df.copy()
        df.replace(np.nan, "NaN", inplace=True)
        datatable = ft.DataTable(
            columns=[
                ft.DataColumn(label=ft.Text("")),
                *[ft.DataColumn(
                    label=ft.Text(col),
                    numeric=pd.api.types.is_numeric_dtype(df[col])
                ) for col in df.columns]
            ],
            rows=[
                ft.DataRow(
                    cells=[
                        ft.DataCell(content=ft.Text(stat_name)),
                        *[
                            ft.DataCell(content=ft.Text(round(row.iloc[i], 2) if isinstance(row.iloc[i], (int, float)) else row.iloc[i]))
                            for i in range(len(df.columns))
                        ]
                    ]
                ) for stat_name, row in df.iterrows()
            ]
        )
        self.datatable_card.visible = True
        self.datatable_container.content = ft.Column(
            scroll=ft.ScrollMode.ALWAYS,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Row([ft.Text(title, font_family="SF thin", size=24, expand=True, text_align="center")]),
                ft.Divider(),
                ft.Row(controls=[datatable], scroll=ft.ScrollMode.AUTO)
            ]
        )
        self.page.update()

    def close_dataset(self, e: ft.ControlEvent | None = None):
        self.parent.dataset.df = None
        self.dataset_path_field.value = None
        self.display_tables_options_row.visible = False
        self.datatable_card.visible = False
        self.datatable_container.content = None
        self.open_dataset_button.height = 50
        self.close_dataset_btn.visible = False
        self.page.navigation_bar.destinations[1].disabled = True
        self.page.navigation_bar.destinations[2].disabled = True
        self.page.update()

    def build_controls(self) -> ft.Column:
        if self.column:
            return self.column
                
        pick_files_dialog = ft.FilePicker(on_result=self._pick_dataset_file_result)
        self.parent.page.overlay.append(pick_files_dialog)
        self.close_dataset_btn = ft.IconButton(
            icon=ft.Icons.CLOSE, visible=False, on_click=self.close_dataset
        )
        self.dataset_path_field = ft.TextField(
            label="Dataset file",
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            read_only=True,
            expand=6,
            suffix=self.close_dataset_btn
        )
        self.open_dataset_button = ft.FilledButton(
            text="Open dataset",
            icon=ft.Icons.FILE_OPEN,
            expand=1,
            height=50,
            on_click=lambda _: pick_files_dialog.pick_files(
                allow_multiple=False,
                allowed_extensions=["csv"]
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        self.pandas_describe_btn = ft.FilledButton(
            text="Pandas describe",
            icon=ft.Icons.ANALYTICS,
            expand=1,
            on_click=lambda _: self._display_datatable(
                self.parent.dataset.describe(include="all"),
                "Pandas describe"
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        self.custom_describe_btn = ft.FilledButton(
            text="Custom describe",
            icon=ft.Icons.ANALYTICS,
            expand=1,
            on_click=lambda _: self._display_datatable(
                self.parent.dataset.custom_describe(),
                "Custom describe"
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        self.missing_percent_btn = ft.FilledButton(
            text="Missing values table",
            icon=ft.Icons.ANALYTICS,
            expand=1,
            on_click=lambda _: self._display_datatable(
                self.parent.dataset.calculate_missing_percent(),
                "Missing values percentage"
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        self.display_tables_options_row = ft.Row(
            visible=False,
            controls=[
                ft.Text("Options:", font_family="SF thin", size=20, expand=1),
                self.pandas_describe_btn,
                self.custom_describe_btn,
                self.missing_percent_btn
            ]
        )
        self.datatable_container = ft.Container(margin=ft.margin.all(15), height=600)
        self.datatable_card = ft.Card(visible=False, content=self.datatable_container)
        self.column = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[
                ft.Text("Dataset Overview", font_family="SF thin", size=30, expand=True, text_align="center"),
                ft.Divider(),
                ft.Card(
                    content=ft.Container(
                        margin=ft.margin.all(15),
                        content=ft.Column(
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                ft.Row(
                                    controls=[
                                        self.dataset_path_field,
                                        self.open_dataset_button
                                    ],
                                ),
                                self.display_tables_options_row
                            ]
                        )
                    )
                ),
                self.datatable_card
            ]
        )
        return self.column
        
@dataclass
class DataScience:
    parent: AppLayout
    page: ft.Page
    column: ft.Column | None = None
    
    def build_controls(self) -> ft.Column:
        if self.column:
            return self.column
        self.column = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[
                ft.Text("Data Science", expand=True, text_align="center"),
                ft.Divider()
            ]
        )
        return self.column


@dataclass
class ModelFactory:
    parent: AppLayout
    page: ft.Page
    column: ft.Column | None = None
    
    def build_controls(self) -> ft.Column:
        if self.column:
            return self.column
        self.column = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[
                ft.Text("Modeling", expand=True, text_align="center"),
                ft.Divider()
            ]
        )
        return self.column


if __name__ == "__main__":
    ft.app(AppLayout, assets_dir=resource_path("assets"))
