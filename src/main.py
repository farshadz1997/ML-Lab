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
        self.home = Home(self, self.page)
        self.data_science = DataScience(self, self.page)
        self.model_factory = ModelFactory(self, self.page)
        self.page.navigation_bar = ft.NavigationBar(
            on_change=self.on_navigation_change,
            destinations=[
                ft.NavigationBarDestination(icon=ft.Icons.DATA_OBJECT, label="Dataset overview"),
                ft.NavigationBarDestination(icon=ft.Icons.ANALYTICS, label="Data Science"),
                ft.NavigationBarDestination(
                    icon=ft.Icons.MODEL_TRAINING,
                    label="Model Factory",
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
            self.page.update()
            self.parent.dataset = DataSet(e.files[0].path)
            self._display_datatable(self.parent.dataset.describe(include="all"))
            
    def _display_datatable(self, df: pd.DataFrame) -> None:
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
                            ft.DataCell(content=ft.Text(row.iloc[i]))
                            for i in range(len(df.columns))
                        ]
                    ]
                ) for stat_name, row in df.iterrows()
            ]
        )
        self.datatable_card.visible = True
        self.datatable_container.content = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            controls=[ft.Row(controls=[datatable], scroll=ft.ScrollMode.AUTO)]
        )
        self.page.update()

            
    def build_controls(self) -> ft.Column:
        if self.column:
            return self.column
                
        pick_files_dialog = ft.FilePicker(on_result=self._pick_dataset_file_result)
        self.parent.page.overlay.append(pick_files_dialog)
        self.dataset_path_field = ft.TextField(
            label="Dataset file",
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            read_only=True,
            expand=6
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
        self.datatable_container = ft.Container(margin=ft.margin.all(15))
        self.datatable_card = ft.Card(visible=False, content=self.datatable_container)
        self.column = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[
                ft.Text("Dataset Overview", font_family="SF thin", size=20, expand=True, text_align="center"),
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
