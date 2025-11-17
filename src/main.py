from __future__ import annotations
import logging
from pathlib import Path
import flet as ft
import pandas as pd
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
        percentiles: list[int],
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
        self.home = Home(self)
        self.data_science = DataScience(self)
        self.model_factory = ModelFactory(self)
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
    column: ft.Column | None = None
    
    def build_controls(self) -> ft.Column:
        if self.column:
            return self.column
        self.column = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[
                ft.Text("Home", expand=True, text_align="center"),
                ft.Divider()
            ]
        )
        return self.column
        
@dataclass
class DataScience:
    parent: AppLayout
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


def main(page: ft.Page):
    DATASET: DataSet | None = None
    
    page.title = "NavigationBar Example"
    page.navigation_bar = ft.NavigationBar(
        on_change=lambda e: print(e.data),
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.DATA_OBJECT, label="Describe data"),
            ft.NavigationBarDestination(icon=ft.Icons.ANALYTICS, label="Data Science"),
            ft.NavigationBarDestination(
                icon=ft.Icons.MODEL_TRAINING,
                label="Modeling",
            ),
        ]
    )
    
    def pick_dataset_file_result(e: ft.FilePickerResultEvent):
        nonlocal DATASET
        if e.files:
            dataset_path.value = e.files[0].path
            DATASET = DataSet(e.files[0].path)
            describe_df = DATASET.df.describe(include="all")
            datatable_container.content = ft.DataTable(
                columns=[ft.DataColumn(label=col) for col in describe_df.columns],
                rows=[
                    ft.DataRow(
                        cells=[
                            ft.DataCell(content=ft.Text(row[i])) for i in range(len(describe_df.columns))
                        ]
                    ) for row_idx, row in describe_df.itertuples()
                ]
            )
            page.update()
            
    pick_files_dialog = ft.FilePicker(on_result=pick_dataset_file_result)
    page.overlay.append(pick_files_dialog)
    dataset_path = ft.TextField(
        label="Dataset file",
        read_only=True,
        # expand=True,
        on_click=lambda _: pick_files_dialog.pick_files(
            allow_multiple=False,
            allowed_extensions=["csv"]
        )
    )
    
    # counter = ft.Text("0", size=50, data=0)
    datatable_container = ft.Container(alignment=ft.alignment.center)

    # page.floating_action_button = ft.FloatingActionButton(
    #     icon=ft.Icons.ADD, on_click=increment_click
    # )
    page.add(
        ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                dataset_path,
                datatable_container
            ]
        )
    )


if __name__ == "__main__":
    ft.app(AppLayout, assets_dir=resource_path("assets"))
