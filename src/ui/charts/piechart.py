from __future__ import annotations
import matplotlib.pyplot as plt
import flet as ft
from flet.matplotlib_chart import MatplotlibChart
from typing import TYPE_CHECKING
from dataclasses import dataclass
from pandas import DataFrame
import seaborn as sns
from .base import BaseChart

if TYPE_CHECKING:
    from ..data_science import DataScience


@dataclass
class PieChart(BaseChart):
    df: DataFrame
    parent: DataScience
    page: ft.Page
    
    def __post_init__(self):
        super().__init__()
    
    def _chart_size_on_blur(self, e: ft.ControlEvent) -> None:
        if e.control.value is None:
            e.control.value = 8
            self.page.update()
            return
        elif isinstance(e.control.value, str):
            try:
                if int(e.control.value) < 5:
                    e.control.value = 8
                    self.page.update()
            except ValueError:
                e.control.value = 8
                self.page.update()
    
    def build_chart_control(self) -> ft.Card:
        if self.column_dropdown.value in (None, "None"):
            self.page.open(ft.SnackBar(ft.Text("Labels column is required", font_family="SF regular")))
            return
        
        labels = self.df[self.column_dropdown.value].value_counts().index.astype(str)
        values = self.df[self.column_dropdown.value].value_counts().values
        
        palette = self.parent.palette_dropdown.value if self.parent.palette_dropdown.value else "deep"
        
        sns.set_theme(palette=palette)
        
        fig = plt.figure(
            figsize=(
                int(self.parent.chart_width.value),
                int(self.parent.chart_height.value)
            )
        )
        
        plt.pie(
            values, 
            labels=labels,
            autopct="%1.1f%%" if self.parent.display_info.value else "",
            startangle=90,
            textprops={"fontsize": 10}
        )
        
        plt.title(f"Pie chart: {self.column_dropdown.value}")
        
        controls = [
            MatplotlibChart(
                figure=fig,
                isolated=True,
                original_size=self.parent.original_size_switch.value,
            )
        ]
        return super().build_chart_control(controls)
    
    def build_chart_settings_control(self) -> ft.Card:
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        all_cols = self.df.columns.tolist()
        
        self.column_dropdown = ft.Dropdown(
            label="Column",
            expand=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption(
                    col, text_style=ft.TextStyle(font_family="SF regular")
                ) for col in all_cols
            ],
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
                            controls=[ft.Text("Pie chart configs", font_family="SF thin", size=24, text_align="center", expand=True)]
                        ),
                        ft.Divider(),
                        self.column_dropdown,
                    ]
                )
            )
        )
