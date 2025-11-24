from __future__ import annotations
import matplotlib.pyplot as plt
import flet as ft
from flet.matplotlib_chart import MatplotlibChart
from typing import TYPE_CHECKING
from dataclasses import dataclass
from pandas import DataFrame
import seaborn as sns
import numpy as np
from .base import BaseChart

if TYPE_CHECKING:
    from ..data_science import DataScience


@dataclass
class HeatmapChart(BaseChart):
    df: DataFrame
    parent: DataScience
    page: ft.Page
    
    def __post_init__(self):
        super().__init__()
    
    def _chart_size_on_blur(self, e: ft.ControlEvent) -> None:
        if e.control.value is None:
            e.control.value = 10
            self.page.update()
            return
        elif isinstance(e.control.value, str):
            try:
                if int(e.control.value) < 5:
                    e.control.value = 10
                    self.page.update()
            except ValueError:
                e.control.value = 10
                self.page.update()
    
    def build_chart_control(self) -> ft.Card:
        numeric_df = self.df.select_dtypes(include=["number"])
        
        if len(numeric_df.columns) < 2:
            self.page.open(ft.SnackBar(ft.Text("At least 2 numeric columns required", font_family="SF regular")))
            return
        
        palette = self.parent.palette_dropdown.value if self.parent.palette_dropdown.value else "viridis"
        context = self.parent.context_dropdown.value if self.parent.context_dropdown.value else "notebook"
        style = self.parent.style_dropdown.value if self.parent.style_dropdown.value else "whitegrid"
        
        sns.set_theme(style=style, context=context)
        
        try:
            title_font_size = int(self.parent.title_size.value)
        except ValueError:
            title_font_size = 16
        
        try:
            axes_font_size = int(self.parent.axes_size.value)
        except ValueError:
            axes_font_size = 14
        
        fig = plt.figure(
            figsize=(
                int(self.parent.chart_width.value),
                int(self.parent.chart_height.value)
            )
        )
        
        if self.column.value == "All":
            corr_matrix = numeric_df.corr()
        else:
            corr_matrix = numeric_df.corr()[self.column.value].drop(self.column.value).to_frame()
            corr_matrix = corr_matrix.reindex(
                corr_matrix[self.column.value].sort_values(ascending=False).index
            )

        
        sns.heatmap(
            corr_matrix, 
            annot=self.parent.display_info.value,
            fmt=".2f",
            cmap="coolwarm",
            cbar=self.parent.show_legend_switch.value,
            square=True,
            linewidths=0.5
        )
        
        plt.title("Correlation Heatmap", fontsize=title_font_size)
        plt.xlabel("Features" if self.column.value == "All" else "Target", fontsize=axes_font_size)
        plt.ylabel("Features", fontsize=axes_font_size)
        
        controls = [
            MatplotlibChart(
                figure=fig,
                isolated=True,
                original_size=self.parent.original_size_switch.value,
            )
        ]
        return super().build_chart_control(controls)
    
    def build_chart_settings_control(self) -> ft.Card:
        self.column = ft.Dropdown(
            label="Column",
            value="All",
            label_style=ft.TextStyle(font_family="SF regular"),
            expand=True,
            options=[
                ft.DropdownOption("All", text_style=ft.TextStyle(font_family="SF regular")),
                *[
                    ft.DropdownOption(col, text_style=ft.TextStyle(font_family="SF regular"))
                    for col in self.df.select_dtypes(include=["number"]).columns
                ]
            ]
        )
        
        return ft.Card(
            expand=1,
            content=ft.Container(
                expand=True,
                margin=ft.margin.all(15),
                alignment=ft.alignment.center,
                content=ft.Column(
                    scroll=ft.ScrollMode.AUTO,
                    # expand=True,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Row(
                            controls=[ft.Text("Heatmap configs", font_family="SF thin", size=24, text_align="center", expand=True)]
                        ),
                        ft.Divider(),
                        self.column,
                    ]
                )
            )
        )
