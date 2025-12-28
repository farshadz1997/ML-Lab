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
    from ..data_visualization import DataVisualization


@dataclass
class BarChart(BaseChart):
    df: DataFrame
    parent: DataVisualization
    page: ft.Page
    
    def __post_init__(self):
        super().__init__()
    
    def _xaxis_on_change(self, e: ft.ControlEvent) -> None:
        if self.y_dropdown.value and e.control.value == self.y_dropdown.value:
            self.y_dropdown.value = "None"
            self.page.update()
    
    def _yaxis_on_change(self, e: ft.ControlEvent) -> None:
        if self.x_dropdown.value and e.control.value == self.x_dropdown.value:
            self.x_dropdown.value = "None"
            self.page.update()
    
    def build_chart_control(self) -> ft.Card:
        if any(selection.value in (None, "None") for selection in [self.x_dropdown, self.y_dropdown]):
            self.page.open(ft.SnackBar(ft.Text("X, Y is required", font_family="SF regular")))
            return
        
        x = self.x_dropdown.value
        y = self.y_dropdown.value
        
        palette = self.parent.palette_dropdown.value if self.parent.palette_dropdown.value else "deep"
        context = self.parent.context_dropdown.value if self.parent.context_dropdown.value else "notebook"
        style = self.parent.style_dropdown.value if self.parent.style_dropdown.value else "whitegrid"
        
        local_df = self.df.groupby(x).agg(result=(y, self.aggregated_by.value)).sort_values(by="result", ascending=False)
        
        sns.set_theme(style=style, context=context, palette=palette)
        
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
        
        ax = sns.barplot(
            x=local_df["result"], y=local_df.index, hue=local_df.index,
            palette=palette,
            legend="auto" if self.parent.show_legend_switch.value else False
        )
        ax.set_title(f"Bar chart: {y} by {x} ({self.aggregated_by.value})", fontsize=title_font_size)
        ax.set_xlabel(x, fontsize=axes_font_size)
        ax.set_ylabel(y, fontsize=axes_font_size)
        ax.grid(visible=True, axis="y")
        
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
        categorical_cols = self.df.columns[self.df.nunique() <= 30].tolist()
        object_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        categorical_cols = list(set(categorical_cols + object_cols))
        
        self.x_dropdown = ft.Dropdown(
            label="X (Category)",
            expand=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None", text_style=ft.TextStyle(font_family="SF regular")),
                *[ft.DropdownOption(
                    col, text_style=ft.TextStyle(font_family="SF regular")
                ) for col in categorical_cols]
            ],
            on_change=self._xaxis_on_change
        )
        
        self.y_dropdown = ft.Dropdown(
            label="Y (Numeric)",
            expand=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None", text_style=ft.TextStyle(font_family="SF regular")),
                *[ft.DropdownOption(
                    col, text_style=ft.TextStyle(font_family="SF regular")
                ) for col in numeric_cols]
            ],
            on_change=self._yaxis_on_change
        )
        
        self.aggregated_by = ft.Dropdown(
            label="Aggregated by",
            value="sum",
            expand=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[ft.DropdownOption(agg, text_style=ft.TextStyle(font_family="SF regular")) for agg in ["sum", "count", "mean", "median", "min", "max"]]
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
                            controls=[ft.Text("Bar chart configs", font_family="SF thin", size=24, text_align="center", expand=True)]
                        ),
                        ft.Divider(),
                        self.x_dropdown,
                        self.y_dropdown,
                        self.aggregated_by,
                    ]
                )
            )
        )
