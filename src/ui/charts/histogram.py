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
class HistogramChart(BaseChart):
    df: DataFrame
    parent: DataVisualization
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
    
    def _bins_on_blur(self, e: ft.ControlEvent) -> None:
        if e.control.value is None or e.control.value == "":
            e.control.value = "10"
            self.page.update()
            return
        elif isinstance(e.control.value, str):
            try:
                if int(e.control.value) < 1:
                    e.control.value = "10"
                    self.page.update()
            except ValueError:
                e.control.value = "10"
                self.page.update()
    
    def build_chart_control(self) -> ft.Card:
        if self.x_dropdown.value in (None, "None"):
            self.page.open(ft.SnackBar(ft.Text("X axis is required", font_family="SF regular")))
            return
        
        x = self.x_dropdown.value
        bins = int(self.bins_field.value) if self.bins_field.value else 10
        
        palette = self.parent.palette_dropdown.value if self.parent.palette_dropdown.value else "deep"
        context = self.parent.context_dropdown.value if self.parent.context_dropdown.value else "notebook"
        style = self.parent.style_dropdown.value if self.parent.style_dropdown.value else "whitegrid"
        
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
        ax = sns.histplot(
            self.df[x], bins=bins, kde=self.kde_switch.value,
            legend="auto" if self.parent.show_legend_switch.value else False
        )
        ax.set_title(f"Histogram of {x}", fontsize=title_font_size)
        ax.set_xlabel(x, fontsize=axes_font_size)
        ax.set_ylabel("Frequency", fontsize=axes_font_size)
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
        
        self.x_dropdown = ft.Dropdown(
            label="X",
            expand=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None"),
                *[ft.DropdownOption(
                    col, text_style=ft.TextStyle(font_family="SF regular")
                ) for col in numeric_cols]
            ],
        )
        
        self.bins_field = ft.TextField(
            value="10",
            label="Bins",
            expand=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            on_blur=self._bins_on_blur
        )
        
        self.kde_switch = ft.Switch(
            value=False,
            label="KDE",
            label_style=ft.TextStyle(font_family="SF regular"),
            label_position=ft.LabelPosition.RIGHT,
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
                            controls=[ft.Text("Histogram configs", font_family="SF thin", size=24, text_align="center", expand=True)]
                        ),
                        ft.Divider(),
                        self.x_dropdown,
                        self.bins_field,
                        self.kde_switch,
                    ]
                )
            )
        )
