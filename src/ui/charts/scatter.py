from __future__ import annotations
import matplotlib.pyplot as plt
import flet as ft
from flet.matplotlib_chart import MatplotlibChart
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from pandas import DataFrame
import seaborn as sns
from .base import BaseChart

if TYPE_CHECKING:
    from ..data_science import DataScience


@dataclass
class ScatterChart(BaseChart):
    df: DataFrame
    parent: DataScience
    page: ft.Page
    
    def __post_init__(self):
        super().__init__()
    
    def _xaxis_on_change(self, e: ft.ControlEvent) -> None:
        if e.control.value == self.y_dropdown.value:
            self.y_dropdown.value = "None"
            self.page.update()
    
    def _yaxis_on_change(self, e: ft.ControlEvent) -> None:
        if e.control.value == self.x_dropdown.value:
            self.x_dropdown.value = "None"
            self.page.update()
    
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
        if any(selection.value in (None, "None") for selection in [self.x_dropdown, self.y_dropdown]):
            self.page.open(ft.SnackBar(ft.Text("X, Y is required", font_family="SF regular")))
            return
        x = self.x_dropdown.value
        y = self.y_dropdown.value
        color = self.color_dropdown.value if self.color_dropdown.value not in ("None", None) else None
        size = self.size_dropdown.value if self.size_dropdown.value not in ("None", None) else None
        symbol = self.symbol_dropdown.value if self.symbol_dropdown.value not in ("None", None) else None
        
        # Apply selected palette, context, and style
        palette = self.parent.palette_dropdown.value if self.parent.palette_dropdown.value else "deep"
        context = self.parent.context_dropdown.value if self.parent.context_dropdown.value else "notebook"
        style = self.parent.style_dropdown.value if self.parent.style_dropdown.value else "whitegrid"
        
        try:
            title_font_size = int(self.parent.title_size.value)
        except ValueError:
            title_font_size = 16
        
        try:
            axes_font_size = int(self.parent.axes_size.value)
        except ValueError:
            axes_font_size = 14
        
        sns.set_theme(style=style, context=context, palette=palette)
        
        fig = plt.figure(
            figsize=(
                int(self.parent.chart_width.value),
                int(self.parent.chart_height.value)
            )
        )
        ax = sns.scatterplot(
            self.df, x=x, y=y, style=symbol, hue=color, size=size,
            legend="auto" if self.parent.show_legend_switch.value else False
        )
        ax.set_title(f"Scatter plot {x} VS {y}", fontsize=title_font_size)
        ax.set_xlabel(x, fontsize=axes_font_size)
        ax.set_ylabel(y, fontsize=axes_font_size)
        ax.grid(visible=True)
        controls = [
            MatplotlibChart(
                figure=fig,
                isolated=True,
                original_size=self.parent.original_size_switch.value,
            )
        ]
        return super().build_chart_control(controls)
               
    def build_chart_settings_control(self) -> ft.Card:
        self.x_dropdown = ft.Dropdown(
            label="X",
            expand=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None"),
                *[ft.DropdownOption(
                    col, text_style=ft.TextStyle(font_family="SF regular")
                ) for col in self.df.columns]
            ],
            on_change=self._xaxis_on_change
        )
        self.y_dropdown = ft.Dropdown(
            label="Y",
            expand=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None"),
                *[ft.DropdownOption(
                    col, text_style=ft.TextStyle(font_family="SF regular")
                ) for col in self.df.columns]
            ],
            on_change=self._yaxis_on_change
        )
        self.color_dropdown = ft.Dropdown(
            label="Color",
            value="None",
            expand=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None"),
                *[ft.DropdownOption(
                    col, text_style=ft.TextStyle(font_family="SF regular")
                ) for col in self.df.columns]
            ],
        )
        self.symbol_dropdown = ft.Dropdown(
            label="Symbol",
            value="None",
            expand=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None"),
                *[ft.DropdownOption(
                    col, text_style=ft.TextStyle(font_family="SF regular")
                ) for col in self.df.columns]
            ],
        )
        self.size_dropdown = ft.Dropdown(
            label="Size",
            value="None",
            expand=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None"),
                *[ft.DropdownOption(
                    col, text_style=ft.TextStyle(font_family="SF regular")
                ) for col in self.df.columns]
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
                    # alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        ft.Row(
                            controls=[ft.Text("Scatter plot configs", font_family="SF thin", size=24, text_align="center", expand=True)]
                        ),
                        ft.Divider(),
                        self.x_dropdown,
                        self.y_dropdown,
                        self.color_dropdown,
                        self.symbol_dropdown,
                        self.size_dropdown
                    ]
                )
            )
        )
        