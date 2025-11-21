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
        palette = self.palette_dropdown.value if self.palette_dropdown.value else "deep"
        context = self.context_dropdown.value if self.context_dropdown.value else "notebook"
        style = self.style_dropdown.value if self.style_dropdown.value else "whitegrid"
        
        sns.set_theme(style=style, context=context, palette=palette)
        
        fig = plt.figure(
            figsize=(
                int(self.chart_width.value),
                int(self.chart_height.value)
            )
        )
        ax = sns.scatterplot(
            self.df, x=x, y=y, style=symbol, hue=color, size=size,
            legend="auto" if self.show_legend_switch.value else False
        )
        ax.set_title(f"Scatter plot {x} VS {y}", fontsize=16)
        ax.set_xlabel(x, fontsize=14)
        ax.set_ylabel(y, fontsize=14)
        ax.grid(visible=self.grid_switch.value)
        # if self.show_legend_switch.value:
        #     ax.legend(bbox_to_anchor=(1, 1))
        controls = [
            MatplotlibChart(
                figure=fig,
                isolated=True,
                original_size=self.original_size_switch.value,
            )
        ]
        return super().build_chart_control(controls)
               
    def build_chart_settings_control(self) -> ft.Card:
        self.x_dropdown = ft.Dropdown(
            label="X",
            width=200,
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
            width=200,
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
            width=200,
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
            width=200,
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
            width=200,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("None"),
                *[ft.DropdownOption(
                    col, text_style=ft.TextStyle(font_family="SF regular")
                ) for col in self.df.columns]
            ],
        )
        self.palette_dropdown = ft.Dropdown(
            value="deep",
            label="Palette",
            width=200,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption(
                    name, text_style=ft.TextStyle(font_family="SF regular")
                ) for name in ["deep", "muted", "pastel", "bright", "dark", "colorblind", "husl", "rocket", "mako", "flare", "crest"]
            ],
        )
        self.context_dropdown = ft.Dropdown(
            value="notebook",
            label="Context",
            width=200,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption(
                    ctx, text_style=ft.TextStyle(font_family="SF regular")
                ) for ctx in ["paper", "notebook", "talk", "poster"]
            ],
        )
        self.style_dropdown = ft.Dropdown(
            value="whitegrid",
            label="Style",
            width=200,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption(
                    style, text_style=ft.TextStyle(font_family="SF regular")
                ) for style in ["darkgrid", "whitegrid", "dark", "white", "ticks"]
            ],
        )
        self.chart_width = ft.TextField(
            value=10,
            label="Width",
            width=150,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            on_blur=self._chart_size_on_blur
        )
        self.chart_height = ft.TextField(
            value=10,
            width=150,
            label="Height",
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            on_blur=self._chart_size_on_blur
        )
        self.show_legend_switch = ft.Switch(
            value=True,
            label="Show legend",
            label_style=ft.TextStyle(font_family="SF regular"),
            label_position=ft.LabelPosition.RIGHT,
        )
        self.grid_switch = ft.Switch(
            value=True,
            label="Grid",
            label_position=ft.LabelPosition.RIGHT,
            label_style=ft.TextStyle(font_family="SF regular")
        )
        self.original_size_switch = ft.Switch(
            value=True,
            label="Original size",
            tooltip="Whether to display original size of chart",
            label_position=ft.LabelPosition.RIGHT,
            label_style=ft.TextStyle(font_family="SF regular")
        )
        return ft.Card(
            expand=True,
            content=ft.Container(
                expand=True,
                margin=ft.margin.all(15),
                alignment=ft.alignment.center,
                content=ft.Column(
                    scroll=ft.ScrollMode.AUTO,
                    # alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Row(
                            controls=[ft.Text("Scatter plot configs", font_family="SF thin", size=24, text_align="center", expand=True)]
                        ),
                        ft.Divider(),
                        ft.Row(
                            scroll=ft.ScrollMode.AUTO,
                            alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                            vertical_alignment=ft.CrossAxisAlignment.START,
                            controls=[
                                ft.Column(
                                    controls=[
                                        ft.Row([ft.Text("Display fields", font_family="SF thin", size=20, expand=True, text_align="center")]),
                                        self.x_dropdown,
                                        self.y_dropdown,
                                        self.color_dropdown,
                                        self.symbol_dropdown,
                                        self.size_dropdown
                                    ]
                                ),
                                ft.Column(
                                    controls=[
                                        ft.Text("Customization", font_family="SF thin", size=20, expand=True, text_align="center"),
                                        self.palette_dropdown,
                                        self.context_dropdown,
                                        self.style_dropdown,
                                    ]
                                ),
                                ft.Column(
                                    controls=[
                                        ft.Text("Figure size", font_family="SF thin", size=20, expand=True, text_align="center"),
                                        self.chart_width,
                                        self.chart_height,
                                        self.original_size_switch
                                    ]
                                ),
                                ft.VerticalDivider(),
                                ft.Column(
                                    controls=[
                                        ft.Text("Guides", font_family="SF thin", size=20, expand=True, text_align="center"),
                                        self.show_legend_switch,
                                        self.grid_switch
                                    ]
                                )
                            ],
                        ),
                    ]
                )
            )
        )
        