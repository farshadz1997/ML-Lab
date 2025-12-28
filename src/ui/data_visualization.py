from __future__ import annotations
import flet as ft
from typing import Literal, TYPE_CHECKING, Callable
from dataclasses import dataclass, field
from .charts import (
    ScatterChart, HistogramChart, BoxPlotChart,
    HeatmapChart, PieChart, BarChart
)


if TYPE_CHECKING:
    from .layout import AppLayout

VIZ_TYPE = Literal["Histogram", "Box", "Scatter", "Heatmap", "Pie", "Bar"]

@dataclass
class DataVisualization:
    parent: AppLayout
    page: ft.Page
    column: ft.Column | None = field(default=None, init=False)
    
    def _on_viz_type_change(self, e: ft.ControlEvent) -> None:
        """Handle visualization type change"""
        viz_type = e.control.value
        self._update_viz_options(viz_type)
    
    def _add_chart_control(self, e: ft.ControlEvent, chart_function: Callable) -> None:
        try:
            if self.chart_card in self.column.controls:
                self.column.controls.remove(self.chart_card)
            e.control.disabled = True
            self.page.navigation_bar.disabled = True
            self.page.update()
            chart_card = chart_function()
            if chart_card is None:
                return
            self.chart_card = chart_card
            self.column.controls.append(self.chart_card)
            self.page.update()
        except Exception as e:
            self.page.open(ft.SnackBar(ft.Text(e, font_family="SF regular")))
        finally:
            e.control.disabled = False
            self.page.navigation_bar.disabled = False
            self.page.update()
    
    def _update_viz_options(self, viz_type: VIZ_TYPE) -> None:
        """Update available options based on visualization type"""
        if self.column and self.parent.dataset:
            if self.config_card in self.column.controls[2].controls:
                self.column.controls[2].controls.remove(self.config_card)
            if self.chart_card in self.column.controls:
                self.column.controls.remove(self.chart_card)
        generate_chart_button = ft.FilledButton(
            text="Generate Visualization",
            icon=ft.Icons.SHOW_CHART,
            on_click=None,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        if viz_type == "Histogram":
            chart_config = HistogramChart(self.parent.dataset.df, self, self.page)
            self.config_card = chart_config.build_chart_settings_control()
            generate_chart_button.on_click = lambda e: self._add_chart_control(e, chart_config.build_chart_control)
            self.config_card.content.content.controls.append(
                ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                    controls=[generate_chart_button]
                )
            )
        elif viz_type == "Box":
            chart_config = BoxPlotChart(self.parent.dataset.df, self, self.page)
            self.config_card = chart_config.build_chart_settings_control()
            generate_chart_button.on_click = lambda e: self._add_chart_control(e, chart_config.build_chart_control)
            self.config_card.content.content.controls.append(
                ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                    controls=[generate_chart_button]
                )
            )
        elif viz_type == "Scatter":
            chart_config = ScatterChart(self.parent.dataset.df, self, self.page)
            self.config_card = chart_config.build_chart_settings_control()
            generate_chart_button.on_click = lambda e: self._add_chart_control(e, chart_config.build_chart_control)
            self.config_card.content.content.controls.append(
                ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                    controls=[generate_chart_button]
                )
            )
        elif viz_type == "Heatmap":
            chart_config = HeatmapChart(self.parent.dataset.df, self, self.page)
            self.config_card = chart_config.build_chart_settings_control()
            generate_chart_button.on_click = lambda e: self._add_chart_control(e, chart_config.build_chart_control)
            self.config_card.content.content.controls.append(
                ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                    controls=[generate_chart_button]
                )
            )
        elif viz_type == "Pie":
            chart_config = PieChart(self.parent.dataset.df, self, self.page)
            self.config_card = chart_config.build_chart_settings_control()
            generate_chart_button.on_click = lambda e: self._add_chart_control(e, chart_config.build_chart_control)
            self.config_card.content.content.controls.append(
                ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                    controls=[generate_chart_button]
                )
            )
        elif viz_type == "Bar":
            chart_config = BarChart(self.parent.dataset.df, self, self.page)
            self.config_card = chart_config.build_chart_settings_control()
            generate_chart_button.on_click = lambda e: self._add_chart_control(e, chart_config.build_chart_control)
            self.config_card.content.content.controls.append(
                ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                    controls=[generate_chart_button]
                )
            )
        
        self.column.controls[2].controls.append(self.config_card)
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
    
    def _font_size_on_blur(self, e: ft.ControlEvent, default: int) -> None:
        if e.control.value is None:
            e.control.value = default
            self.page.update()
            return
        elif isinstance(e.control.value, str):
            try:
                if int(e.control.value) < 10:
                    e.control.value = default
                    self.page.update()
            except ValueError:
                e.control.value = default
                self.page.update()
    
    def build_controls(self) -> ft.Column:
        self.viz_type_dropdown = ft.Dropdown(
            value="Histogram",
            expand=True,
            label="Visualization Type",
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("Histogram", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("Scatter", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("Box", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("Heatmap", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("Pie", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("Bar", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            on_change=self._on_viz_type_change,
        )
        
        self.chart_width = ft.TextField(
            value="10",
            label="Width",
            expand=1,
            max_length=2,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            on_blur=self._chart_size_on_blur
        )
        
        self.chart_height = ft.TextField(
            value="10",
            expand=1,
            max_length=2,
            label="Height",
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            on_blur=self._chart_size_on_blur
        )
        
        self.title_size = ft.TextField(
            value="16",
            label="Title",
            expand=1,
            max_length=2,
            input_filter=ft.NumbersOnlyInputFilter(),
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            on_blur=lambda e: self._font_size_on_blur(e, 16)
        )
        self.axes_size = ft.TextField(
            value="14",
            label="Axes",
            expand=1,
            max_length=2,
            input_filter=ft.NumbersOnlyInputFilter(),
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            on_blur=lambda e: self._font_size_on_blur(e, 14)
        )
        
        self.palette_dropdown = ft.Dropdown(
            value="deep",
            label="Palette",
            expand=1,
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
            expand=1,
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
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption(
                    style, text_style=ft.TextStyle(font_family="SF regular")
                ) for style in ["darkgrid", "whitegrid", "dark", "white", "ticks"]
            ],
        )
        
        self.display_info = ft.Switch(
            value=True,
            expand=1,
            label="Display chart info",
            label_position=ft.LabelPosition.RIGHT,
            label_style=ft.TextStyle(font_family="SF regular"),
        )
        self.show_legend_switch = ft.Switch(
            value=True,
            expand=1,
            label="Display legend",
            label_position=ft.LabelPosition.RIGHT,
            label_style=ft.TextStyle(font_family="SF regular"),
        )
        self.original_size_switch = ft.Switch(
            value=True,
            expand=1,
            label="Original size",
            tooltip="Whether to display chart in orginal size",
            label_position=ft.LabelPosition.RIGHT,
            label_style=ft.TextStyle(font_family="SF regular"),
        )
        
        chart_config = HistogramChart(self.parent.dataset.df, self, self.page)
        self.config_card = chart_config.build_chart_settings_control()
        generate_chart_button = ft.FilledButton(
            text="Generate Visualization",
            icon=ft.Icons.SHOW_CHART,
            on_click=None,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        generate_chart_button.on_click = lambda e: self._add_chart_control(e, chart_config.build_chart_control)
        self.config_card.content.content.controls.append(
            ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[generate_chart_button]
            )
        )
        self.chart_card = ft.Card(visible=False)
        
        self.column = ft.Column(
            scroll=ft.ScrollMode.ALWAYS,
            horizontal_alignment=ft.CrossAxisAlignment.START,
            alignment=ft.MainAxisAlignment.START,
            controls=[
                ft.Row([ft.Text("Data Visualization & Analysis", expand=True, size=30, font_family="SF thin", text_align="center")]),
                ft.Divider(),
                ft.Row(
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        ft.Card(
                            expand=2,
                            content=ft.Container(
                                margin=ft.margin.all(15),
                                content=ft.Column(
                                    spacing=15,
                                    controls=[
                                        ft.Row([ft.Text("Visualization Settings", font_family="SF thin", size=24, expand=True, text_align="center")]),
                                        ft.Divider(),
                                        self.viz_type_dropdown,
                                        ft.Text("Chart global options", font_family="SF thin", size=16),
                                        ft.Row(
                                            controls=[
                                                ft.Text("Customization", font_family="SF thin", size=12, expand=1),
                                                self.palette_dropdown,
                                                self.context_dropdown,
                                                self.style_dropdown
                                            ]
                                        ),
                                        ft.Row(
                                            controls=[
                                                ft.Text("Figure size", font_family="SF thin", size=12, expand=2),
                                                self.chart_width,
                                                self.chart_height,
                                            ]
                                        ),
                                        ft.Row(
                                            controls=[
                                                ft.Text("Font size", font_family="SF thin", size=12, expand=2),
                                                self.title_size,
                                                self.axes_size,
                                            ]
                                        ),
                                        ft.Row(
                                            controls=[
                                                self.display_info,
                                                self.show_legend_switch,
                                                self.original_size_switch
                                            ],
                                            expand=True,
                                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                                        )
                                    ]
                                )
                            )
                        ),
                        self.config_card,
                    ]
                ),
                # self.config_card,
                self.chart_card
            ]
        )
        return self.column
