from __future__ import annotations
import logging
from pathlib import Path
import flet as ft
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Literal, Any, TYPE_CHECKING, Callable
from dataclasses import dataclass, field
from helpers import resource_path
import tempfile
import webbrowser
from .charts.scatter import ScatterChart


if TYPE_CHECKING:
    from .layout import AppLayout

VIZ_TYPE = Literal["Histogram", "Box", "Scatter", "Heatmap", "Pie", "Bsar"]

@dataclass
class DataScience:
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
            self.page.update()
            chart_card = chart_function()
            if chart_card is None:
                e.control.disabled = False
                return
            self.chart_card = chart_card
            e.control.disabled = False
            self.column.controls.append(self.chart_card)
            self.page.update()
        except Exception as e:
            e.control.disabled = False
            self.page.update()
            self.page.open(ft.SnackBar(ft.Text(e, font_family="SF regular")))
    
    def _update_viz_options(self, viz_type: VIZ_TYPE) -> None:
        """Update available options based on visualization type"""
        if self.column and self.parent.dataset:
            if self.config_card in self.column.controls:
                self.column.controls.remove(self.config_card)
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
            self.config_card = ft.Card(visible=False)
        elif viz_type == "Box":
            self.config_card = ft.Card(visible=False)
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
            self.config_card = ft.Card(visible=False)
        elif viz_type == "Pie":
            self.config_card = ft.Card(visible=False)
        elif viz_type == "Bar":
            self.config_card = ft.Card(visible=False)
        
        self.column.controls.append(self.config_card)
        self.page.update()
    
    def build_controls(self) -> ft.Column:
        if self.column and self.parent.dataset:
            return self.column
        
        self.viz_type_dropdown = ft.Dropdown(
            value="Histogram",
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
        
        self.config_card = ft.Card(visible=False)
        self.chart_card = ft.Card(visible=False)
        
        self.column = ft.Column(
            scroll=ft.ScrollMode.ALWAYS,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.START,
            controls=[
                ft.Text("Data Visualization & Analysis", expand=False, size=30, font_family="SF thin", text_align="center"),
                ft.Divider(),
                ft.Card(
                    content=ft.Container(
                        margin=ft.margin.all(15),
                        content=ft.Column(
                            spacing=15,
                            controls=[
                                ft.Text("Visualization Settings", font_family="SF regular", weight="bold", size=16),
                                self.viz_type_dropdown,
                            ]
                        )
                    )
                ),
                self.config_card,
                self.chart_card
            ]
        )
        return self.column
