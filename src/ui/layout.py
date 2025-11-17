from __future__ import annotations
import logging
from pathlib import Path
import flet as ft
import pandas as pd
import numpy as np
from typing import List, Literal, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from helpers import resource_path
from .home import Home
from .data_science import DataScience
from .model_factory import ModelFactory
from utils import DataSet



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
        self.page.window.min_height = 1000
        self.page.window.min_width = 1000
        self.page.window.width = 1000
        self.page.window.height = 1000
        self.home = Home(self, self.page)
        self.data_science = DataScience(self, self.page)
        self.model_factory = ModelFactory(self, self.page)
        self.page.navigation_bar = ft.NavigationBar(
            on_change=self.on_navigation_change,
            destinations=[
                ft.NavigationBarDestination(icon=ft.Icons.DATA_OBJECT, label="Dataset overview"),
                ft.NavigationBarDestination(icon=ft.Icons.ANALYTICS, label="Data Science", disabled=True),
                ft.NavigationBarDestination(
                    icon=ft.Icons.MODEL_TRAINING,
                    label="Model Factory",
                    disabled=True
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