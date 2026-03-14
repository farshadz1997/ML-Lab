import flet as ft
from dataclasses import dataclass
import warnings, webbrowser, contextlib
from .dataset_explorer import DatasetExplorer
from .data_visualization import DataVisualization
from .model_factory import ModelFactory
from core import DataSet



@dataclass
class AppLayout:
    page: ft.Page
    dataset: DataSet | None = None
    
    def __post_init__(self):
        self.page.title = "ML Lab"
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
        self.page.theme_mode = "dark"
        self.home = DatasetExplorer(self, self.page)
        self.data_science = DataVisualization(self, self.page)
        self.model_factory = ModelFactory(self, self.page)
        self.page.navigation_bar = ft.NavigationBar(
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            on_change=self.on_navigation_change,
            destinations=[
                ft.NavigationBarDestination(icon=ft.Icons.DATA_OBJECT, label="Dataset Explorer"),
                ft.NavigationBarDestination(icon=ft.Icons.ANALYTICS, label="Data Visualization", disabled=True),
                ft.NavigationBarDestination(
                    icon=ft.Icons.MODEL_TRAINING,
                    label="Model Factory",
                    disabled=True
                ),
            ]
        )
        self.toggle_theme_mode_button = ft.IconButton(
            ft.Icons.MODE_NIGHT if self.page.theme_mode == "light" else ft.Icons.WB_SUNNY_ROUNDED,
            on_click=self.toggle_theme_mode,
        )
        self.page.appbar = ft.AppBar(
            title=ft.Text("Dataset Explorer", font_family="SF thin", text_align=ft.TextAlign.CENTER, size=25),
            center_title=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            elevation=5,
            actions=[
                self.toggle_theme_mode_button,
                ft.IconButton(
                    ft.Icons.OPEN_IN_BROWSER,
                    tooltip="GitHub page",
                    on_click=lambda _: webbrowser.open("https://github.com/farshadz1997/ML-Lab")
                )
            ],
        )
        self.page.scroll = ft.ScrollMode.ALWAYS
        self.page.controls = [self.home.build_controls()]
        self.page.update()
        warnings.showwarning = self.warning_handler
    
    def warning_handler(self, message, category, filename, lineno, file=None, line=None):
        with contextlib.suppress(AssertionError, Exception):
            if "Matplotlib" in str(message): # Ignore matplotlib warnings
                return
            self.page.open(ft.SnackBar(
                ft.Text(f"Warning: {message}", font_family="SF regular"),
                bgcolor=ft.Colors.AMBER_ACCENT_400,
                action="Alright!",
                duration=20000
            ))

    def toggle_theme_mode(self, e: ft.ControlEvent) -> None:
        self.page.theme_mode = "dark" if self.page.theme_mode == "light" else "light"
        self.toggle_theme_mode_button.icon = (
            ft.Icons.MODE_NIGHT if self.page.theme_mode == "light" else ft.Icons.WB_SUNNY_ROUNDED
        )
        self.page.update()

    def on_navigation_change(self, e: ft.ControlEvent):
        nav_index = int(e.data)
        if nav_index == 0:
            column = self.home.build_controls()
            self.page.appbar.title.value = "Dataset Explorer"
        elif nav_index == 1:
            column = self.data_science.build_controls()
            self.page.appbar.title.value = "Data Visualization"
        elif nav_index == 2:
            column = self.model_factory.build_controls()
            self.page.appbar.title.value = "Model Factory"
        else:
            column = self.home.build_controls()
            self.page.appbar.title.value = "Dataset Explorer"
        self.page.controls.clear()
        self.page.controls = [column]
        self.page.update()
    
    def on_route_change(self, e: ft.RouteChangeEvent):
        print(e.route)
        
    def view_pop(self, view):
        self.page.views.pop()
        top_view = self.page.views[-1]
        self.page.go(top_view.route)