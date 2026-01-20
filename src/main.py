from __future__ import annotations
import logging, os
import flet as ft
from helpers import resource_path
from ui.layout import AppLayout


if __name__ == "__main__":
    assets_dir = resource_path("src/assets") if os.path.exists(resource_path("src")) else resource_path("assets")
    ft.app(AppLayout, assets_dir=assets_dir)
