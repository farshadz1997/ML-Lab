from __future__ import annotations
import logging
import flet as ft
from helpers import resource_path
from ui.layout import AppLayout


if __name__ == "__main__":
    ft.app(AppLayout, assets_dir=resource_path("assets"))
