from __future__ import annotations
import logging
from pathlib import Path
import flet as ft
import pandas as pd
import numpy as np
from typing import List, Literal, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from helpers import resource_path

if TYPE_CHECKING:
    from .layout import AppLayout
    from pandas._typing import Dtype


@dataclass
class ModelFactory:
    parent: AppLayout
    page: ft.Page
    column: ft.Column | None = None
    
    def build_controls(self) -> ft.Column:
        if self.column:
            return self.column
        self.column = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[
                ft.Text("Modeling", expand=True, text_align="center"),
                ft.Divider()
            ]
        )
        return self.column