from __future__ import annotations
import logging
from pathlib import Path
import flet as ft
import pandas as pd
import numpy as np
from typing import List, Literal, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from helpers import resource_path
from utils import DataSet

if TYPE_CHECKING:
    from .layout import AppLayout
    from pandas._typing import Dtype


def hex_to_rgb(hex_color: str) -> tuple[int,int,int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb: tuple[int,int,int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(c))) for c in rgb])

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def sample_gradient(stops: list[str], t: float) -> str:
    """
    stops: list of hex colors (e.g. ['#ffffff', '#ff0000', '#000000'])
    t: 0..1 normalized position
    returns: hex color string
    """
    if not stops:
        return "#ffffff"
    if t <= 0:
        return stops[0]
    if t >= 1:
        return stops[-1]
    # find segment
    seg_len = 1 / (len(stops) - 1)
    idx = int(t / seg_len)
    t_seg = (t - idx * seg_len) / seg_len
    c1 = hex_to_rgb(stops[idx])
    c2 = hex_to_rgb(stops[idx+1])
    interp = tuple(lerp(c1[i], c2[i], t_seg) for i in range(3))
    return rgb_to_hex(interp)
 
    
@dataclass
class Home:
    parent: AppLayout
    page: ft.Page
    column: ft.Column | None = None
    
    def _pick_dataset_file_result(self, e: ft.FilePickerResultEvent) -> None:
        try:
            if e.files:
                self.parent.dataset = DataSet(e.files[0].path)
                self.dataset_path_field.value = e.files[0].path
                self.display_tables_options_row.visible = True
                self.close_dataset_btn.visible = True
                self.open_dataset_button.height = 65
                self.page.navigation_bar.destinations[1].disabled = False
                self.page.navigation_bar.destinations[2].disabled = False
                self.page.update()
                self._display_describe_datatable()
        except Exception as e:
            print(e)
    
    def _drop_column(self, column: str):
        if len(self.parent.dataset.df.columns) == 1:
            self.parent.page.open(ft.SnackBar(ft.Text(f"'{column}' can't be removed! at least one column need to be in dataset", font_family="SF regular")))
            return
        is_removed = self.parent.dataset.drop_column(column)
        if is_removed:
            self.parent.page.open(ft.SnackBar(ft.Text(f"'{column}' has removed from dataset", font_family="SF regular")))
            self._display_describe_datatable()
            return
        self.parent.page.open(ft.SnackBar(ft.Text(f"'{column}' not found in dataset", font_family="SF regular")))
          
    def _display_custom_datatable(self, df: pd.DataFrame, title: str = "Describe") -> None:
        df = df.copy()
        df.replace(np.nan, "NaN", inplace=True)
        datatable = ft.DataTable(
            columns=[
                ft.DataColumn(label=ft.Text("")),
                *[ft.DataColumn(
                    label=ft.Text(col, font_family="SF regular"),
                    numeric=pd.api.types.is_numeric_dtype(df[col])
                ) for col in df.columns]
            ],
            rows=[
                ft.DataRow(
                    cells=[
                        ft.DataCell(content=ft.Text(stat_name)),
                        *[
                            ft.DataCell(
                                content=ft.Text(
                                    value=round(row.iloc[i], 2) if isinstance(
                                        row.iloc[i], (int, float)
                                    ) else row.iloc[i],
                                    font_family="SF regular"
                                )
                            )
                            for i in range(len(df.columns))
                        ]
                    ]
                ) for stat_name, row in df.iterrows()
            ]
        )
        self.datatable_card.visible = True
        self.datatable_container.content = ft.Column(
            scroll=ft.ScrollMode.ALWAYS,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Row([ft.Text(title, font_family="SF thin", size=24, expand=True, text_align="center")]),
                ft.Divider(),
                ft.Row(controls=[datatable], scroll=ft.ScrollMode.AUTO)
            ]
        )
        self.page.update()

    def _display_describe_datatable(self):
        df = self.parent.dataset.describe(include="all")
        df.replace(np.nan, "NaN", inplace=True)
        datatable = ft.DataTable(
            columns=[
                ft.DataColumn(label=ft.Text("")),
                *[ft.DataColumn(
                    label=ft.MenuBar(
                        controls=[
                            ft.SubmenuButton(
                                content=ft.Text(col, font_family="SF regular"),
                                controls=[
                                    ft.MenuItemButton(
                                        content=ft.Text("Remove column"),
                                        leading=ft.Icon(ft.Icons.DELETE),
                                        style=ft.ButtonStyle(
                                            bgcolor={ft.ControlState.HOVERED: ft.Colors.RED}
                                        ),
                                        on_click=lambda _, c=col: self._drop_column(c)
                                    ),
                                    ft.MenuItemButton(
                                        content=ft.Text("Rename column"),
                                        leading=ft.Icon(ft.Icons.EDIT),
                                        style=ft.ButtonStyle(
                                            bgcolor={ft.ControlState.HOVERED: ft.Colors.BLUE}
                                        ),
                                        on_click=lambda _, c=col: self._open_rename_dialog(c)
                                    ),
                                ]
                            )
                        ]
                    ),
                    numeric=pd.api.types.is_numeric_dtype(df[col])
                ) for col in df.columns]
            ],
            rows=[
                ft.DataRow(
                    cells=[
                        ft.DataCell(content=ft.Text(stat_name)),
                        *[
                            ft.DataCell(
                                content=ft.Text(
                                    value=round(row.iloc[i], 2) if isinstance(
                                        row.iloc[i], (int, float)
                                    ) else row.iloc[i],
                                    font_family="SF regular"
                                )
                            )
                            for i in range(len(df.columns))
                        ]
                    ]
                ) for stat_name, row in df.iterrows()
            ]
        )
        self.datatable_card.visible = True
        self.datatable_container.content = ft.Column(
            scroll=ft.ScrollMode.ALWAYS,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Row([ft.Text("Pandas describe", font_family="SF thin", size=24, expand=True, text_align="center")]),
                ft.Divider(),
                ft.Row(controls=[datatable], scroll=ft.ScrollMode.AUTO)
            ]
        )
        self.page.update()
    
    def close_dataset(self, e: ft.ControlEvent | None = None) -> None:
        self.parent.dataset = None
        self.dataset_path_field.value = None
        self.display_tables_options_row.visible = False
        self.datatable_card.visible = False
        self.datatable_container.content = None
        self.open_dataset_button.height = 50
        self.close_dataset_btn.visible = False
        self.page.navigation_bar.destinations[1].disabled = True
        self.page.navigation_bar.destinations[2].disabled = True
        self.displaying_dataset = None
        self.page.update()

    def _open_rename_dialog(self, column: str):
        self.rename_dlg.title.value = f"Rename column ({column})?"
        self.rename_dlg.actions[0].on_click = lambda _: self._rename_column(column)
        self.page.update()
        self.page.open(self.rename_dlg)

    def _rename_column(self, column: str):
        new_name = self.rename_dlg.content.value
        if new_name in (None, ""):
            return
        elif any(new_name.startswith(str(i)) for i in range(10)):
            return
        else:
            self.parent.dataset.rename_column(column, new_name.strip())
            self.page.close(self.rename_dlg)
            self.rename_dlg.title.value = "Rename column"
            self.rename_dlg.actions[0].on_click = None
            self.rename_dlg.content.value = None
            self.page.update()
            self._display_describe_datatable()

    def build_controls(self) -> ft.Column:
        if self.column:
            return self.column

        self.rename_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Rename column", font_family="SF regular"),
            content=ft.TextField(
                label="New name",
                label_style=ft.TextStyle(font_family="SF regular"),
                hint_text="Enter new name for the selected column",
                hint_style=ft.TextStyle(font_family="SF regular"),
                input_filter=ft.InputFilter(r"^\D.*$")
            ),
            actions=[
                ft.FilledButton(
                    text="Apply",
                    expand=1,
                    width=100,
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=8),
                        elevation=5,
                        text_style=ft.TextStyle(font_family="SF regular"),
                    ),
                ),
                ft.TextButton(
                    text="Close",
                    expand=1,
                    width=100,
                    on_click=lambda _: self.page.close(self.rename_dlg),
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=8),
                        elevation=5,
                        text_style=ft.TextStyle(font_family="SF regular"),
                    ),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.CENTER,
        )

        pick_files_dialog = ft.FilePicker(on_result=self._pick_dataset_file_result)
        self.parent.page.overlay.append(pick_files_dialog)
        self.close_dataset_btn = ft.IconButton(
            icon=ft.Icons.CLOSE, visible=False, on_click=self.close_dataset
        )
        self.dataset_path_field = ft.TextField(
            label="Dataset file",
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            read_only=True,
            expand=6,
            suffix=self.close_dataset_btn
        )
        self.open_dataset_button = ft.FilledButton(
            text="Open dataset",
            icon=ft.Icons.FILE_OPEN,
            expand=1,
            height=50,
            on_click=lambda _: pick_files_dialog.pick_files(
                allow_multiple=False,
                allowed_extensions=["csv"]
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        self.pandas_describe_btn = ft.FilledButton(
            text="Pandas describe",
            icon=ft.Icons.ANALYTICS,
            expand=1,
            on_click=lambda _: self._display_describe_datatable(),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        self.custom_describe_btn = ft.FilledButton(
            text="Custom describe",
            icon=ft.Icons.ANALYTICS,
            expand=1,
            on_click=lambda _: self._display_custom_datatable(
                self.parent.dataset.custom_describe(),
                "Custom describe",
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        self.missing_percent_btn = ft.FilledButton(
            text="Missing values table",
            icon=ft.Icons.ANALYTICS,
            expand=1,
            on_click=lambda _: self._display_custom_datatable(
                self.parent.dataset.calculate_missing_percent(),
                "Missing values percentage",
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        self.display_tables_options_row = ft.Row(
            visible=False,
            controls=[
                ft.Text("Options:", font_family="SF thin", size=20, expand=1),
                self.pandas_describe_btn,
                self.custom_describe_btn,
                self.missing_percent_btn
            ]
        )
        self.datatable_container = ft.Container(margin=ft.margin.all(15), height=600)
        self.datatable_card = ft.Card(visible=False, content=self.datatable_container)
        self.column = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[
                ft.Text("Dataset Overview", font_family="SF thin", size=30, expand=True, text_align="center"),
                ft.Divider(),
                ft.Card(
                    content=ft.Container(
                        margin=ft.margin.all(15),
                        content=ft.Column(
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                ft.Row(
                                    controls=[
                                        self.dataset_path_field,
                                        self.open_dataset_button
                                    ],
                                ),
                                self.display_tables_options_row
                            ]
                        )
                    )
                ),
                self.datatable_card
            ]
        )
        return self.column