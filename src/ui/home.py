from __future__ import annotations
import logging
from pathlib import Path
import flet as ft
import pandas as pd
import numpy as np
from typing import List, Literal, Any, TYPE_CHECKING, Callable
from dataclasses import dataclass, field
from helpers import resource_path
from utils import DataSet

if TYPE_CHECKING:
    from .layout import AppLayout
    from pandas._typing import Dtype, float

DISPLAY_MODE = Literal["describe", "custom_describe", "missing_values", "nan_rows", "duplicate_rows", "browse"]

@dataclass
class Home:
    parent: AppLayout
    page: ft.Page
    column: ft.Column | None = None
    current_display_mode: DISPLAY_MODE | None = None
    
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
                self._create_describe_datatable()
        except Exception as e:
            print(e)
    
    def _drop_column(self, control: ft.Control) -> None:
        column = control.parent.content.value
        if len(self.parent.dataset.df.columns) == 1:
            self.parent.page.open(ft.SnackBar(ft.Text(f"'{column}' can't be removed! at least one column need to be in dataset", font_family="SF regular")))
            return
        is_removed = self.parent.dataset.drop_column(column)
        if is_removed:
            column_control: ft.DataColumn = control.parent.parent.parent
            datatable: ft.DataTable = column_control.parent
            column_idx = datatable.columns.index(column_control)
            datatable.columns.remove(column_control)
            for row in datatable.rows:
                row.cells.pop(column_idx)
            self.parent.page.open(ft.SnackBar(ft.Text(f"'{column}' has been removed from dataset", font_family="SF regular")))
            self.page.update()
            return
        self.parent.page.open(ft.SnackBar(ft.Text(f"'{column}' not found in dataset", font_family="SF regular")))
    
    def _drop_row(self, idx: int, control: ft.Control) -> None:
        if idx < 0 or idx >= self.parent.dataset.shape[0]:
            self.parent.page.open(ft.SnackBar(ft.Text(f"Row index '{idx}' is out of bounds", font_family="SF regular")))
            return
        is_dropped = self.parent.dataset.drop_row(idx)
        if is_dropped:
            data_row = control.parent.parent.parent.parent
            datatable: ft.DataTable = data_row.parent
            datatable.rows.remove(data_row)
            self.parent.page.open(ft.SnackBar(ft.Text(f"Row index '{idx}' has been removed from dataset", font_family="SF regular")))
            self.page.update()
            return
        self.parent.page.open(ft.SnackBar(ft.Text(f"Row index '{idx}' not found in dataset", font_family="SF regular")))
        
    def _open_rename_field(self, control: ft.Control) -> None:
        def on_blur_event(e: ft.ControlEvent, previous_control: ft.MenuBar, previous_name: str) -> None:
            new_name: str | None = e.control.value
            if new_name is None:
                new_name = previous_name
            elif isinstance(new_name, str):
                if new_name.strip() == "":
                    new_name = previous_name
            previous_control.controls[0].content.value = new_name.strip()
            e.control.parent.label = previous_control
            if new_name != previous_name:
                self.parent.dataset.rename_column(current_column_name, new_name.strip())
            self.page.update()
        
        current_column_name: str = control.parent.content.value
        menubar_control: ft.MenuBar = control.parent.parent
        column_control: ft.DataColumn = menubar_control.parent
        column_control.label = ft.TextField(
            label="New name",
            width=100,
            shift_enter=False,
            multiline=False,
            autofocus=True,
            label_style=ft.TextStyle(font_family="SF regular"),
            hint_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.InputFilter(r"^(?!\d).*$"),
            on_blur=lambda e: on_blur_event(e, menubar_control, current_column_name),
            on_submit=lambda e: on_blur_event(e, menubar_control, current_column_name)
        )
        self.page.update()
    
    def _drop_duplicate_rows(self, control: ft.Control) -> None:
        """Remove all duplicate rows from dataset"""
        initial_count = len(self.parent.dataset.df)
        self.parent.dataset.drop_duplicates()
        final_count = len(self.parent.dataset.df)
        removed_count = initial_count - final_count
        
        message = f"Removed {removed_count} duplicate row(s). Dataset now has {final_count} rows."
        self.parent.page.open(ft.SnackBar(ft.Text(message, font_family="SF regular")))
        
        # Refresh current display
        if self.current_display_mode:
            self._refresh_display()
    
    def _drop_nan_rows(self, control: ft.Control) -> None:
        """Remove all rows with NaN values"""
        initial_count = len(self.parent.dataset.df)
        self.parent.dataset.drop_nan_rows()
        final_count = len(self.parent.dataset.df)
        removed_count = initial_count - final_count
        
        message = f"Removed {removed_count} row(s) with NaN values. Dataset now has {final_count} rows."
        self.parent.page.open(ft.SnackBar(ft.Text(message, font_family="SF regular")))
        
        # Refresh current display
        if self.current_display_mode:
            self._refresh_display()
    
    def _refresh_display(self) -> None:
        """Refresh the current display based on current_display_mode"""
        if self.current_display_mode == "describe":
            self._create_describe_datatable()
        elif self.current_display_mode == "custom_describe":
            self._create_custom_datatable(
                self.parent.dataset.custom_describe(),
                "Custom describe"
            )
        elif self.current_display_mode == "missing_values":
            self._create_custom_datatable(
                self.parent.dataset.calculate_missing_percent(),
                "Missing values percentage"
            )
        elif self.current_display_mode == "nan_rows":
            self._create_nan_rows_table()
        elif self.current_display_mode == "duplicate_rows":
            self._create_duplicate_rows_table()
        elif self.current_display_mode == "browse":
            self._create_dataset_browser_table(1)
    
    def _create_nan_rows_table(self, page: int = 1) -> None:
        """Display rows containing NaN values"""
        nan_df = self.parent.dataset.get_rows_with_nan()
        
        if len(nan_df) == 0:
            controls = [
                ft.Row([ft.Text("No rows with NaN values", font_family="SF thin", size=24, expand=True, text_align="center")]),
                ft.Divider(),
            ]
            self._display_table(controls)
            return
        
        page_size = 10
        max_page = len(nan_df) // page_size + 1
        if page < 1 or page > max_page:
            return
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        df = nan_df.iloc[start_idx:end_idx].reset_index()
        
        datatable = ft.DataTable(
            column_spacing=100,
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
                                        on_click=lambda e: self._drop_column(e.control)
                                    ),
                                    ft.MenuItemButton(
                                        content=ft.Text("Rename column"),
                                        leading=ft.Icon(ft.Icons.EDIT),
                                        style=ft.ButtonStyle(
                                            bgcolor={ft.ControlState.HOVERED: ft.Colors.BLUE}
                                        ),
                                        on_click=lambda e: self._open_rename_field(e.control)
                                    ),
                                ]
                            )
                        ]
                    ),
                    numeric=pd.api.types.is_numeric_dtype(nan_df[col])
                ) for col in nan_df.columns]
            ],
            rows=[
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            content=ft.TextButton(
                                content=ft.Row(
                                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                                    controls=[
                                        ft.Text(str(row.iloc[0]), font_family="SF regular"),
                                        ft.IconButton(
                                            icon=ft.Icons.DELETE,
                                            tooltip="Delete row",
                                            on_click=lambda e, r=int(row.iloc[0]): self._drop_row(r, e.control),
                                            style=ft.ButtonStyle(
                                                bgcolor={ft.ControlState.HOVERED: ft.Colors.RED}
                                            ),
                                        )
                                    ]
                                ),
                                style=ft.ButtonStyle(
                                    shape=ft.RoundedRectangleBorder(radius=8),
                                    elevation=5,
                                    text_style=ft.TextStyle(font_family="SF regular"),
                                ),
                            )
                        ),
                        *[
                            ft.DataCell(
                                content=ft.Text(
                                    value=str(round(row.iloc[i+1], 2)) if isinstance(
                                        row.iloc[i+1], (int, float)
                                    ) else str(row.iloc[i+1]),
                                    font_family="SF regular",
                                    color=ft.Colors.RED if pd.isna(row.iloc[i+1]) else None
                                )
                            )
                            for i in range(len(nan_df.columns))
                        ]
                    ]
                ) for _, row in df.iterrows()
            ]
        )
        
        controls = [
            ft.Row([ft.Text(f"Rows with NaN - Page {page}/{max_page}", font_family="SF thin", size=24, expand=True, text_align="center")]),
            ft.Divider(),
            ft.Row(controls=[datatable], scroll=ft.ScrollMode.AUTO),
            ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.FilledButton(
                        text="First page",
                        icon=ft.Icons.FIRST_PAGE,
                        disabled=(page == 1),
                        on_click=lambda _: self._create_nan_rows_table(1),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                    ft.FilledButton(
                        text="Previous",
                        icon=ft.Icons.ARROW_BACK,
                        on_click=lambda _: self._create_nan_rows_table(page - 1),
                        disabled=(page == 1),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                    ft.FilledButton(
                        text="Next",
                        icon=ft.Icons.ARROW_FORWARD,
                        on_click=lambda _: self._create_nan_rows_table(page + 1),
                        disabled=(page == max_page),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                    ft.FilledButton(
                        text="Last Page",
                        icon=ft.Icons.LAST_PAGE,
                        disabled=(page == max_page),
                        on_click=lambda _: self._create_nan_rows_table(max_page),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                ]
            )
        ]
        self._display_table(controls)
        self.current_display_mode = "nan_rows"
    
    def _create_duplicate_rows_table(self, page: int = 1) -> None:
        """Display duplicate rows in dataset"""
        dup_df = self.parent.dataset.get_duplicate_rows()
        
        if len(dup_df) == 0:
            controls = [
                ft.Row([ft.Text("No duplicate rows found", font_family="SF thin", size=24, expand=True, text_align="center")]),
                ft.Divider(),
            ]
            self._display_table(controls)
            return
        
        page_size = 10
        max_page = len(dup_df) // page_size + 1
        if page < 1 or page > max_page:
            return
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        df = dup_df.iloc[start_idx:end_idx].reset_index()
        
        datatable = ft.DataTable(
            column_spacing=100,
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
                                        on_click=lambda e: self._drop_column(e.control)
                                    ),
                                    ft.MenuItemButton(
                                        content=ft.Text("Rename column"),
                                        leading=ft.Icon(ft.Icons.EDIT),
                                        style=ft.ButtonStyle(
                                            bgcolor={ft.ControlState.HOVERED: ft.Colors.BLUE}
                                        ),
                                        on_click=lambda e: self._open_rename_field(e.control)
                                    ),
                                ]
                            )
                        ]
                    ),
                    numeric=pd.api.types.is_numeric_dtype(dup_df[col])
                ) for col in dup_df.columns]
            ],
            rows=[
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            content=ft.TextButton(
                                content=ft.Row(
                                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                                    controls=[
                                        ft.Text(str(row.iloc[0]), font_family="SF regular"),
                                        ft.IconButton(
                                            icon=ft.Icons.DELETE,
                                            tooltip="Delete row",
                                            on_click=lambda e, r=int(row.iloc[0]): self._drop_row(r, e.control),
                                            style=ft.ButtonStyle(
                                                bgcolor={ft.ControlState.HOVERED: ft.Colors.RED}
                                            ),
                                        )
                                    ]
                                ),
                                style=ft.ButtonStyle(
                                    shape=ft.RoundedRectangleBorder(radius=8),
                                    elevation=5,
                                    text_style=ft.TextStyle(font_family="SF regular"),
                                ),
                            )
                        ),
                        *[
                            ft.DataCell(
                                content=ft.Text(
                                    value=str(round(row.iloc[i+1], 2)) if isinstance(
                                        row.iloc[i+1], (int, float)
                                    ) else str(row.iloc[i+1]),
                                    font_family="SF regular",
                                    color=ft.Colors.RED if pd.isna(row.iloc[i+1]) else None
                                )
                            )
                            for i in range(len(dup_df.columns))
                        ]
                    ]
                ) for _, row in df.iterrows()
            ]
        )
        
        controls = [
            ft.Row([ft.Text(f"Duplicate rows - Page {page}/{max_page}", font_family="SF thin", size=24, expand=True, text_align="center")]),
            ft.Divider(),
            ft.Row(controls=[datatable], scroll=ft.ScrollMode.AUTO),
            ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.FilledButton(
                        text="First page",
                        icon=ft.Icons.FIRST_PAGE,
                        disabled=(page == 1),
                        on_click=lambda _: self._create_duplicate_rows_table(1),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                    ft.FilledButton(
                        text="Previous",
                        icon=ft.Icons.ARROW_BACK,
                        on_click=lambda _: self._create_duplicate_rows_table(page - 1),
                        disabled=(page == 1),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                    ft.FilledButton(
                        text="Next",
                        icon=ft.Icons.ARROW_FORWARD,
                        on_click=lambda _: self._create_duplicate_rows_table(page + 1),
                        disabled=(page == max_page),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                    ft.FilledButton(
                        text="Last Page",
                        icon=ft.Icons.LAST_PAGE,
                        disabled=(page == max_page),
                        on_click=lambda _: self._create_duplicate_rows_table(max_page),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                ]
            )
        ]
        self._display_table(controls)
        self.current_display_mode = "duplicate_rows"

    def _display_table(self, controls: list[ft.Control]) -> None:
        self.datatable_card.visible = True
        self.datatable_container.content = ft.Column(
            scroll=ft.ScrollMode.ALWAYS,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=controls
        )
        self.page.update()
          
    def _create_custom_datatable(self, df: pd.DataFrame, title: str = "Describe") -> None:
        df = df.copy()
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
        
        refresh_btn = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Refresh display",
            on_click=lambda _: self._refresh_display(),
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        
        controls = [
            ft.Row([
                ft.Text(title, font_family="SF thin", size=24, expand=True, text_align="center"),
                refresh_btn
            ]),
            ft.Divider(),
            ft.Row(controls=[datatable], scroll=ft.ScrollMode.AUTO)
        ]
        
        if title == "Custom describe":
            self.current_display_mode = "custom_describe"
        elif title == "Missing values percentage":
            self.current_display_mode = "missing_values"
        
        self._display_table(controls)

    def _create_describe_datatable(self) -> None:
        df = self.parent.dataset.describe(include="all")
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
                                        on_click=lambda e: self._drop_column(e.control)
                                    ),
                                    ft.MenuItemButton(
                                        content=ft.Text("Rename column"),
                                        leading=ft.Icon(ft.Icons.EDIT),
                                        style=ft.ButtonStyle(
                                            bgcolor={ft.ControlState.HOVERED: ft.Colors.BLUE}
                                        ),
                                        on_click=lambda e: self._open_rename_field(e.control)
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
                                    font_family="SF regular",
                                    color=ft.Colors.RED if pd.isna(row.iloc[i]) else None
                                )
                            )
                            for i in range(len(df.columns))
                        ]
                    ]
                ) for stat_name, row in df.iterrows()
            ]
        )
        
        refresh_btn = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Refresh display",
            on_click=lambda _: self._refresh_display(),
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        
        controls = [
            ft.Row([
                ft.Text("Pandas describe", font_family="SF thin", size=24, expand=True, text_align="center"),
                refresh_btn
            ]),
            ft.Divider(),
            ft.Row(controls=[datatable], scroll=ft.ScrollMode.AUTO)
        ]
        self.current_display_mode = "describe"
        self._display_table(controls)
    
    def _export_csv(self, e: ft.ControlEvent | None = None) -> None:
        try:
            export_path = self.parent.dataset.export_csv()
            self.parent.page.open(ft.SnackBar(ft.Text(f"Dataset exported successfully to {export_path}", font_family="SF regular")))
        except Exception as e:
            self.parent.page.open(ft.SnackBar(ft.Text(f"Failed to export dataset: {e}", font_family="SF regular")))
    
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

    def _create_dataset_browser_table(self, page: int = 1) -> None:
        page_size = 10
        max_page = self.parent.dataset.shape[0] // page_size + 1
        if page < 1 or page > max_page:
            return
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        df = self.parent.dataset.df.iloc[start_idx:end_idx]
        datatable = ft.DataTable(
            column_spacing=100,
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
                                        on_click=lambda e: self._drop_column(e.control)
                                    ),
                                    ft.MenuItemButton(
                                        content=ft.Text("Rename column"),
                                        leading=ft.Icon(ft.Icons.EDIT),
                                        style=ft.ButtonStyle(
                                            bgcolor={ft.ControlState.HOVERED: ft.Colors.BLUE}
                                        ),
                                        on_click=lambda e: self._open_rename_field(e.control)
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
                        ft.DataCell(
                            content=ft.TextButton(
                                content=ft.Row(
                                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                                    controls=[
                                        ft.Text(stat_name, font_family="SF regular"),
                                        ft.IconButton(
                                            icon=ft.Icons.DELETE,
                                            tooltip="Delete row",
                                            on_click=lambda e, r=int(stat_name): self._drop_row(r, e.control),
                                            style=ft.ButtonStyle(
                                                bgcolor={ft.ControlState.HOVERED: ft.Colors.RED}
                                            ),
                                        )
                                    ]
                                ),
                                style=ft.ButtonStyle(
                                    shape=ft.RoundedRectangleBorder(radius=8),
                                    elevation=5,
                                    text_style=ft.TextStyle(font_family="SF regular"),
                                ),
                            )
                        ),
                        *[
                            ft.DataCell(
                                content=ft.Text(
                                    value=round(row.iloc[i], 2) if isinstance(
                                        row.iloc[i], (int, float)
                                    ) else row.iloc[i],
                                    font_family="SF regular",
                                    color=ft.Colors.RED if pd.isna(row.iloc[i]) else None
                                )
                            )
                            for i in range(len(df.columns))
                        ]
                    ]
                ) for stat_name, row in df.iterrows()
            ]
        )
        
        refresh_btn = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Refresh display",
            on_click=lambda _: self._refresh_display(),
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        
        controls = [
            ft.Row([
                ft.Text(f"Dataset browser - Page {page}/{max_page}", font_family="SF thin", size=24, expand=True, text_align="center"),
                refresh_btn
            ]),
            ft.Divider(),
            ft.Row(controls=[datatable], scroll=ft.ScrollMode.AUTO),
            ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.FilledButton(
                        text="First page",
                        icon=ft.Icons.FIRST_PAGE,
                        disabled=(page == 1),
                        on_click=lambda _: self._create_dataset_browser_table(1),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                    ft.FilledButton(
                        text="Previous",
                        icon=ft.Icons.ARROW_BACK,
                        on_click=lambda _: self._create_dataset_browser_table(page - 1),
                        disabled=(page == 1),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                    ft.FilledButton(
                        text="Next",
                        icon=ft.Icons.ARROW_FORWARD,
                        on_click=lambda _: self._create_dataset_browser_table(page + 1),
                        disabled=(page == (len(self.parent.dataset.df) // page_size) + 1),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                    ft.FilledButton(
                        text="Last Page",
                        icon=ft.Icons.LAST_PAGE,
                        disabled=(page == (len(self.parent.dataset.df) // page_size) + 1),
                        on_click=lambda _: self._create_dataset_browser_table((len(self.parent.dataset.df) // page_size) + 1),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                ]
            )
        ]
        self._display_table(controls)
        self.current_display_mode = "browse"
        
    def _reset_index(self, e: ft.ControlEvent | None = None) -> None:
        """Reset the DataFrame index"""
        self.parent.dataset.reset_index()
        self.parent.page.open(ft.SnackBar(ft.Text("Dataset index has been reset.", font_family="SF regular")))
        
        # Refresh current display
        if self.current_display_mode:
            self._refresh_display()
    
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
        
        # Display Options Menu
        display_options_menu = ft.MenuBar(
            controls=[
                ft.SubmenuButton(
                    content=ft.Text(
                        value="Display Options",
                        font_family="SF regular"
                    ),
                    leading=ft.Icon(ft.Icons.MENU),
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=8),
                        elevation=5,
                        text_style=ft.TextStyle(font_family="SF regular"),
                    ),
                    controls=[
                        ft.MenuItemButton(
                            content=ft.Text("Browse dataset", font_family="SF regular"),
                            leading=ft.Icon(ft.Icons.SEARCH),
                            on_click=lambda _: self._create_dataset_browser_table(1),
                        ),
                        ft.MenuItemButton(
                            content=ft.Text("Pandas describe", font_family="SF regular"),
                            leading=ft.Icon(ft.Icons.ANALYTICS),
                            on_click=lambda _: self._create_describe_datatable(),
                        ),
                        ft.MenuItemButton(
                            content=ft.Text("Custom describe", font_family="SF regular"),
                            leading=ft.Icon(ft.Icons.INFO),
                            on_click=lambda _: self._create_custom_datatable(
                                self.parent.dataset.custom_describe(),
                                "Custom describe",
                            ),
                        ),
                        ft.MenuItemButton(
                            content=ft.Text("Missing values", font_family="SF regular"),
                            leading=ft.Icon(ft.Icons.WARNING),
                            on_click=lambda _: self._create_custom_datatable(
                                self.parent.dataset.calculate_missing_percent(),
                                "Missing values percentage",
                            ),
                        ),
                    ]
                )
            ]
        )
        
        # Data Quality Menu
        data_quality_menu = ft.MenuBar(
            controls=[
                ft.SubmenuButton(
                    content=ft.Text(
                        value="Data Quality",
                        font_family="SF regular",
                    ),
                    leading=ft.Icon(ft.Icons.MENU),
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=8),
                        elevation=5,
                        text_style=ft.TextStyle(font_family="SF regular"),
                    ),
                    controls=[
                        ft.MenuItemButton(
                            content=ft.Text("Show NaN rows", font_family="SF regular"),
                            leading=ft.Icon(ft.Icons.VISIBILITY),
                            on_click=lambda _: self._create_nan_rows_table(),
                        ),
                        ft.MenuItemButton(
                            content=ft.Text("Show duplicate rows", font_family="SF regular"),
                            leading=ft.Icon(ft.Icons.COPY),
                            on_click=lambda _: self._create_duplicate_rows_table(),
                        ),
                        ft.Divider(),
                        ft.MenuItemButton(
                            content=ft.Text("Remove NaN rows", font_family="SF regular"),
                            leading=ft.Icon(ft.Icons.DELETE),
                            style=ft.ButtonStyle(
                                bgcolor={ft.ControlState.HOVERED: ft.Colors.RED}
                            ),
                            on_click=lambda e: self._drop_nan_rows(e.control),
                        ),
                        ft.MenuItemButton(
                            content=ft.Text("Remove duplicates", font_family="SF regular"),
                            leading=ft.Icon(ft.Icons.DELETE),
                            style=ft.ButtonStyle(
                                bgcolor={ft.ControlState.HOVERED: ft.Colors.RED}
                            ),
                            on_click=lambda e: self._drop_duplicate_rows(e.control),
                        ),
                        ft.MenuItemButton(
                            content=ft.Text("Reset index", font_family="SF regular"),
                            leading=ft.Icon(ft.Icons.RESTART_ALT),
                            on_click=lambda e: self._reset_index(e.control),
                        ),
                    ]
                )
            ]
        )
        self.export_csv_button = ft.FilledButton(
            text="Export CSV",
            icon=ft.Icons.FILE_DOWNLOAD,
            height=40,
            on_click=self._export_csv,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        self.display_tables_options_row = ft.Row(
            visible=False,
            controls=[
                ft.Text("Tools:", font_family="SF thin", size=16, expand=1),
                self.export_csv_button,
                display_options_menu,
                data_quality_menu
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