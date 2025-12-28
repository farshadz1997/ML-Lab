from __future__ import annotations
import logging
from pathlib import Path
import flet as ft
import pandas as pd
import numpy as np
from typing import List, Literal, Any, TYPE_CHECKING, Callable
from dataclasses import dataclass, field
from helpers import resource_path
from core import DataSet

if TYPE_CHECKING:
    from .layout import AppLayout
    from pandas._typing import Dtype, float

DISPLAY_MODE = Literal["describe", "custom_describe", "missing_values", "nan_rows", "duplicate_rows", "browse"]

@dataclass
class DatasetExplorer:
    parent: AppLayout
    page: ft.Page
    column: ft.Column | None = None
    current_display_mode: DISPLAY_MODE | None = None
    current_display_page: int = 1
    
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
            dense=True,
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
        # Use the remembered page where applicable. Table creators will
        # clamp the page to a valid value if necessary.
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
            self._create_nan_rows_table(self.current_display_page)
        elif self.current_display_mode == "duplicate_rows":
            self._create_duplicate_rows_table(self.current_display_page)
        elif self.current_display_mode == "browse":
            self._create_dataset_browser_table(self.current_display_page)

    def _make_datatable_columns(self, df: pd.DataFrame) -> list[ft.DataColumn]:
        """Return DataTable column definitions for a given dataframe.

        Columns include a leading empty column for row actions and a
        MenuBar label for each dataframe column (remove/rename actions).
        """
        cols: list[ft.DataColumn] = [ft.DataColumn(label=ft.Text(""))]
        for col in df.columns:
            cols.append(
                ft.DataColumn(
                    label=ft.MenuBar(
                        controls=[
                            ft.SubmenuButton(
                                content=ft.Text(col, font_family="SF regular"),
                                controls=[
                                    ft.MenuItemButton(
                                        content=ft.Text("Remove column"),
                                        leading=ft.Icon(ft.Icons.DELETE),
                                        style=ft.ButtonStyle(bgcolor={ft.ControlState.HOVERED: ft.Colors.RED}),
                                        on_click=lambda e: self._drop_column(e.control)
                                    ),
                                    ft.MenuItemButton(
                                        content=ft.Text("Rename column"),
                                        leading=ft.Icon(ft.Icons.EDIT),
                                        style=ft.ButtonStyle(bgcolor={ft.ControlState.HOVERED: ft.Colors.BLUE}),
                                        on_click=lambda e: self._open_rename_field(e.control)
                                    ),
                                    ft.MenuItemButton(
                                        content=ft.Text("Unique values"),
                                        leading=ft.Icon(ft.Icons.DIFFERENCE),
                                        on_click=lambda e, col=col: self._display_unique_values(col)
                                    ),
                                ]
                            )
                        ]
                    ),
                    numeric=pd.api.types.is_numeric_dtype(df[col])
                )
            )
        return cols

    def _make_datatable_rows(self, df_slice: pd.DataFrame) -> list[ft.DataRow]:
        """Create DataRow list for the provided dataframe slice.

        The dataframe slice is expected to have been produced by
        df.reset_index() so the original row index is at position 0.
        """
        rows: list[ft.DataRow] = []
        for _, row in df_slice.iterrows():
            # first column contains the original index and delete button
            idx_value = row.iloc[0]
            first_cell = ft.DataCell(
                content=ft.TextButton(
                    content=ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_AROUND,
                        controls=[
                            ft.Text(str(int(idx_value)), font_family="SF regular"),
                            ft.IconButton(
                                icon=ft.Icons.DELETE,
                                tooltip="Delete row",
                                on_click=lambda e, r=int(idx_value): self._drop_row(r, e.control),
                                style=ft.ButtonStyle(bgcolor={ft.ControlState.HOVERED: ft.Colors.RED}),
                            )
                        ]
                    ),
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=8),
                        elevation=5,
                        text_style=ft.TextStyle(font_family="SF regular"),
                    ),
                )
            )

            other_cells = []
            for i in range(len(df_slice.columns) - 1):
                val = row.iloc[i + 1]
                display_val = (
                    str(round(val, 2)) if isinstance(val, (int, float)) and not pd.isna(val) else str(val)
                )
                other_cells.append(
                    ft.DataCell(
                        content=ft.Text(
                            value=display_val,
                            font_family="SF regular",
                            color=ft.Colors.RED if pd.isna(val) else None
                        )
                    )
                )

            rows.append(ft.DataRow(cells=[first_cell, *other_cells]))
        return rows

    def _jump_to_page(self, page_field: ft.TextField, df: pd.DataFrame, title: str) -> None:
        """Jump to a specific page based on user input."""
        try:
            page = int(page_field.value) if page_field.value else 1
            page_size = 10
            max_page = (len(df) - 1) // page_size + 1 if len(df) > 0 else 1
            
            if page < 1 or page > max_page:
                self.parent.page.open(
                    ft.SnackBar(ft.Text(f"Page must be between 1 and {max_page}", font_family="SF regular"))
                )
                return
            
            self._create_paginated_table(df, title, page)
        except ValueError:
            self.parent.page.open(
                ft.SnackBar(ft.Text("Invalid page number. Please enter a valid integer.", font_family="SF regular"))
            )

    def _filter_by_row_index(self, index_field: ft.TextField, df: pd.DataFrame, title: str) -> None:
        """Filter rows by single index or range (e.g. '5' or '5-10')."""
        try:
            input_val = index_field.value.strip() if index_field.value else ""
            if not input_val:
                self.parent.page.open(
                    ft.SnackBar(ft.Text("Please enter a row index or range (e.g., '5' or '5-10')", font_family="SF regular"))
                )
                return
            
            max_idx = len(df) - 1
            
            # Check if it's a range (contains dash)
            if "-" in input_val:
                parts = input_val.split("-")
                if len(parts) != 2:
                    raise ValueError("Range format must be 'start-end'")
                
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                
                if start < 0 or end < 0:
                    raise ValueError("Indices must be non-negative")
                if start > end:
                    raise ValueError("Start index must be <= end index")
                if start > max_idx:
                    raise ValueError(f"Start index {start} exceeds max index {max_idx}")
                
                # Clamp end to max available index
                end = min(end, max_idx)
                filtered_df = df.iloc[start:end+1]
            else:
                # Single index
                idx = int(input_val.strip())
                if idx < 0 or idx > max_idx:
                    raise ValueError(f"Index must be between 0 and {max_idx}")
                
                filtered_df = df.iloc[[idx]]
            
            # Display the filtered data as a one-off table (page 1)
            self._create_paginated_table(filtered_df, f"{title} (Filtered by index)", 1)
            index_field.value = ""
            self.page.update()
        
        except ValueError as e:
            self.parent.page.open(
                ft.SnackBar(ft.Text(f"Invalid input: {str(e)}", font_family="SF regular"))
            )

    def _create_paginated_table(self, df: pd.DataFrame, title: str, page: int = 1) -> None:
        """Generic paginated table builder used by several view functions.

        - df: source DataFrame (not reset)
        - title: header shown above table
        - page: 1-based page number to display
        """
        if df is None:
            return
        total = len(df)
        page_size = 10
        max_page = (total - 1) // page_size + 1 if total > 0 else 1

        # Clamp the requested page to valid range. The refresh behavior
        # will call this with the previously remembered page and we only
        # move to start if the remembered page doesn't exist anymore.
        if page < 1:
            page = 1
        if page > max_page:
            page = max_page

        # Remember the current page so refresh can reuse it.
        self.current_display_page = page

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        df_slice = df.iloc[start_idx:end_idx].reset_index()

        datatable = ft.DataTable(
            column_spacing=100,
            columns=self._make_datatable_columns(df),
            rows=self._make_datatable_rows(df_slice)
        )

        refresh_btn = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Refresh display",
            on_click=lambda _: self._refresh_display(),
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )

        # Input fields for navigation
        page_field = ft.TextField(
            label="Go to page",
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            hint_text="e.g., 2",
            hint_style=ft.TextStyle(font_family="SF regular"),
            width=120,
            input_filter=ft.NumbersOnlyInputFilter(),
            dense=True,
            on_submit=lambda _: self._jump_to_page(page_field, df, title),
        )

        index_field = ft.TextField(
            label="Filter by index",
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            hint_text="e.g., 5 or 5-10",
            hint_style=ft.TextStyle(font_family="SF regular"),
            width=150,
            dense=True,
            on_submit=lambda _: self._filter_by_row_index(index_field, df, title),
        )

        controls = [
            ft.Row([ft.Text(f"{title} - Page {page}/{max_page}", font_family="SF thin", size=24, expand=True, text_align="center"), refresh_btn]),
            ft.Divider(),
            ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=10,
                controls=[
                    page_field,
                    ft.FilledButton(
                        text="Jump",
                        icon=ft.Icons.ARROW_FORWARD,
                        on_click=lambda _: self._jump_to_page(page_field, df, title),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                    ft.VerticalDivider(),
                    index_field,
                    ft.FilledButton(
                        text="Filter",
                        icon=ft.Icons.FILTER_LIST,
                        on_click=lambda _: self._filter_by_row_index(index_field, df, title),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                ]
            ),
            ft.Row(controls=[datatable], scroll=ft.ScrollMode.AUTO),
            ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.FilledButton(
                        text="First page",
                        icon=ft.Icons.FIRST_PAGE,
                        disabled=(page == 1),
                        on_click=lambda _: self._create_paginated_table(df, title, 1),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                    ft.FilledButton(
                        text="Previous",
                        icon=ft.Icons.ARROW_BACK,
                        on_click=lambda _: self._create_paginated_table(df, title, page - 1),
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
                        on_click=lambda _: self._create_paginated_table(df, title, page + 1),
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
                        on_click=lambda _: self._create_paginated_table(df, title, max_page),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                            elevation=5,
                            text_style=ft.TextStyle(font_family="SF regular"),
                        )
                    ),
                ]
            )
        ]

        # set display mode based on title heuristics
        if "NaN" in title or "NaN".lower() in title.lower() or "Missing" in title:
            self.current_display_mode = "nan_rows"
        elif "Duplicate" in title or "Duplicate".lower() in title.lower():
            self.current_display_mode = "duplicate_rows"
        elif "Dataset browser" in title or "Dataset" in title:
            self.current_display_mode = "browse"

        self._display_table(controls)
    
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

        # Use the generic paginated table builder
        self._create_paginated_table(nan_df, "Rows with NaN", page)
    
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

        # Use the generic paginated table builder
        self._create_paginated_table(dup_df, "Duplicate rows", page)

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
                                    ft.MenuItemButton(
                                        content=ft.Text("Unique values"),
                                        leading=ft.Icon(ft.Icons.DIFFERENCE),
                                        on_click=lambda e, col=col: self._display_unique_values(col)
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
        self.current_display_mode = None
        self.current_display_page = 1
        self.page.update()

    def _create_dataset_browser_table(self, page: int = 1) -> None:
        df = self.parent.dataset.df

        if len(df) == 0:
            controls = [
                ft.Row([ft.Text("No data available in dataset", font_family="SF thin", size=24, expand=True, text_align="center")]),
                ft.Divider(),
            ]
            self._display_table(controls)
            return

        self._create_paginated_table(df, "Dataset browser", page)
        
    def _reset_index(self, e: ft.ControlEvent | None = None) -> None:
        """Reset the DataFrame index"""
        self.parent.dataset.reset_index()
        self.parent.page.open(ft.SnackBar(ft.Text("Dataset index has been reset.", font_family="SF regular")))
        
        if self.current_display_mode:
            self._refresh_display()
            
    def _display_unique_values(self, column: str, e: ft.ControlEvent | None = None) -> None:
        """Display unique values for a specific column"""
        try:
            series = self.parent.dataset.df[column]
        except Exception:
            self.parent.page.open(ft.SnackBar(ft.Text(f"Column '{column}' not found", font_family="SF regular")))
            return

        vc = series.value_counts(dropna=False)
        unique_df = vc.reset_index()
        unique_df.columns = [column, "count"]

        def _render(page: int = 1) -> None:
            total = len(unique_df)
            page_size = 10
            max_page = (total - 1) // page_size + 1 if total > 0 else 1
            if page < 1:
                page = 1
            if page > max_page:
                page = max_page

            start = (page - 1) * page_size
            end = start + page_size
            df_slice = unique_df.iloc[start:end]

            datatable = ft.DataTable(
                columns=[
                    ft.DataColumn(label=ft.Text(column, font_family="SF regular")),
                    ft.DataColumn(label=ft.Text("Count", font_family="SF regular"), numeric=True),
                ],
                rows=[
                    ft.DataRow(cells=[
                        ft.DataCell(content=ft.Text("NaN" if pd.isna(row.iloc[0]) else str(row.iloc[0]), font_family="SF regular")),
                        ft.DataCell(content=ft.Text(str(int(row.iloc[1])), font_family="SF regular"))
                    ]) for _, row in df_slice.iterrows()
                ]
            )

            controls = [
                ft.Row([ft.Text(f"Unique values for '{column}' - Page {page}/{max_page}", font_family="SF thin", size=20, expand=True, text_align="center")]),
                ft.Divider(),
                ft.Row(controls=[datatable], scroll=ft.ScrollMode.AUTO),
                ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                    controls=[
                        ft.FilledButton(text="First page", icon=ft.Icons.FIRST_PAGE, disabled=(page == 1), on_click=lambda _: _render(1)),
                        ft.FilledButton(text="Previous", icon=ft.Icons.ARROW_BACK, disabled=(page == 1), on_click=lambda _: _render(page - 1)),
                        ft.FilledButton(text="Next", icon=ft.Icons.ARROW_FORWARD, disabled=(page == max_page), on_click=lambda _: _render(page + 1)),
                        ft.FilledButton(text="Last Page", icon=ft.Icons.LAST_PAGE, disabled=(page == max_page), on_click=lambda _: _render(max_page)),
                    ]
                )
            ]

            self._display_table(controls)

        _render(1)
    
    def _export_ydata_profiling(self) -> None:
        try:
            dialog = ft.AlertDialog(
                modal=True,
                title=ft.Text("Exporting Ydata-profiling Report", font_family="SF regular"),
                content=ft.Row([ft.ProgressRing(visible=True)], expand=True, alignment=ft.MainAxisAlignment.CENTER),
                actions_alignment=ft.MainAxisAlignment.CENTER,
            )
            ok_btn = ft.FilledTonalButton(
                text="OK",
                on_click=lambda e: self.page.close(dialog),
                disabled=True,
                style=ft.ButtonStyle(
                    shape=ft.RoundedRectangleBorder(radius=8),
                    elevation=5,
                    text_style=ft.TextStyle(font_family="SF regular"),
                )
            )
            dialog.actions = [ok_btn]
            self.page.open(dialog)
            report_path = self.parent.dataset.export_ydata_profiling_report()
            dialog.content = ft.Text(f"Ydata-profiling report exported to {report_path}", font_family="SF regular")
        except Exception as e:
            dialog.content = ft.Text(f"Failed to export report: {e}", font_family="SF regular")
        finally:
            ok_btn.disabled = False
            self.page.update()
        
    def build_controls(self) -> ft.Column:
        if self.column:
            return self.column

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
        auto_eda_menu = ft.MenuBar(
            controls=[
                ft.SubmenuButton(
                    content=ft.Text(
                        value="Auto EDA",
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
                            content=ft.Text("Ydata-profiling", font_family="SF regular"),
                            leading=ft.Icon(ft.Icons.ANALYTICS),
                            on_click=lambda _: self._export_ydata_profiling(),
                        )
                    ],
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
                auto_eda_menu,
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