import flet as ft
from typing import List, Dict


class BaseChart:
    def __init__(self):
        self.attrs: Dict[str, int | float | str] = {}
    
    def build_chart_settings_control(self):
        raise NotImplementedError(f"{self.__class__.__name__} doesn not have method 'build_chart_settings_control'")
    
    def build_chart_control(self, controls: List[ft.Control]) -> ft.Card:
        return ft.Card(
            content=ft.Container(
                margin=ft.margin.all(15),
                # alignment=ft.alignment.center,
                content=ft.Column(
                    alignment=ft.MainAxisAlignment.START,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    scroll=ft.ScrollMode.ALWAYS,
                    controls=controls
                )
            )
        )
    
    def set_attr(self, key: str, value: int | float | str) -> None:
        self.attrs[key] = value
        
    def del_attr(self, key: str) -> None:
        if key in self.attrs:
            self.attrs.pop(key)