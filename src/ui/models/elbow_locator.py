"""
Elbow Locator Model using KneeLocator

Uses the kneed library to automatically find the optimal elbow point for KMeans clustering
by analyzing the inertia curve. This helps determine the best number of clusters.

Configurable hyperparameters:
- max_clusters: Maximum number of clusters to evaluate (default: 10)
- curve: Curve type ('concave' or 'convex', default: 'concave')
- direction: Direction ('increasing' or 'decreasing', default: 'decreasing')
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from kneed import KneeLocator

from utils.model_utils import (
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar
)

if TYPE_CHECKING:
    from ..model_factory import ModelFactory


@dataclass
class ElbowLocatorModel:
    """Elbow Locator model for automatic optimal cluster detection."""
    
    parent: ModelFactory
    df: DataFrame = None
    train_btn: Optional[ft.ElevatedButton] = None
    
    def __post_init__(self):
        """Initialize the model with parent reference."""
        self.df = self.df.copy()
    
    def _prepare_data(self) -> Optional[Tuple]:
        """
        Prepare and scale data for elbow detection.
        
        Returns:
            Tuple of (scaled_data, feature_columns) or None if preparation fails
        """
        try:
            X = self.df.copy()
            
            # Handle missing values
            X = X.fillna(X.mean(numeric_only=True))
            
            # Scale data
            scaler = StandardScaler() if self.parent.scaler_dropdown.value == "standard_scaler" else MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            return X_scaled, X
        
        except Exception as e:
            self.parent.page.open(ft.SnackBar(
                ft.Text(f"Data preparation error: {str(e)}", font_family="SF regular")
            ))
            return None
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train elbow locator model and display results with optimal cluster recommendation."""
        try:
            e.control.disabled = True
            disable_navigation_bar(self.parent.page)
            self.parent.page.update()
            
            data = self._prepare_data()
            if data is None:
                enable_navigation_bar(self.parent.page)
                return
            
            X_scaled, feature_cols = data
            
            # Get hyperparameters
            max_clusters = int(self.max_clusters_field.value)
            curve = self.curve_dropdown.value
            direction = self.direction_dropdown.value
            
            # Validate parameters
            if max_clusters < 2 or max_clusters > 20:
                self.parent.page.open(ft.SnackBar(
                    ft.Text("Max clusters must be between 2 and 20", font_family="SF regular")
                ))
                enable_navigation_bar(self.parent.page)
                return
            
            # Compute inertias for range of clusters
            inertias = []
            K_range = range(1, max_clusters + 1)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            # Apply KneeLocator to find elbow
            try:
                knee_locator = KneeLocator(
                    list(K_range),
                    inertias,
                    curve=curve,
                    direction=direction,
                    online=False
                )
                
                optimal_clusters = knee_locator.elbow
                
                if optimal_clusters is None:
                    optimal_clusters = max_clusters // 2
            except Exception as e:
                # Fallback: use heuristic if KneeLocator fails
                optimal_clusters = max(2, max_clusters // 2)
            
            # Format results
            metrics_dict = {
                'optimal_clusters': int(optimal_clusters),
                'max_clusters_tested': max_clusters,
                'curve_type': curve,
                'direction': direction,
                'final_inertia': float(inertias[-1]),
                'optimal_inertia': float(inertias[int(optimal_clusters) - 1])
            }
            
            # Build detailed result text
            result_text = "# Elbow Locator Results\n\n"
            result_text += f"**Recommended Optimal Clusters:** {optimal_clusters}\n\n"
            result_text += "## Analysis Details\n\n"
            result_text += f"- **Curve Type:** {curve}\n"
            result_text += f"- **Direction:** {direction}\n"
            result_text += f"- **Max Clusters Tested:** {max_clusters}\n"
            result_text += f"- **Optimal Inertia:** {inertias[int(optimal_clusters) - 1]:.4f}\n\n"
            
            result_text += "## Inertia Values by Cluster Count\n\n"
            for k, inertia in zip(K_range, inertias):
                marker = " ← **OPTIMAL**" if k == optimal_clusters else ""
                result_text += f"- K={k}: {inertia:.4f}{marker}\n"
            
            result_text += "\n## Recommendation\n\n"
            result_text += f"Based on the elbow method analysis, **{optimal_clusters} clusters** is recommended. "
            result_text += "This represents the point where adding more clusters provides diminishing returns "
            result_text += "in terms of reducing inertia (within-cluster sum of squares).\n"
            
            # Display results dialog
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "Elbow Locator Results",
                result_text,
                "Elbow Locator"
            )
            self.parent.page.open(evaluation_dialog)
            enable_navigation_bar(self.parent.page)
        
        except Exception as e:
            enable_navigation_bar(self.parent.page)
            self.parent.page.open(ft.SnackBar(
                ft.Text(f"Training failed: {str(e)}", font_family="SF regular")
            ))
        
        finally:
            self.train_btn.disabled = False
            self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for Elbow Locator hyperparameter configuration."""
        
        self.max_clusters_field = ft.TextField(
            label="Maximum Clusters",
            value="10",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            tooltip="Maximum number of clusters to test. Range: 2-20",
        )
        
        self.curve_dropdown = ft.Dropdown(
            label="Curve Type",
            value="concave",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("concave", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("convex", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="Curve type for elbow detection. 'concave'=typical elbow shape (default), 'convex'=reversed",
        )
        
        self.direction_dropdown = ft.Dropdown(
            label="Direction",
            value="decreasing",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("decreasing", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("increasing", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="Direction of change. 'decreasing'=inertia decreases with clusters (default), 'increasing'=opposite",
        )
        
        self.train_btn = ft.FilledButton(
            text="Detect Optimal Clusters",
            icon=ft.Icons.ANALYTICS,
            expand=1,
            on_click=self._train_and_evaluate_model,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=5,
                text_style=ft.TextStyle(font_family="SF regular"),
            )
        )
        
        return ft.Card(
            expand=2,
            content=ft.Container(
                expand=True,
                margin=ft.margin.all(15),
                alignment=ft.alignment.center,
                content=ft.Column(
                    controls=[
                        ft.Row(
                            controls=[ft.Text("Elbow Locator Configuration", font_family="SF thin", size=24, text_align="center", expand=True)]
                        ),
                        ft.Divider(height=20),
                        ft.Row(
                            controls=[self.max_clusters_field],
                        ),
                        ft.Row(
                            controls=[self.curve_dropdown, self.direction_dropdown],
                        ),
                        ft.Row(
                            controls=[self.train_btn],
                        ),
                        ft.Text(
                            "This model automatically detects the optimal number of clusters using the elbow method. "
                            "Adjust max clusters and curve parameters to fine-tune detection.",
                            size=12,
                            color=ft.Colors.GREY_700,
                            font_family="SF regular",
                        ),
                    ],
                ),
            ),
        )
