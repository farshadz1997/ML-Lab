"""
K-Means Clustering Model

Partitioning-based clustering that groups data into k clusters
by minimizing within-cluster variance.

Configurable hyperparameters:
- n_clusters: Number of clusters to create
- init: Initialization strategy ('k-means++' or 'random')
- n_init: Number of times to run with different centroids
- random_state: Random seed for reproducibility
"""

from __future__ import annotations
from typing import Optional, Tuple
import flet as ft
from dataclasses import dataclass
from sklearn.cluster import KMeans

from utils.model_utils import (
    calculate_clustering_metrics,
    format_results_markdown,
    create_results_dialog,
    validate_hyperparameters,
    disable_navigation_bar,
    enable_navigation_bar,
)
from .base_model import BaseModel


@dataclass
class KMeansModel(BaseModel):
    """K-Means clustering model."""
    
    def _prepare_data(self):
        """Prepare data for clustering."""
        return self._prepare_data_clustering()
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train K-Means model and display evaluation results."""
        try:
            e.control.disabled = True
            self.parent.disable_model_selection()
            disable_navigation_bar(self.parent.page)
            self.parent.page.update()
            
            data = self._prepare_data()
            if data is None:
                enable_navigation_bar(self.parent.page)
                return
            
            X_scaled, feature_cols = data
            
            # Validate hyperparameters
            hyperparams = {
                'n_clusters': int(self.n_clusters_field.value),
                'n_init': int(self.n_init_field.value) if self.n_init_field.value.strip() != "auto" else "auto",
                'max_iter': int(self.max_iter_field.value),
            }
            
            validation_rules = {
                'n_clusters': {'type': int, 'min': 2},
                'n_init': {'type': int} if self.n_init_field.value.strip() != "auto" else {"type": str, "allowed": ["auto"]},
                'max_iter': {'type': int, 'min': 1},
            }
            
            is_valid, error_msg = validate_hyperparameters(hyperparams, 'kmeans', validation_rules)
            if not is_valid:
                self._show_snackbar(f"Hyperparameter error: {error_msg}", bgcolor=ft.Colors.RED_500)
                return
            
            # Train K-Means
            model = KMeans(
                n_clusters=int(self.n_clusters_field.value),
                init=self.init_dropdown.value,
                n_init=int(self.n_init_field.value) if self.n_init_field.value.strip() != "auto" else "auto",
                max_iter=int(self.max_iter_field.value),
                algorithm=self.algorithm_dropdown.value,
                random_state=42,
            )
            model.fit(X_scaled)
            labels = model.labels_ 
            # Calculate metrics
            metrics_dict = calculate_clustering_metrics(X_scaled, labels, inertia=model.inertia_)
            result_text = format_results_markdown(metrics_dict, task_type="clustering")
            
            # Display results dialog with copy button
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "K-Means Clustering Results",
                result_text,
                "K-Means"
            )
            self.parent.page.open(evaluation_dialog)
            enable_navigation_bar(self.parent.page)
        
        except Exception as e:
            enable_navigation_bar(self.parent.page)
            self._show_snackbar(f"Training failed: {str(e)}", bgcolor=ft.Colors.RED_500)
        
        finally:
            self.parent.enable_model_selection()
            self.train_btn.disabled = False
            self.parent.page.update()
    
    def _reset_n_init_to_auto(self, e: ft.ControlEvent) -> None:
        self.n_init_field.value = "auto"
        self.parent.page.update()
        
    def _n_init_on_click(self, e: ft.ControlEvent) -> None:
        if e.control.value.strip() == "auto":
            e.control.value = ""
            self.parent.page.update()
            
    def _n_init_on_blur(self, e: ft.ControlEvent) -> None:
        if e.control.value.strip() == "":
            e.control.value = "auto"
            self.parent.page.update()
    
    def build_model_control(self) -> ft.Card:
        """Build Flet UI card for K-Means hyperparameter configuration."""
        
        self.n_clusters_field = ft.TextField(
            label="Number of Clusters",
            value="3",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The number of clusters to form as well as the number of centroids to generate.",
        )
        
        self.max_iter_field = ft.TextField(
            label="Max Iterations",
            value="300",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Maximum number of iterations of the k-means algorithm for a single run.",
        )

        self.init_dropdown = ft.Dropdown(
            label="Initialization",
            value="k-means++",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("k-means++"),
                ft.DropdownOption("random"),
            ],
            tooltip="Initialization strategy. k-means++=smart initialization (better), random=random centroids",
        )
        
        self.algorithm_dropdown = ft.Dropdown(
            label="Algoritm",
            value="lloyd",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            text_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.dropdown.Option("lloyd"),
                ft.dropdown.Option("elkan"),
            ],
            tooltip='K-means algorithm to use. The classical EM-style algorithm is "lloyd". The "elkan" variation can be more efficient on some datasets with well-defined clusters, by using the triangle inequality. However it’s more memory intensive due to the allocation of an extra array of shape (n_samples, n_clusters).',
        )
        
        self.n_init_field = ft.TextField(
            label="Number of Initializations",
            value="auto",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            on_click=self._n_init_on_click,
            on_blur=self._n_init_on_blur,
            suffix_icon=ft.TextButton('Auto', on_click=self._reset_n_init_to_auto),
            tooltip="Number of times the k-means algorithm is run with different centroid seeds. The final results is the best output of n_init consecutive runs in terms of inertia. When n_init='auto', the number of runs depends on the value of init: 10 if using init='random' or init is a callable; 1 if using init='k-means++' or init is an array-like.",
        )
        
        self._build_train_button()
        
        return ft.Card(
            expand=2,
            content=ft.Container(
                expand=True,
                margin=ft.margin.all(15),
                alignment=ft.alignment.center,
                content=ft.Column(
                    scroll=ft.ScrollMode.AUTO,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Text("K-Means Clustering",
                                       font_family="SF thin",
                                       size=24,
                                       text_align="center",
                                       expand=True)
                            ]
                        ),
                        ft.Divider(),
                        ft.Text("Hyperparameters",
                               font_family="SF regular",
                               weight="bold",
                               size=14),
                        ft.Row([self.n_clusters_field, self.max_iter_field]),
                        ft.Row([self.init_dropdown, self.algorithm_dropdown]),
                        self.n_init_field,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
