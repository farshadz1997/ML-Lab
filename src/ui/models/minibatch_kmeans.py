"""
MiniBatch K-Means Clustering Model

Memory-efficient variant of K-Means using mini-batch sampling.
Faster training with slightly lower quality clusters, good for large datasets.

Configurable hyperparameters:
- n_clusters: Number of clusters
- batch_size: Size of each mini-batch
- n_init: Number of initializations
- random_state: Random seed for reproducibility
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import flet as ft
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer

from utils.model_utils import (
    validate_hyperparameters,
    calculate_clustering_metrics,
    format_results_markdown,
    create_results_dialog,
    disable_navigation_bar,
    enable_navigation_bar,
)
from core.data_preparation import prepare_data_for_training_no_split

if TYPE_CHECKING:
    from ..model_factory import ModelFactory


@dataclass
class MiniBatchKMeansModel:
    """MiniBatch K-Means clustering model."""
    
    parent: ModelFactory
    df: DataFrame
    
    def __post_init__(self):
        """Ensure dataset is copied to avoid mutations."""
        self.df = self.df.copy()
    
    def _prepare_data(self) -> Optional[Tuple]:
        """
        Prepare data for clustering with categorical encoding support (no train-test split).
        
        Uses prepare_data_for_training_no_split() which:
        - Detects and encodes categorical columns
        - Validates cardinality
        - Fits encoders on full dataset (clustering is unsupervised)
        - Returns encoded features and metadata
        
        Returns:
            Tuple of (X_scaled, feature_cols) or None if error
        """
        try:
            # Call spec-compliant data preparation for clustering (no split)
            (
                X_encoded,
                _,  # y is None for clustering
                categorical_cols,
                numeric_cols,
                encoders,
                cardinality_warnings,
            ) = prepare_data_for_training_no_split(
                self.df.copy(),
                target_col=None,  # No target for clustering
                raise_on_unseen=True,
            )
            
            # Store encoding metadata
            self.categorical_cols = categorical_cols
            self.numeric_cols = numeric_cols
            self.encoders = encoders
            self.cardinality_warnings = cardinality_warnings
            
            # Warn about high-cardinality columns
            if cardinality_warnings:
                warning_msgs = [
                    f"{col}: {w.message}"
                    for col, w in cardinality_warnings.items()
                ]
                self.parent.page.open(ft.SnackBar(
                    ft.Text(
                        "Cardinality warnings: " + "; ".join(warning_msgs),
                        font_family="SF regular",
                    ),
                    bgcolor="#FF9800"
                ))
            
            # Apply scaling if requested
            if self.parent.scaler_dropdown.value == "standard_scaler":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_encoded)
            elif self.parent.scaler_dropdown.value == "minmax_scaler":
                scaler = MinMaxScaler(feature_range=(0, 1))
                X_scaled = scaler.fit_transform(X_encoded)
            else:
                X_scaled = X_encoded.values
            
            return X_scaled, X_encoded.columns.tolist()
        
        except Exception as e:
            self.parent.page.open(ft.SnackBar(
                ft.Text(f"Data preparation error: {str(e)}", font_family="SF regular")
            ))
            return None
    
    def _train_and_evaluate_model(self, e: ft.ControlEvent) -> None:
        """Train MiniBatch K-Means model and display evaluation results."""
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
                'batch_size': int(self.batch_size_field.value),
                'n_init': int(self.n_init_field.value) if self.n_init_field.value.strip() != "auto" else "auto",
                'max_iter': int(self.max_iter_field.value),
            }
            
            validation_rules = {
                'n_clusters': {'type': int, 'min': 2},
                'batch_size': {'type': int, 'min': 1},
                'n_init': {'type': int, 'min': 1, 'max': 100} if self.n_init_field.value.strip() != "auto" else {"type": str, "allowed": ['auto']},
                'max_iter': {'type': int, 'min': 1},
            }
            
            is_valid, error_msg = validate_hyperparameters(hyperparams, 'minibatch_kmeans', validation_rules)
            if not is_valid:
                self.parent.page.open(ft.SnackBar(
                    ft.Text(f"Hyperparameter error: {error_msg}", font_family="SF regular")
                ))
                return
            
            # Train MiniBatch K-Means
            model = MiniBatchKMeans(
                n_clusters=int(self.n_clusters_field.value),
                init=self.init_dropdown.value,
                batch_size=int(self.batch_size_field.value),
                n_init=int(self.n_init_field.value) if self.n_init_field.value.strip() != "auto" else "auto",
                random_state=42,
                max_iter=int(self.max_iter_field.value),
            )
            labels = model.fit_predict(X_scaled)
            
            # Calculate metrics
            metrics_dict = calculate_clustering_metrics(X_scaled, labels, inertia=model.inertia_)
            result_text = format_results_markdown(metrics_dict, task_type="clustering")
            
            # Display results dialog with copy button
            evaluation_dialog = create_results_dialog(
                self.parent.page,
                "MiniBatch K-Means Clustering Results",
                result_text,
                "MiniBatch K-Means"
            )
            self.parent.page.open(evaluation_dialog)
            enable_navigation_bar(self.parent.page)
        
        except Exception as e:
            enable_navigation_bar(self.parent.page)
            self.parent.page.open(ft.SnackBar(
                ft.Text(f"Training failed: {str(e)}", font_family="SF regular")
            ))
        
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
        """Build Flet UI card for MiniBatch K-Means hyperparameter configuration."""
        
        self.n_clusters_field = ft.TextField(
            label="Number of Clusters",
            value="3",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="The number of clusters to form as well as the number of centroids to generate.",
        )
        
        self.batch_size_field = ft.TextField(
            label="Batch Size",
            value="1024",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Size of the mini batches. For faster computations, you can set batch_size > 256 * number_of_cores to enable parallelism on all cores.",
        )
        
        self.init_dropdown = ft.Dropdown(
            label="Initialization",
            value="k-means++",
            expand=1,
            label_style=ft.TextStyle(font_family="SF regular"),
            options=[
                ft.DropdownOption("k-means++", text_style=ft.TextStyle(font_family="SF regular")),
                ft.DropdownOption("random", text_style=ft.TextStyle(font_family="SF regular")),
            ],
            tooltip="Initialization strategy. k-means++=smart initialization (better), random=random centroids",
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
            suffix=ft.TextButton('Auto', on_click=self._reset_n_init_to_auto),
            tooltip="Number of times the k-means algorithm is run with different centroid seeds. The final results is the best output of n_init consecutive runs in terms of inertia. When n_init='auto', the number of runs depends on the value of init: 10 if using init='random' or init is a callable; 1 if using init='k-means++' or init is an array-like.",
        )

        self.max_iter_field = ft.TextField(
            label="Max Iterations",
            value="100",
            expand=1,
            text_style=ft.TextStyle(font_family="SF regular"),
            label_style=ft.TextStyle(font_family="SF regular"),
            input_filter=ft.NumbersOnlyInputFilter(),
            tooltip="Maximum number of iterations over the complete dataset before stopping independently of any early stopping criterion heuristics.",
        )
        
        self.train_btn = ft.FilledButton(
            text="Train and evaluate model",
            icon=ft.Icons.PSYCHOLOGY,
            on_click=self._train_and_evaluate_model,
            expand=1,
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
                    scroll=ft.ScrollMode.AUTO,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Text("MiniBatch K-Means Clustering",
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
                        ft.Row([self.n_clusters_field, self.batch_size_field]),
                        ft.Row([self.init_dropdown, self.max_iter_field]),
                        self.n_init_field,
                        ft.Row([self.train_btn])
                    ]
                )
            )
        )
