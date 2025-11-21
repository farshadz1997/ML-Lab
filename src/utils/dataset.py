from dataclasses import dataclass, field
from typing import Any, Dict, TYPE_CHECKING, Literal
import pandas as pd
from pathlib import Path
from datetime import datetime
from pandas._typing import Dtype


@dataclass
class DataSet:
    dataset_path: str
    path: Path = field(init=False)
    df: pd.DataFrame = field(init=False)
    file_name: str = field(init=False)

    def __post_init__(self):
        self.df: pd.DataFrame = pd.read_csv(self.dataset_path, encoding_errors="ignore")
        self.path = Path(self.dataset_path)
        self.file_name: str = self.path.stem
        
    @property
    def shape(self) -> tuple[int, int]:
        return self.df.shape
    
    def describe(
        self,
        percentiles: list[float] | None = None,
        include: Literal["all"] | list[Dtype] | None = None,
        exclude: list[Dtype] | None = None
    ) -> pd.DataFrame:
        return self.df.describe(percentiles, include, exclude)
    
    def custom_describe(self) -> pd.DataFrame:
        variables = []
        dtypes = []
        count = []
        unique = []
        missing = []
        for item in self.df.columns:
            variables.append(item)
            dtypes.append(self.df[item].dtype)
            count.append(len(self.df[item]))
            unique.append(len(self.df[item].unique()))
            missing.append(self.df[item].isna().sum())
        output = pd.DataFrame({
            'Variable': variables, 
            'Dtype': dtypes,
            'Count': count,
            'Unique': unique,
            'Missing value': missing
        })    
        return output

    def calculate_missing_percent(self) -> pd.DataFrame:
        percent_missing = self.df.isnull().sum() * 100 / len(self.df)
        missing_value_df = pd.DataFrame(
            {
                'column_name': self.df.columns,
                'percent_missing': percent_missing
            }
        )
        percent = pd.DataFrame(missing_value_df)
        return percent
    
    def get_rows_with_nan(self) -> pd.DataFrame:
        return self.df[self.df.isna().any(axis=1)]
    
    def get_duplicate_rows(self) -> pd.DataFrame:
        """Get all duplicate rows in the dataframe"""
        return self.df[self.df.duplicated(keep=False)]
    
    def drop_duplicates(self) -> bool:
        """Remove all duplicate rows, keeping only first occurrence"""
        try:
            initial_len = len(self.df)
            self.df.drop_duplicates(keep='first', inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            return True
        except Exception as e:
            print(e)
            return False
    
    def drop_nan_rows(self) -> bool:
        """Remove all rows containing NaN values"""
        try:
            self.df.dropna(inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            return True
        except Exception as e:
            print(e)
            return False
    
    def drop_column(self, column: str) -> bool:
        try:
            if column in self.df.columns:
                self.df.drop(columns=[column], inplace=True)
                return True
            return False
        except Exception as e:
            print(e)
            return False
        
    def drop_row(self, row_index: int) -> bool:
        try:
            if 0 <= row_index < len(self.df):
                self.df.drop(index=row_index, inplace=True)
                self.df.reset_index(drop=True, inplace=True)
                return True
            return False
        except Exception as e:
            print(e)
            return False
        
    def get_unique_values(self, column: str) -> list[Any] | None:
        try:
            if column in self.df.columns:
                return self.df[column].unique().tolist()
            return None
        except Exception as e:
            print(e)
            return None
    
    def rename_column(self, current_column: str, new_column: str) -> bool:
        try:
            if current_column in self.df.columns:
                self.df.rename(columns={current_column: new_column}, inplace=True)
                return True
            return False
        except Exception as e:
            print(e)
            return False
        
    def reset_index(self) -> None:
        self.df.reset_index(drop=True, inplace=True)
    
    def export_csv(self) -> str:
        path = Path(self.dataset_path).parent.absolute()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.df.to_csv(path / f"{self.file_name}-{timestamp}.csv")
        return str(path / f"{self.file_name}-{timestamp}.csv")