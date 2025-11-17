from dataclasses import dataclass, field
from typing import Any, Dict, TYPE_CHECKING, Literal
import pandas as pd
from pathlib import Path

from pandas._typing import Dtype


@dataclass
class DataSet:
    dataset_path: str

    def __post_init__(self):
        self.df: pd.DataFrame = pd.read_csv(self.dataset_path, encoding_errors="ignore")
        
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
    
    def drop_column(self, column: str) -> bool:
        try:
            if column in self.df.columns:
                self.df.drop(columns=[column], inplace=True)
                return True
            return False
        except Exception as e:
            print(e)
            return False
    
    def rename_column(self, current_column: str, new_column: str) -> bool:
        try:
            if current_column in self.df.columns:
                self.df.rename(columns={current_column: new_column}, inplace=True)
                return True
            return False
        except Exception as e:
            print(e)
            return False
    
    def export_csv(self) -> None:
        path = Path(self.dataset_path).parent.absolute()
        self.df.to_csv(path / "new_data.csv")