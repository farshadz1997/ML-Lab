"""
Data models for managing categorical encoding state during model training.

This module provides dataclasses for tracking encoding information across
train-test splits and for preserving encoding state in model results.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


@dataclass
class TrainingEncodingState:
    """
    Complete encoding state after model training.
    
    Stores metadata about categorical column encoding applied during training,
    enabling reproducibility and transparency about feature transformations.
    
    Attributes:
        categorical_cols: List of column names that were encoded
        numeric_cols: List of column names that were kept as-is
        encodings: Dict mapping column_name -> CategoricalEncodingInfo
        target_column: Name of the target column (optional, for supervised learning)
        target_encoding: Target column encoding info (optional)
        high_cardinality_warnings: Dict of column_name -> warning message
        pipeline_preserved: Whether the sklearn Pipeline with encoders was preserved
        timestamp: When the encoding was created
    """
    
    categorical_cols: List[str]
    numeric_cols: List[str]
    encodings: Dict[str, Any]  # Dict[str, CategoricalEncodingInfo]
    target_column: Optional[str] = None
    target_encoding: Optional[Dict[str, Any]] = None
    high_cardinality_warnings: Dict[str, str] = field(default_factory=dict)
    pipeline_preserved: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_encoding_summary(self) -> str:
        """
        Generate a human-readable summary of the encoding.
        
        Returns:
            Formatted string describing which columns were encoded and warnings
            
        Examples:
            >>> state = TrainingEncodingState(
            ...     categorical_cols=["color"],
            ...     numeric_cols=["age"],
            ...     encodings={}
            ... )
            >>> print(state.get_encoding_summary())
            Categorical Columns Encoded: color
            Numeric Columns (unchanged): age
        """
        lines = []
        
        if self.categorical_cols:
            lines.append(f"Categorical Columns Encoded: {', '.join(self.categorical_cols)}")
        else:
            lines.append("No categorical columns to encode.")
        
        if self.numeric_cols:
            lines.append(f"Numeric Columns (unchanged): {', '.join(self.numeric_cols)}")
        
        if self.high_cardinality_warnings:
            lines.append("\n⚠️  High Cardinality Warnings:")
            for col, warning in self.high_cardinality_warnings.items():
                lines.append(f"  - {col}: {warning}")
        
        if self.target_encoding:
            lines.append(f"\nTarget Column '{self.target_column}' encoded separately for interpretability")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize encoding state to dictionary.
        
        Returns:
            Dictionary representation of the encoding state
        """
        return {
            "categorical_cols": self.categorical_cols,
            "numeric_cols": self.numeric_cols,
            "encodings": self.encodings,
            "target_column": self.target_column,
            "target_encoding": self.target_encoding,
            "high_cardinality_warnings": self.high_cardinality_warnings,
            "pipeline_preserved": self.pipeline_preserved,
            "timestamp": self.timestamp,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Serialize encoding state to JSON string.
        
        Args:
            indent: JSON indentation level (default: 2)
            
        Returns:
            JSON string representation of the encoding state
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingEncodingState":
        """
        Deserialize encoding state from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            TrainingEncodingState instance
        """
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TrainingEncodingState":
        """
        Deserialize encoding state from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            TrainingEncodingState instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
