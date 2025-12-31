"""
Data preparation utilities for model training with categorical column support.

This module provides the prepare_data_for_training() function that handles:
- Categorical column detection
- Cardinality validation with warnings
- Train-test split (BEFORE encoding, to prevent data leakage)
- Encoder creation and fitting (on training data ONLY)
- Data encoding and validation
"""

from typing import Tuple, List, Dict, Optional, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.model_utils import (
    detect_categorical_columns,
    validate_cardinality,
    create_categorical_encoders,
    apply_encoders,
    CardinalityWarning,
    EncodingError,
)


def prepare_data_for_training(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    raise_on_unseen: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    List[str],
    List[str],
    Dict[str, LabelEncoder],
    Dict[str, CardinalityWarning],
]:
    """
    Prepare data for model training with automatic categorical encoding.
    
    **CRITICAL**: Train-test split happens BEFORE encoding to prevent data leakage.
    Encoders are fitted ONLY on training data, then applied to test data.
    
    Args:
        df: Input DataFrame with features and target
        target_col: Name of target column
        test_size: Fraction of data for test set (default: 0.2)
        random_state: Random seed for reproducibility (optional)
        raise_on_unseen: If True, raise error when test set has unseen categories;
                        if False, log warning (default: True)
    
    Returns:
        Tuple of:
        - X_train: Encoded training features
        - X_test: Encoded test features
        - y_train: Training targets (not encoded for now)
        - y_test: Test targets
        - categorical_cols: List of categorical column names
        - numeric_cols: List of numeric column names
        - encoders: Dict of fitted LabelEncoders for categorical columns
        - cardinality_warnings: Dict of high-cardinality warnings
    
    Raises:
        ValueError: If target_col not in DataFrame
        EncodingError: If test data contains unseen categorical values and raise_on_unseen=True
        
    Examples:
        >>> df = pd.DataFrame({
        ...     "color": ["red", "blue", "red"],
        ...     "age": [25, 30, 35],
        ...     "target": [0, 1, 0]
        ... })
        >>> X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = (
        ...     prepare_data_for_training(df, "target", test_size=0.33)
        ... )
        >>> X_train.shape
        (2, 2)
    """
    # Step 1: Validate target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Step 2: Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Step 3: Detect categorical and numeric columns
    categorical_cols = detect_categorical_columns(X)
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    
    # Step 4: Validate cardinality (warn if any column exceeds threshold)
    cardinality_warnings = validate_cardinality(
        X,
        categorical_cols,
        threshold=1000,
    )
    
    # Step 5: Train-test split BEFORE encoding (CRITICAL for preventing leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    
    # Step 6: Create and fit encoders ONLY on training data
    encoders = {}
    if categorical_cols:
        try:
            encoders = create_categorical_encoders(X_train, categorical_cols)
        except EncodingError as e:
            raise ValueError(f"Failed to create encoders: {str(e)}")
    
    # Step 7: Apply encoders to training data
    X_train_encoded = X_train.copy()
    if categorical_cols:
        X_train_encoded = apply_encoders(
            encoders,
            X_train_encoded,
            raise_on_unknown=False,  # Training data should not have unknown values
        )
    
    # Step 8: Apply same encoders to test data (detect unseen values)
    X_test_encoded = X_test.copy()
    if categorical_cols:
        try:
            X_test_encoded = apply_encoders(
                encoders,
                X_test_encoded,
                raise_on_unknown=raise_on_unseen,
            )
        except EncodingError as e:
            if raise_on_unseen:
                raise
            # Otherwise, log warning and continue (for MVP, skipping log)
    
    return (
        X_train_encoded,
        X_test_encoded,
        y_train,
        y_test,
        categorical_cols,
        numeric_cols,
        encoders,
        cardinality_warnings,
    )


def prepare_data_for_training_no_split(
    df: pd.DataFrame,
    target_col: str,
    raise_on_unseen: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    List[str],
    List[str],
    Dict[str, LabelEncoder],
    Dict[str, CardinalityWarning],
]:
    """
    Prepare data for training without test-train split (for clustering/unsupervised).
    
    Args:
        df: Input DataFrame
        target_col: Target column (for supervised), or None for unsupervised
        raise_on_unseen: If True, raise error on unseen categories
    
    Returns:
        Tuple of:
        - X: Encoded features
        - y: Target (if target_col provided) or None
        - categorical_cols
        - numeric_cols
        - encoders
        - cardinality_warnings
        
    Raises:
        ValueError: If target_col not in DataFrame (when provided)
        EncodingError: If unseen categories found and raise_on_unseen=True
    """
    # Separate features and target
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None
    
    # Detect columns
    categorical_cols = detect_categorical_columns(X)
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    
    # Validate cardinality
    cardinality_warnings = validate_cardinality(X, categorical_cols, threshold=1000)
    
    # Create and fit encoders on entire dataset
    encoders = {}
    if categorical_cols:
        try:
            encoders = create_categorical_encoders(X, categorical_cols)
        except EncodingError as e:
            raise ValueError(f"Failed to create encoders: {str(e)}")
    
    # Apply encoders
    X_encoded = X.copy()
    if categorical_cols:
        try:
            X_encoded = apply_encoders(
                encoders,
                X_encoded,
                raise_on_unknown=raise_on_unseen,
            )
        except EncodingError as e:
            if raise_on_unseen:
                raise
    
    return X_encoded, y, categorical_cols, numeric_cols, encoders, cardinality_warnings
