"""
Data preparation utilities for model training with categorical column support.

This module provides the prepare_data_for_training() function that handles:
- Categorical column detection
- Cardinality validation with warnings
- Train-test split (BEFORE encoding, to prevent data leakage)
- Encoder creation and fitting (on training data ONLY)
- Data encoding and validation
"""

from typing import Tuple, List, Dict, Optional, Any, Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder, TargetEncoder, OrdinalEncoder

from utils.model_utils import (
    detect_categorical_columns,
    validate_cardinality,
    create_categorical_encoders,
    apply_encoders,
    encode_with_one_hot_encoder,
    CardinalityWarning,
    EncodingError,
)


def prepare_data_for_training(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    raise_on_unseen: bool = True,
    scaler_mode: Literal['standard_scaler', 'minmax_scaler', 'none'] = 'standard_scaler',
    features_encoder: Literal["OrdinalEncoder", "TargetEncoder", "LabelEncoder", "OneHotEncoder"] = "OrdinalEncoder",
    target_encoder: Literal["None", "LabelEncoder"] = "None"
) -> Tuple[
    np.ndarray,
    np.ndarray,
    pd.Series,
    pd.Series,
    List[str],
    List[str],
    Dict[str, LabelEncoder | TargetEncoder | OrdinalEncoder] | OneHotEncoder,
    Optional[StandardScaler | MinMaxScaler],
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
        scaler_mode: What scaler to use: 'none' | 'standard_scaler' | 'minmax_scaler'
    
    Returns:
        Tuple of:
        - X_train: Encoded training features
        - X_test: Encoded test features
        - y_train: Training targets (not encoded for now)
        - y_test: Test targets
        - categorical_cols: List of categorical column names
        - numeric_cols: List of numeric column names
        - encoders: Dict of fitted LabelEncoders for categorical columns
        - scaler: could return one of StandardScaler or MinMaxScaler or None
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
    df = df.copy()
    # Step 1: Validate target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Step 2: Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if target_encoder == "LabelEncoder" and target_col in df.select_dtypes(include=["object", "category"]).columns.tolist():
        df[target_col] = LabelEncoder().fit_transform(y)
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
            if features_encoder == "OneHotEncoder":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoders = encoder
            else:
                encoders = create_categorical_encoders(X_train, y_train, categorical_cols, features_encoder)
        except EncodingError as e:
            raise ValueError(f"Failed to create encoders: {str(e)}")
    
    # Step 7: Apply encoders to training data
    X_train_encoded = X_train.copy()
    if categorical_cols:
        if features_encoder == "OneHotEncoder":
            X_train_encoded = encode_with_one_hot_encoder(encoder, X_train_encoded, categorical_cols, True)
        else:
            X_train_encoded = apply_encoders(
                encoders,
                X_train_encoded,
                raise_on_unknown=False,  # Training data should not have unknown values
            )
    
    # Step 8: Apply same encoders to test data (detect unseen values)
    X_test_encoded = X_test.copy()
    if categorical_cols:
        try:
            if features_encoder == "OneHotEncoder":
                X_test_encoded = encode_with_one_hot_encoder(encoder, X_test_encoded, categorical_cols, False)
            else:
                X_test_encoded = apply_encoders(
                    encoders,
                    X_test_encoded,
                    raise_on_unknown=raise_on_unseen,
                )
        except EncodingError as e:
            if raise_on_unseen:
                raise
            # Otherwise, log warning and continue (for MVP, skipping log)
    
    # Step 9: Apply scaling based on user selection
    if scaler_mode == "standard_scaler":
        scaler = StandardScaler()
        scaler.fit(X_train_encoded)
        X_train_encoded = scaler.transform(X_train_encoded)
        X_test_encoded = scaler.transform(X_test_encoded)
    elif scaler_mode == "minmax_scaler":
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train_encoded)
        X_train_encoded = scaler.transform(X_train_encoded)
        X_test_encoded = scaler.transform(X_test_encoded)
    else:
        scaler = None
        X_train_encoded = X_train_encoded.values
        X_test_encoded = X_test_encoded.values
        
    return (
        X_train_encoded,
        X_test_encoded,
        y_train,
        y_test,
        categorical_cols,
        numeric_cols,
        encoders,
        scaler,
        cardinality_warnings,
    )


def prepare_data_for_training_no_split(
    df: pd.DataFrame,
    target_col: str,
    raise_on_unseen: bool = True,
    scaler_mode: Literal['standard_scaler', 'minmax_scaler', 'none'] = 'standard_scaler',
    features_encoder: Literal["OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "LabelEncoder"] = "OrdinalEncoder",
    target_encoder: Literal["None", "LabelEncoder"] = "None"
) -> Tuple[
    np.ndarray,
    pd.Series | None,
    List[str],
    List[str],
    Dict[str, LabelEncoder | OrdinalEncoder | TargetEncoder] | OneHotEncoder,
    Optional[StandardScaler | MinMaxScaler],
    Dict[str, CardinalityWarning],
]:
    """
    Prepare data for training without test-train split (for clustering/unsupervised).
    
    Args:
        df: Input DataFrame
        target_col: Target column (for supervised), or None for unsupervised
        raise_on_unseen: If True, raise error on unseen categories
        features_encoder: Type of encoder for features
        target_encoder: Type of encoder for target
    
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
        if target_encoder == "LabelEncoder" and target_col in df.select_dtypes(include=["object", "category"]).columns.tolist():
            y = LabelEncoder().fit_transform(y)
            df[target_col] = y
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
            if features_encoder == "OneHotEncoder":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoders = encoder
            else:
                encoders = create_categorical_encoders(X, y, categorical_cols, features_encoder)
        except EncodingError as e:
            raise ValueError(f"Failed to create encoders: {str(e)}")
    
    # Apply encoders
    X_encoded = X.copy()
    if categorical_cols:
        try:
            if features_encoder == "OneHotEncoder":
                X_encoded = encode_with_one_hot_encoder(encoders, X_encoded, categorical_cols, True)
            else:
                X_encoded = apply_encoders(
                    encoders,
                    X_encoded,
                    raise_on_unknown=raise_on_unseen,
                )
        except EncodingError as e:
            if raise_on_unseen:
                raise
    
    # Apply scaling based on user selection
    if scaler_mode == "standard_scaler":
        scaler = StandardScaler()
        X_encoded = scaler.fit_transform(X_encoded)
    elif scaler_mode == "minmax_scaler":
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_encoded = scaler.fit_transform(X_encoded)
    else:
        scaler = None
    
    return X_encoded, y, categorical_cols, numeric_cols, encoders, scaler, cardinality_warnings
