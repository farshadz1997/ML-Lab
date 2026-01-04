#!/usr/bin/env python3
"""
Quick integration test to verify T018-T020 implementation works with real data.
"""
import pandas as pd
from src.core.data_preparation import prepare_data_for_training
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_integration():
    # Load customer data
    df = pd.read_csv('Customer_Data.csv')
    
    # Drop CUST_ID (not a feature)
    df = df.drop('CUST_ID', axis=1)
    
    print(f'✓ Loaded Customer_Data.csv: shape {df.shape}')
    
    # Create a binary target based on balance (high/low)
    df['HIGH_BALANCE'] = (df['BALANCE'] > df['BALANCE'].median()).astype(int)
    target_col = 'HIGH_BALANCE'
    
    cat_cols_before = df.select_dtypes(include=['object']).columns.tolist()
    print(f'  Categorical columns in data: {cat_cols_before}')
    
    # Prepare data
    try:
        X_train, X_test, y_train, y_test, cat_cols, num_cols, encoders, warnings = prepare_data_for_training(
            df, target_col, test_size=0.2, random_state=42
        )
        print(f'✓ Data preparation successful!')
        print(f'  X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')
        print(f'  Categorical columns detected: {cat_cols}')
        print(f'  Numeric columns detected: {len(num_cols)}')
        
        if warnings:
            print(f'  Cardinality warnings: {list(warnings.keys())}')
        else:
            print(f'  No cardinality warnings')
    except Exception as e:
        print(f'✗ Data preparation failed: {e}')
        return False
    
    # Test LogisticRegression
    try:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'✓ LogisticRegression training successful!')
        print(f'  Test accuracy: {accuracy:.4f}')
    except Exception as e:
        print(f'✗ LogisticRegression training failed: {e}')
        return False
    
    # Test RandomForest
    try:
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'✓ RandomForestClassifier training successful!')
        print(f'  Test accuracy: {accuracy:.4f}')
    except Exception as e:
        print(f'✗ RandomForestClassifier training failed: {e}')
        return False
    
    print(f'\n✓ All integration tests passed! T018-T020 implementation verified.')
    return True

if __name__ == '__main__':
    success = test_integration()
    exit(0 if success else 1)
