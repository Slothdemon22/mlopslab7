#!/usr/bin/env python3
"""
House Price Prediction Training Script
This script trains a machine learning model for house price prediction using DVC parameters.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import joblib
import json
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_params(params_path="params.yaml"):
    """Load parameters from YAML file."""
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params


def slugify(name):
    """Convert string to slug format for form fields."""
    name = str(name)
    name = name.strip()
    # replace non-alphanumeric with underscore
    return re.sub(r'[^0-9a-zA-Z_]+', '_', name)


def load_and_preprocess_data(params):
    """Load and preprocess the dataset according to parameters."""
    print("Loading dataset...")
    df = pd.read_csv(params['data']['input_file'])
    print(f"Original dataset shape: {df.shape}")
    
    # Drop specified columns
    if 'drop_columns' in params['features']:
        df = df.drop(columns=params['features']['drop_columns'], errors='ignore')
        print(f"After dropping columns: {df.shape}")
    
    # Handle missing values
    if params['preprocessing']['drop_na']:
        df_clean = df.dropna(axis=0, how='any').reset_index(drop=True)
        print(f"After dropping NA values: {df_clean.shape}")
    else:
        df_clean = df.copy()
    
    # Select only the features we need
    feature_columns = params['features']['categorical_columns'] + params['features']['numeric_columns']
    feature_columns.append(params['data']['target_column'])
    
    df_clean = df_clean[feature_columns]
    print(f"Final dataset shape: {df_clean.shape}")
    
    return df_clean


def encode_categorical_features(df, params):
    """Encode categorical features using LabelEncoder."""
    print("Encoding categorical features...")
    encoders = {}
    
    for col in params['features']['categorical_columns']:
        if col in df.columns:
            le = LabelEncoder()
            # Convert all to string first so LabelEncoder is deterministic
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"Encoded '{col}' -> {len(le.classes_)} classes")
    
    return df, encoders


def train_model(X_train, y_train, params):
    """Train the model using specified parameters."""
    print("Training model...")
    
    model_params = params['model'].copy()
    model_type = model_params.pop('type')
    
    if model_type == "RandomForestRegressor":
        model = RandomForestRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    print(f"Model trained with {model_type}")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    metrics = {
        'mse': float(mean_squared_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred))
    }
    
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    return metrics


def save_artifacts(model, features_list, encoders, feature_field_map, metrics, params):
    """Save all model artifacts and metrics."""
    print("Saving artifacts...")
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    # Save model and artifacts
    joblib.dump(model, params['outputs']['model_path'])
    joblib.dump(features_list, params['outputs']['features_path'])
    joblib.dump(encoders, params['outputs']['encoders_path'])
    joblib.dump(feature_field_map, params['outputs']['feature_map_path'])
    
    # Save metrics
    with open(params['outputs']['metrics_path'], 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Artifacts saved successfully!")
    print(f"Model: {params['outputs']['model_path']}")
    print(f"Features: {params['outputs']['features_path']}")
    print(f"Encoders: {params['outputs']['encoders_path']}")
    print(f"Feature Map: {params['outputs']['feature_map_path']}")
    print(f"Metrics: {params['outputs']['metrics_path']}")


def main():
    """Main training pipeline."""
    print("Starting House Price Prediction Training Pipeline")
    print("=" * 50)
    
    # Load parameters
    params = load_params()
    
    # Load and preprocess data
    df = load_and_preprocess_data(params)
    
    # Encode categorical features
    df_encoded, encoders = encode_categorical_features(df, params)
    
    # Prepare features and target
    X = df_encoded.drop(params['data']['target_column'], axis=1)
    y = df_encoded[params['data']['target_column']]
    
    # Create feature field mapping for web form
    features_list = X.columns.tolist()
    feature_field_map = {col: slugify(col) for col in features_list}
    
    print(f"Features used: {features_list}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['data']['test_size'], 
        random_state=params['data']['random_state']
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    model = train_model(X_train, y_train, params)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save artifacts
    save_artifacts(model, features_list, encoders, feature_field_map, metrics, params)
    
    print("=" * 50)
    print("Training pipeline completed successfully!")
    return metrics


if __name__ == "__main__":
    main()
