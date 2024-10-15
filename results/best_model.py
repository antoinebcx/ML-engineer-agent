import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(X):
    # Separate numeric and categorical features
    numeric_features = X.select_dtypes(include=['float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create a ColumnTransformer to handle both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Apply transformations
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed

class Model:
    def __init__(self):
        # Initialize a RandomForestRegressor with hyperparameters
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.preprocessor = None

    def fit(self, X, y):
        # Preprocess the data
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), X.select_dtypes(include=['float64']).columns.tolist()),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), X.select_dtypes(include=['object']).columns.tolist())
            ])
        
        X_preprocessed = self.preprocessor.fit_transform(X)
        # Fit the model
        self.model.fit(X_preprocessed, y)

    def predict(self, X):
        # Preprocess the data
        X_preprocessed = self.preprocessor.transform(X)
        # Make predictions
        return self.model.predict(X_preprocessed)