import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

def preprocess_data(X):
    # Identify numerical and categorical columns
    numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                          'total_bedrooms', 'population', 'households', 'median_income']
    categorical_features = ['ocean_proximity']
    
    # Create preprocessing pipelines for both numeric and categorical data
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine the pipelines into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
    
    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed

class Model:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
        self.preprocessor = None

    def fit(self, X, y):
        # Preprocess the data
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                     'total_bedrooms', 'population', 'households', 'median_income']),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), ['ocean_proximity'])
            ])
        
        X_preprocessed = self.preprocessor.fit_transform(X)
        self.model.fit(X_preprocessed, y)

    def predict(self, X):
        X_preprocessed = self.preprocessor.transform(X)
        return self.model.predict(X_preprocessed)