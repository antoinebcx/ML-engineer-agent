from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

@dataclass
class ModelIteration:
    iteration: int
    code: str
    score: float
    features_used: List[str]

class AIAgent:
    def __init__(self, data_handler, code_generator, model_evaluator, result_manager, task_type):
        self.data_handler = data_handler
        self.code_generator = code_generator
        self.model_evaluator = model_evaluator
        self.result_manager = result_manager
        self.task_type = task_type
        self.best_model = None
        self.iteration_history = []
        self.feature_importance = None
        self.original_features = None

    def optimize(self, num_iterations=20):
        try:
            self._calculate_feature_importance()
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            print("Continuing optimization without feature importance...")

        for i in range(num_iterations):
            prompt = self._generate_prompt(i)
            code = self.code_generator.generate_code(prompt)

            try:
                score, features_used = self.model_evaluator.evaluate_code(code, self.data_handler.get_data())
            except Exception as e:
                print(f"Error evaluating code in iteration {i+1}: {str(e)}")
                continue
            
            iteration = ModelIteration(i+1, code, score, features_used)
            self.iteration_history.append(iteration)
            self.result_manager.save_iteration(iteration)
            
            if self.best_model is None or score > self.best_model.score:
                self.best_model = iteration
                print(f"New best model found! Score: {score:.4f}")
            else:
                print(f"Iteration {i+1}: Score = {score:.4f} (Best: {self.best_model.score:.4f})")

    def _preprocess_data(self, X, y):
        self.original_features = X.columns.tolist()
        
        # Separate numerical and categorical columns
        num_columns = X.select_dtypes(include=['int64', 'float64']).columns
        cat_columns = X.select_dtypes(include=['object']).columns

        # Handle missing values for numerical columns
        if len(num_columns) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            X[num_columns] = num_imputer.fit_transform(X[num_columns])

        # Handle missing values and encode categorical columns
        for column in cat_columns:
            # Impute missing values with the most frequent value
            X[column] = X[column].fillna(X[column].mode()[0])
            # Encode categorical variables
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))

        # Convert y to numeric if it's categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))

        return X, y

    def _calculate_feature_importance(self):
        data = self.data_handler.get_data()
        
        if not all(key in data for key in ['X_train', 'y_train']):
            raise ValueError("get_data() should return a dict with at least 'X_train' and 'y_train' keys")

        X, y = data['X_train'], data['y_train']

        # Preprocess the data
        X, y = self._preprocess_data(X, y)

        # Create LightGBM datasets
        train_data = lgb.Dataset(X, label=y)

        # Set up parameters
        params = {
            'objective': 'regression' if self.task_type == 'regression' else 'multiclass',
            'metric': 'rmse' if self.task_type == 'regression' else 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'num_iterations': 100,
            'verbose': -1
        }

        # Train the model
        model = lgb.train(params, train_data)

        # Get feature importance
        feature_importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        
        self.feature_importance = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)

    def _generate_prompt(self, iteration):
        data_info = self.data_handler.get_data_summary()
        recent_iterations = self.result_manager.get_recent_iterations(5)
        best_iterations = self.result_manager.get_best_iterations(3)
        
        best_code = self.best_model.code if self.best_model else "No best model yet."

        feature_importance_str = self._format_feature_importance() if self.feature_importance else "Feature importance not available."

        prompt = f"""
        You are an expert machine learning engineer tasked with creating the best {self.task_type} model for the given data.
        
        Data summary: {data_info}
        Original features: {self.original_features}
        Current best score: {self.best_model.score if self.best_model else 'None'}
        Iteration: {iteration + 1}
        
        Current best model code:
        ```python
        {best_code}
        ```
        
        Recent performance history:
        {self._format_iterations(recent_iterations)}
        
        Best performing models:
        {self._format_iterations(best_iterations)}
        
        Feature importance analysis:
        {feature_importance_str}

        Based on this information, you have to build an even better model, focusing on:
        1. Data preprocessing (handling missing values, encoding categorical variables)
        2. Feature engineering
        3. Model architecture (optimized for {self.task_type})

        IMPORTANT: Ensure that your preprocessing step handles ALL input features. Do not drop any features unless you have a specific reason to do so.

        The code should follow this structure:
        1. Import necessary libraries
        2. Define a function called 'preprocess_data(X)' that takes a DataFrame X and returns preprocessed features
        3. Define a class called 'Model' with the following methods:
           - __init__(self): Initialize the model
           - fit(self, X, y): Fit the model to the data (do not include any additional parameters)
           - predict(self, X): Make predictions (do not include any additional parameters)
        4. Do not include any code to load data, train/test split or evaluate the model
        5. Do not use heavy grid search or automated hyperparameter optimization methods as it is an iterative process

        Example structure:
        ```python
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.impute import SimpleImputer

        def preprocess_data(X):
            # Separate numerical and categorical columns
            num_columns = X.select_dtypes(include=['int64', 'float64']).columns
            cat_columns = X.select_dtypes(include=['object']).columns

            # Handle missing values for numerical columns
            if len(num_columns) > 0:
                num_imputer = SimpleImputer(strategy='mean')
                X[num_columns] = num_imputer.fit_transform(X[num_columns])

            # Handle missing values and encode categorical columns
            for column in cat_columns:
                # Impute missing values with the most frequent value
                X[column] = X[column].fillna(X[column].mode()[0])
                # Encode categorical variables
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column].astype(str))
            
            return X

        class Model:
            def __init__(self):
                self.model = RandomForestRegressor()
                self.scaler = StandardScaler()

            def fit(self, X, y):
                X_preprocessed = preprocess_data(X)
                X_scaled = self.scaler.fit_transform(X_preprocessed)
                self.model.fit(X_scaled, y)

            def predict(self, X):
                X_preprocessed = preprocess_data(X)
                X_scaled = self.scaler.transform(X_preprocessed)
                return self.model.predict(X_scaled)
        ```

        You have access to the following libraries:
        pandas, numpy, scikit-learn, xgboost

        Follow best practices for data preparation (encoding, scaling...) and machine learning.
        Consider implementing early stopping and learning rate scheduling for deep learning and gradient boosting models.
        Start simple, with the necessary, and improve progressively.

        Provide only the runnable code in the specified format.
        The code you generate will be exported to a Python compiler for evaluation:
        ```python
        model = generated_model.Model()
        model.fit(data['X_train'], data['y_train'])
        y_pred = model.predict(data['X_val'])
        ```
        Ensure the code is complete and executable without any additional context or explanation outside the code itself.

        Improve progressively on the previous steps.
        """
        
        return prompt

    def _format_iterations(self, iterations):
        return "\n".join([f"Iteration {it.iteration}: Score = {it.score:.4f}, Features = {it.features_used}" for it in iterations])

    def _format_feature_importance(self):
        return "Most important features: " + ", ".join([f"{feature} ({importance:.4f})" for feature, importance in self.feature_importance])

    def get_best_model(self):
        return self.best_model