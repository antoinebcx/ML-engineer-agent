from dataclasses import dataclass
from typing import List, Dict
import numpy as np

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

    def optimize(self, num_iterations=20):
        for i in range(num_iterations):
            prompt = self._generate_prompt(i)
            code = self.code_generator.generate_code(prompt)

            score, features_used = self.model_evaluator.evaluate_code(code, self.data_handler.get_data())
            
            iteration = ModelIteration(i+1, code, score, features_used)
            self.iteration_history.append(iteration)
            self.result_manager.save_iteration(iteration)
            
            if self.best_model is None or score > self.best_model.score:
                self.best_model = iteration
                print(f"New best model found! Score: {score:.4f}")
            else:
                print(f"Iteration {i+1}: Score = {score:.4f} (Best: {self.best_model.score:.4f})")

    def _generate_prompt(self, iteration):
        data_info = self.data_handler.get_data_summary()
        recent_iterations = self.result_manager.get_recent_iterations(5)
        best_iterations = self.result_manager.get_best_iterations(3)
        
        feature_importance = self._analyze_feature_importance(recent_iterations)
        
        best_code = self.best_model.code if self.best_model else "No best model yet."

        prompt = f"""
        You are an expert machine learning engineer tasked with creating the best {self.task_type} model for the given data.
        
        Data summary: {data_info}
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
        {feature_importance}

        Based on this information, you have to build an even better model, focusing on:
        1. Feature selection and engineering
        2. Model architecture (appropriate for {self.task_type})

        ----

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
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        def preprocess_data(X):
            # Preprocess the data
            # ...
            return X_preprocessed

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

        ----

        You have access to the following libraries:
        pandas, numpy, scikit-learn, xgboost

        Follow best practices for data preparation (encoding, scaling...) and machine learning.
        Consider implementing early stopping and learning rate scheduling for deep learning and gradient boosting models.

        Provide only the runnable code in the specified format.
        The code you generate will be exported to a Python compiler for evaluation:
        ```python
        model = generated_model.Model()
        model.fit(data['X_train'], data['y_train'])
        y_pred = model.predict(data['X_val'])
        ```
        Ensure the code is complete and executable without any additional context or explanation outside the code itself.
        """
        
        return prompt

    def _format_iterations(self, iterations):
        return "\n".join([f"Iteration {it.iteration}: Score = {it.score:.4f}, Features = {it.features_used}" for it in iterations])

    def _analyze_feature_importance(self, iterations):
        feature_scores = {}
        for it in iterations:
            for feature in it.features_used:
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append(it.score)
        
        feature_importance = {feature: np.mean(scores) for feature, scores in feature_scores.items()}
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return "Most important features (based on average score): " + ", ".join([f"{feature} ({score:.4f})" for feature, score in sorted_features[:5]])

    def get_best_model(self):
        return self.best_model