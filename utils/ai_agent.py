from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class ModelIteration:
    iteration: int
    code: str
    score: float
    features_used: List[str]
    hyperparameters: Dict

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
            score, features_used, hyperparameters = self.model_evaluator.evaluate_code(code, self.data_handler.get_data())
            
            iteration = ModelIteration(i+1, code, score, features_used, hyperparameters)
            self.iteration_history.append(iteration)
            self.result_manager.save_iteration(iteration)
            
            if self.best_model is None or score > self.best_model.score:
                self.best_model = iteration
            
            print(f"Iteration {i+1}: Score = {score:.4f}")

    def _generate_prompt(self, iteration):
        data_info = self.data_handler.get_data_summary()
        recent_iterations = self.result_manager.get_recent_iterations(5)
        best_iterations = self.result_manager.get_best_iterations(3)
        
        feature_importance = self._analyze_feature_importance(recent_iterations)
        hyperparameter_trends = self._analyze_hyperparameter_trends(recent_iterations)
        
        return f"""
        Generate Python code for a {self.task_type} model to predict the target variable.
        Data summary: {data_info}
        Current best score: {self.best_model.score if self.best_model else 'None'}
        Iteration: {iteration + 1}
        
        Recent performance history:
        {self._format_iterations(recent_iterations)}
        
        Best performing models:
        {self._format_iterations(best_iterations)}
        
        Feature importance analysis:
        {feature_importance}
        
        Hyperparameter trends:
        {hyperparameter_trends}
        
        Based on this information, suggest improvements focusing on:
        1. Feature selection and engineering
        2. Model architecture (appropriate for {self.task_type})
        3. Hyperparameter tuning
        4. Ensemble methods (if appropriate)
        
        Provide the complete code for the improved model, including necessary imports and data preprocessing steps.
        """

    def _format_iterations(self, iterations):
        return "\n".join([f"Iteration {it.iteration}: Score = {it.score}, Features = {it.features_used}, Hyperparameters = {it.hyperparameters}" for it in iterations])

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

    def _analyze_hyperparameter_trends(self, iterations):
        hyperparameter_values = {}
        for it in iterations:
            for param, value in it.hyperparameters.items():
                if param not in hyperparameter_values:
                    hyperparameter_values[param] = []
                hyperparameter_values[param].append((value, it.score))
        
        trends = {}
        for param, values in hyperparameter_values.items():
            sorted_values = sorted(values, key=lambda x: x[1], reverse=True)  # Sort by score
            trends[param] = f"Best value: {sorted_values[0][0]} (score: {sorted_values[0][1]:.4f})"
        
        return "\n".join([f"{param}: {trend}" for param, trend in trends.items()])

    def get_best_model(self):
        return self.best_model