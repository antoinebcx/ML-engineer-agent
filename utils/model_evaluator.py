import importlib.util
import sys
from sklearn.metrics import root_mean_squared_error, r2_score, f1_score, accuracy_score
import numpy as np
import inspect

class ModelEvaluator:
    def __init__(self, task_type):
        self.task_type = task_type

    def evaluate_code(self, code, data):
        try:            
            # Save the cleaned code to a temporary file
            with open('temp_model.py', 'w') as f:
                f.write(code)

            # Import the temporary module
            spec = importlib.util.spec_from_file_location("temp_model", "temp_model.py")
            temp_model = importlib.util.module_from_spec(spec)
            sys.modules["temp_model"] = temp_model
            spec.loader.exec_module(temp_model)

            # Create and train the model
            model = temp_model.Model()
            
            # Use flexible fit method
            self._flexible_fit(model, data['X_train'], data['y_train'])

            # Make predictions
            y_pred = model.predict(data['X_val'])

            # Evaluate the model
            if self.task_type == 'regression':
                rmse = root_mean_squared_error(data['y_val'], y_pred)
                r2 = r2_score(data['y_val'], y_pred)
                score = r2
                print(f"RMSE: {rmse}, R²: {r2}")
            elif self.task_type == 'classification':
                f1 = f1_score(data['y_val'], y_pred, average='weighted')
                accuracy = accuracy_score(data['y_val'], y_pred)
                score = f1
                print(f"F1-score: {f1}, Accuracy: {accuracy}")

            # Get features used
            features_used = list(data['X_train'].columns)  # Assumes all features are used

            return score, features_used
        except Exception as e:
            print(f"Error evaluating code: {str(e)}")
            return float('-inf'), []

    def _flexible_fit(self, model, X_train, y_train, X_val=None, y_val=None):
        fit_signature = inspect.signature(model.fit)
        fit_params = fit_signature.parameters

        fit_args = {'X': X_train, 'y': y_train}

        optional_params = {
            # Common parameters
            'sample_weight': None,
            'eval_set': [(X_val, y_val)],
            'eval_metric': 'rmse' if self.task_type == 'regression' else 'logloss',
            'early_stopping_rounds': 50,
            'verbose': False,
            
            # XGBoost specific
            'xgb_model': None,
            'sample_weight_eval_set': None,
            'feature_weights': None,
            
            # Scikit-learn specific
            'sample_weight': None,
            'coef_init': None,
            'intercept_init': None,
            'warm_start': False,
            
            # CatBoost specific
            'cat_features': None,
            'eval_set': [(X_val, y_val)],
            'plot': False,
        }

        for param, value in optional_params.items():
            if param in fit_params:
                fit_args[param] = value

        # Call fit with the prepared arguments
        model.fit(**fit_args)