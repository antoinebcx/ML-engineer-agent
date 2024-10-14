import importlib.util
import sys
from sklearn.metrics import mean_squared_error, r2_score, f1_score
import numpy as np
import re

class ModelEvaluator:
    def __init__(self, task_type):
        self.task_type = task_type

    def clean_code(self, code):
        # Remove ```python and ``` delimiters
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*$', '', code)
        # Remove any leading or trailing whitespace
        code = code.strip()
        return code

    def evaluate_code(self, code, data):
        try:
            # Clean the code
            cleaned_code = self.clean_code(code)
            
            # Save the cleaned code to a temporary file
            with open('temp_model.py', 'w') as f:
                f.write(cleaned_code)

            # Import the temporary module
            spec = importlib.util.spec_from_file_location("temp_model", "temp_model.py")
            temp_model = importlib.util.module_from_spec(spec)
            sys.modules["temp_model"] = temp_model
            spec.loader.exec_module(temp_model)

            # Create and train the model
            model = temp_model.Model()
            model.fit(data['X_train'], data['y_train'])

            # Make predictions
            y_pred = model.predict(data['X_val'])

            # Evaluate the model
            if self.task_type == 'regression':
                mse = mean_squared_error(data['y_val'], y_pred)
                r2 = r2_score(data['y_val'], y_pred)
                score = r2  # Use R² score instead of negative MSE
                print(f"MSE: {mse}, R² Score: {r2}")
            else:  # classification
                score = f1_score(data['y_val'], y_pred, average='weighted')

            # Get features used and hyperparameters
            features_used = list(data['X_train'].columns)  # Assumes all features are used
            hyperparameters = model.model.get_params() if hasattr(model.model, 'get_params') else {}

            return score, features_used, hyperparameters
        except Exception as e:
            print(f"Error evaluating code: {str(e)}")
            return float('-inf'), [], {}