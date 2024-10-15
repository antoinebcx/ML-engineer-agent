import importlib.util
import sys
from sklearn.metrics import root_mean_squared_error, r2_score, f1_score, accuracy_score
import numpy as np

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
            model.fit(data['X_train'], data['y_train'])

            # Make predictions
            y_pred = model.predict(data['X_val'])

            # Evaluate the model
            if self.task_type == 'regression':
                rmse = root_mean_squared_error(data['y_val'], y_pred)
                r2 = r2_score(data['y_val'], y_pred)
                score = rmse
                print(f"RMSE: {rmse}, R²: {r2}")
            elif self.task_type == 'classification':
                f1 = f1_score(data['y_val'], y_pred, average='weighted')
                accuracy = accuracy_score(data['y_val'], y_pred, average='weighted')
                score = f1
                print(f"F-score: {score}, Accuracy: {accuracy}")

            # Get features used
            features_used = list(data['X_train'].columns)  # Assumes all features are used

            return score, features_used
        except Exception as e:
            print(f"Error evaluating code: {str(e)}")
            return float('-inf'), [], {}