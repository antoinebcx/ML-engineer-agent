import os
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

class CodeGenerator:
    def __init__(self):
        pass

    def clean_code(self, code):
        # Extract content between ```python and ``` delimiters or remove them
        match = re.search(r'```python\s*([\s\S]*?)\s*```', code)
        if match:
            code = match.group(1)
        else:
            code = re.sub(r'```python\s*', '', code)
            code = re.sub(r'```\s*$', '', code)
        
        code = code.strip()
        return code

    def generate_code(self, prompt):
        system_message = """
        You are an exceptional machine learning engineer.
        Generate the best Python code for the given task.
        The code should follow this structure:

        1. Import necessary libraries
        2. Define a function called 'preprocess_data(X)' that takes a DataFrame X and returns preprocessed features
        3. Define a class called 'Model' with the following methods:
           - __init__(self): Initialize the model
           - fit(self, X, y): Fit the model to the data
           - predict(self, X): Make predictions
        4. Do not include any code to load data or evaluate the model

        ----

        Example structure:

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

        ----

        You have access to the following libraries:
        pandas, numpy, scikit-learn, lightgbm, xgboost

        Follow best practices for data preparation and machine learning.

        Give only the runnable code in the given format and only that.
        There should be a correct `Model` class with `fit` and `predict` methods.
        You can use intermediary steps but make sure this class and its methods are working and are enough for evaluation.
        The code you generate will be exported to a Python compiler to evaluate it.
        """

        try:
            response = client.chat.completions.create(
                model='gpt-4o-2024-08-06',
                # later, potentially structured outputs or {"type": "json_object"}
                response_format={"type": "text"},
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4000,
                temperature=0.5
            )
            return self.clean_code(response.choices[0].message.content)
        except Exception as e:
            raise Exception(f"Failed to generate code with GPT: {e}")