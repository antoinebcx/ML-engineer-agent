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
        You generate the best Python code for the given task.
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