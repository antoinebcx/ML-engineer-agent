import os
import json
from datetime import datetime

class ResultManager:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.iterations = []

    def save_iteration(self, iteration):
        self.iterations.append(iteration)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iteration_{iteration.iteration}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                'iteration': iteration.iteration,
                'code': iteration.code,
                'score': iteration.score,
                'features_used': iteration.features_used,
                'hyperparameters': iteration.hyperparameters
            }, f, indent=2)

    def get_recent_iterations(self, n=5):
        return sorted(self.iterations, key=lambda x: x.iteration, reverse=True)[:n]

    def get_best_iterations(self, n=3):
        return sorted(self.iterations, key=lambda x: x.score, reverse=True)[:n]

    def save_best_model(self, best_model):
        # Save model details
        filename = "best_model.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                'iteration': best_model.iteration,
                'code': best_model.code,
                'score': best_model.score,
                'features_used': best_model.features_used,
                'hyperparameters': best_model.hyperparameters
            }, f, indent=2)

        # Save model code
        code_filename = "best_model.py"
        code_filepath = os.path.join(self.results_dir, code_filename)
        
        with open(code_filepath, 'w') as f:
            f.write(best_model.code)

    def cleanup_temp_files(self):
        if os.path.exists('temp_model.py'):
            os.remove('temp_model.py')