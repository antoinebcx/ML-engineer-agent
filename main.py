import os
import argparse
from dotenv import load_dotenv
from utils.ai_agent import AIAgent
from utils.data_handler import DataHandler
from utils.code_generator import CodeGenerator
from utils.model_evaluator import ModelEvaluator
from utils.result_manager import ResultManager

def main():
    load_dotenv()  # load environment variables from .env file
    
    parser = argparse.ArgumentParser(description="AI Agent for Data Science Tasks")
    parser.add_argument("--data", required=True, help="Path to the CSV data file")
    parser.add_argument("--task", choices=["regression", "classification"], required=True, help="Type of machine learning task")
    parser.add_argument("--target", required=True, help="Name of the target column")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations to run")
    parser.add_argument("--output", default="results", help="Directory to store results")
    args = parser.parse_args()
    
    print(f"Initializing AI Agent for {args.task} task...")
    print(f"Loading data from {args.data}")
    
    data_handler = DataHandler(args.data, args.target, args.task)
    code_generator = CodeGenerator()
    model_evaluator = ModelEvaluator(args.task)
    result_manager = ResultManager(args.output)
    
    agent = AIAgent(data_handler, code_generator, model_evaluator, result_manager, args.task)
    
    print(f"Starting optimization process for {args.iterations} iterations...")
    agent.optimize(num_iterations=args.iterations)
    
    best_model = agent.get_best_model()
    print("\nOptimization complete!")
    print(f"Best model achieved a score of {best_model.score:.4f} on iteration {best_model.iteration}")
    print("\nBest model code:")
    print(best_model.code)
    
    print("\nSaving best model...")
    result_manager.save_best_model(best_model)
    print(f"Best model saved in {args.output} directory")

    print("\nCleaning up temporary files...")
    result_manager.cleanup_temp_files()
    print("Cleanup complete.")

if __name__ == "__main__":
    main()