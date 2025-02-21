import json
import os
from src.logger import ExperimentLogger
from experiments.baselines import (run_baseline_experiments, 
                                 run_main_experiment,
                                 analyze_results)

def load_config(config_path: str = "config/config.json") -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize logger
    logger = ExperimentLogger(
        base_dir=config['experiment_settings']['save_path']
    )
    
    # Run baseline experiments
    print("Running baseline experiments...")
    baseline_results = run_baseline_experiments(
        api_keys=config['api_keys'],
        logger=logger,
        role_config=config['agent_roles'],
        seeds=range(config['experiment_settings']['seeds'])
    )
    
    # Run main experiment
    print("\nRunning main experiment...")
    main_results = run_main_experiment(
        api_keys=config['api_keys'],
        logger=logger,
        role_config=config['agent_roles'],
        seeds=range(config['experiment_settings']['seeds'])
    )
    
    # Combine results
    all_results = {
        **baseline_results,
        'main_experiment': main_results
    }
    
    # Save experiment data
    exp_dir = logger.exp_dir
    
    # Run analysis
    print("\nAnalyzing results...")
    os.system(f"python analysis.py {exp_dir}")

if __name__ == "__main__":
    main()