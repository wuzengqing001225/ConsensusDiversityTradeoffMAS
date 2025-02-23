import json
import os
from datetime import datetime
from typing import Dict, List
import numpy as np
from logger import ExperimentLogger
from baselines import run_baseline_experiments
from runner import InfoSpreadExperimentRunner

def load_config(config_path: str = "config/config.json") -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def run_main_experiment(api_keys: Dict[str, str],
                       logger: ExperimentLogger,
                       role_config: Dict,
                       seeds: List[int] = None) -> Dict:
    """Run main information spread experiment"""
    
    if seeds is None:
        seeds = list(range(5))
        
    diversity_levels = ['low', 'medium', 'high']
    volatility_levels = ['low', 'moderate', 'high']
    
    results = {div: {vol: [] for vol in volatility_levels} 
              for div in diversity_levels}
    
    for diversity in diversity_levels:
        for volatility in volatility_levels:
            print(f"\nRunning main experiment: {diversity} diversity, {volatility} volatility")
            
            for seed in seeds:
                print(f"  Seed {seed}")
                np.random.seed(seed)
                
                # Create unique run ID
                run_id = f"main_{diversity}_{volatility}_seed{seed}"
                
                # Run experiment
                runner = InfoSpreadExperimentRunner(
                    num_agents=3,
                    consensus_type='implicit',
                    diversity_level=diversity,
                    volatility=volatility,
                    api_keys=api_keys,
                    role_config=role_config
                )
                
                result = runner.run_experiment()
                results[diversity][volatility].append(result)
                
                # Log run
                if logger:
                    logger.log_run(
                        run_id=run_id,
                        config={
                            'experiment_type': 'main',
                            'diversity_level': diversity,
                            'volatility': volatility,
                            'seed': seed
                        },
                        round_logs=result['round_logs']
                    )
                    
    return results

def main():
    # Load configuration
    config = load_config()
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config['experiment_settings']['save_path'], timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Initialize logger
    logger = ExperimentLogger(base_dir=exp_dir)
    
    # Save experiment config
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Run baseline experiments
        print("\nRunning baseline experiments...")
        baseline_results = run_baseline_experiments(
            api_keys=config['api_keys'],
            logger=logger,
            role_config=config['info_spread_roles'],
            seeds=range(config['experiment_settings']['seeds'])
        )
        
        # Run main experiment
        print("\nRunning main experiment...")
        main_results = run_main_experiment(
            api_keys=config['api_keys'],
            logger=logger,
            role_config=config['info_spread_roles'],
            seeds=range(config['experiment_settings']['seeds'])
        )
        
        # Combine results
        all_results = {
            **baseline_results,
            'main_experiment': main_results
        }
        
        # Save final results
        results_file = os.path.join(exp_dir, 'experiment_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        print(f"\nExperiment completed. Results saved to: {exp_dir}")
        
        # Run analysis
        print("\nRunning analysis...")
        os.system(f"python analysis.py {exp_dir}")
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()