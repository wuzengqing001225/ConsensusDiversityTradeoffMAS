from typing import Dict, List
import numpy as np
from src.logger import ExperimentLogger
from experiments.runner import ExperimentRunner

def run_baseline_experiments(
    api_keys: Dict[str, str],
    logger: ExperimentLogger,
    role_config: Dict,
    seeds: List[int] = None
) -> Dict:
    """Run all baseline experiments"""
    
    if seeds is None:
        seeds = list(range(5))
    
    # Configuration for different baselines
    baseline_configs = {
        'single_llm': {
            'num_agents': 1,
            'consensus_type': 'implicit',
            'diversity_level': 'none'
        },
        'no_interaction': {
            'num_agents': 10,
            'consensus_type': 'none',
            'diversity_level': 'none'
        },
        'explicit_consensus': {
            'num_agents': 10,
            'consensus_type': 'explicit',
            'diversity_level': 'medium'
        },
        'random_strategy': {
            'num_agents': 10,
            'consensus_type': 'implicit',
            'diversity_level': 'random'
        },
        'no_diversity': {
            'num_agents': 10,
            'consensus_type': 'implicit',
            'diversity_level': 'low'
        }
    }
    
    # Experiment configurations
    diversity_levels = ['low', 'medium', 'high']
    volatility_levels = ['low', 'moderate', 'high']
    
    results = {}
    
    # Run baselines
    for baseline, config in baseline_configs.items():
        baseline_results = {div: {vol: [] for vol in volatility_levels} 
                          for div in diversity_levels}
        
        for diversity in diversity_levels:
            for volatility in volatility_levels:
                for seed in seeds:
                    np.random.seed(seed)
                    
                    # Create unique run ID
                    run_id = f"{baseline}_{diversity}_{volatility}_seed{seed}"
                    
                    # Update config with current settings
                    run_config = config.copy()
                    if baseline != 'no_diversity':  # Respect diversity setting except for no_diversity baseline
                        run_config['diversity_level'] = diversity
                    run_config['volatility'] = volatility
                    
                    # Run experiment
                    runner = ExperimentRunner(
                        **run_config,
                        api_keys=api_keys,
                        logger=logger,
                        role_config=role_config
                    )
                    
                    result = runner.run_experiment()
                    baseline_results[diversity][volatility].append(result)
                    
                    # Log run data
                    if logger:
                        logger.log_run(
                            run_id=run_id,
                            config={
                                'baseline': baseline,
                                'diversity_level': diversity,
                                'volatility': volatility,
                                'seed': seed,
                                **run_config
                            },
                            round_logs=result['round_logs']
                        )
                        
        results[baseline] = baseline_results
    
    return results

def run_main_experiment(
    api_keys: Dict[str, str],
    logger: ExperimentLogger,
    role_config: Dict,
    seeds: List[int] = None
) -> Dict:
    """Run main experiment with implicit consensus"""
    
    if seeds is None:
        seeds = list(range(5))
    
    diversity_levels = ['low', 'medium', 'high']
    volatility_levels = ['low', 'moderate', 'high']
    
    results = {div: {vol: [] for vol in volatility_levels} 
              for div in diversity_levels}
    
    for diversity in diversity_levels:
        for volatility in volatility_levels:
            for seed in seeds:
                np.random.seed(seed)
                
                # Create unique run ID
                run_id = f"main_{diversity}_{volatility}_seed{seed}"
                
                # Run experiment
                runner = ExperimentRunner(
                    num_agents=3,
                    consensus_type='implicit',
                    diversity_level=diversity,
                    volatility=volatility,
                    api_keys=api_keys,
                    logger=logger,
                    role_config=role_config
                )
                
                result = runner.run_experiment()
                results[diversity][volatility].append(result)
                
                # Log run data
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

def analyze_results(results: Dict, logger: ExperimentLogger) -> Dict:
    """Analyze and compare experiment results"""
    analysis = {}
    
    # Calculate mean performance for each condition
    for exp_type in results:
        analysis[exp_type] = {}
        for diversity in results[exp_type]:
            analysis[exp_type][diversity] = {}
            for volatility in results[exp_type][diversity]:
                runs = results[exp_type][diversity][volatility]
                
                # Extract performance scores from aggregate metrics
                performance_scores = []
                for run in runs:
                    try:
                        # Calculate performance score from individual metrics
                        metrics = run['aggregate_metrics']
                        performance = (
                            metrics['coverage_rate']['mean'] -
                            0.1 * metrics['misallocation_penalty']['mean'] -
                            0.1 * metrics['response_delay']['mean']
                        )
                        performance_scores.append(performance)
                    except (KeyError, TypeError) as e:
                        print(f"Warning: Could not calculate performance for run in {exp_type}/{diversity}/{volatility}: {e}")
                        continue
                
                if performance_scores:
                    analysis[exp_type][diversity][volatility] = {
                        'mean_performance': np.mean(performance_scores),
                        'std_performance': np.std(performance_scores),
                        'num_runs': len(performance_scores)
                    }
                else:
                    print(f"Warning: No valid performance scores for {exp_type}/{diversity}/{volatility}")
    
    # Generate plots
    if logger:
        logger.plot_metrics_by_condition(results)
        logger.plot_deviation_performance()
        
    return analysis