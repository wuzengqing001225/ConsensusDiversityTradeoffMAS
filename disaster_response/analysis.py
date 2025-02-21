import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

def load_run_data(run_dir: str) -> Dict:
    """Load data from a single experimental run"""
    try:
        # Load config
        with open(os.path.join(run_dir, "config.json"), 'r') as f:
            config = json.load(f)
            
        # Load interactions
        interactions_df = pd.read_csv(os.path.join(run_dir, "agent_interactions.csv"))
        
        # Load metrics
        metrics_df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
        
        # Reconstruct round logs
        round_logs = []
        for round_idx in metrics_df['round'].unique():
            round_metrics = metrics_df[metrics_df['round'] == round_idx].iloc[0]
            round_interactions = interactions_df[interactions_df['round'] == round_idx]
            
            agent_interactions = {}
            for _, row in round_interactions.iterrows():
                agent_interactions[row['agent_id']] = {
                    'role': row['role'],
                    'prompt': row['prompt'],
                    'response': {
                        'analysis': row['analysis'],
                        'action': eval(row['action']),
                        'message': row['message']
                    }
                }
            
            round_logs.append({
                'environment_state': round_interactions.iloc[0]['environment_state'],
                'agent_interactions': agent_interactions,
                'metrics': {
                    'coverage_rate': round_metrics['coverage_rate'],
                    'misallocation_penalty': round_metrics['misallocation_penalty'],
                    'response_delay': round_metrics['response_delay'],
                    'mean_deviation': round_metrics['mean_deviation']
                }
            })
        
        return {
            'config': config,
            'round_logs': round_logs
        }
        
    except Exception as e:
        print(f"Error loading run from {run_dir}: {e}")
        return None

def reconstruct_results(exp_dir: str) -> Dict:
    """Reconstruct results dictionary from experiment directory"""
    runs_dir = os.path.join(exp_dir, "runs")
    if not os.path.exists(runs_dir):
        raise ValueError(f"Runs directory not found: {runs_dir}")
        
    results = {}
    
    # Process each run directory
    for run_id in os.listdir(runs_dir):
        run_dir = os.path.join(runs_dir, run_id)
        if not os.path.isdir(run_dir):
            continue
            
        run_data = load_run_data(run_dir)
        if run_data is None:
            continue
            
        # Parse run_id to get conditions
        # Expected format: {experiment_type}_{diversity}_{volatility}_seed{seed}
        parts = run_id.split('_')
        if len(parts) < 4:
            print(f"Skipping run with invalid ID format: {run_id}")
            continue
            
        exp_type = parts[0]
        diversity = parts[1]
        volatility = parts[2]
        
        # Initialize nested dictionaries if needed
        if exp_type not in results:
            results[exp_type] = {}
        if diversity not in results[exp_type]:
            results[exp_type][diversity] = {}
        if volatility not in results[exp_type][diversity]:
            results[exp_type][diversity][volatility] = []
            
        results[exp_type][diversity][volatility].append(run_data)
        
    return results

def generate_summary_statistics(results: Dict) -> Dict:
    """Generate summary statistics for all conditions"""
    summary = {}
    
    for condition, condition_results in results.items():
        summary[condition] = {}
        for diversity in condition_results:
            summary[condition][diversity] = {}
            for volatility in condition_results[diversity]:
                runs = condition_results[diversity][volatility]
                metrics = {
                    'coverage_rate': [],
                    'misallocation_penalty': [],
                    'response_delay': [],
                    'mean_deviation': []
                }
                
                for run in runs:
                    for metric in metrics:
                        values = [round_data['metrics'][metric] 
                                for round_data in run['round_logs']]
                        metrics[metric].append(np.mean(values))
                
                summary[condition][diversity][volatility] = {
                    metric: {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                    for metric, values in metrics.items()
                }
    
    return summary

def analyze_experiment(exp_dir: str):
    """Analyze experiment results from directory"""
    print(f"\nAnalyzing experiment results from: {exp_dir}")
    print("=" * 50)
    
    # Create analysis directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(exp_dir, f"analysis_{timestamp}")
        
    try:
        # Load and reconstruct results
        print("\nLoading experimental data...")
        results = reconstruct_results(exp_dir)
        
        # Print experiment structure
        print("\nExperiment Structure:")
        for condition in results:
            print(f"\nCondition: {condition}")
            for diversity in results[condition]:
                print(f"  Diversity: {diversity}")
                for volatility in results[condition][diversity]:
                    num_runs = len(results[condition][diversity][volatility])
                    print(f"    Volatility: {volatility} ({num_runs} runs)")
        
        # Generate summary statistics
        print("\nGenerating summary statistics...")
        summary_stats = generate_summary_statistics(results)
        
        # Save summary statistics
        stats_file = os.path.join(analysis_dir, 'summary_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Summary statistics saved to: {stats_file}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # Deviation-performance relationship
        print("\nAnalyzing deviation-performance relationship...")
        deviation_data = []
        total_rounds = 0
        for exp_type, conditions in results.items():
            for diversity in conditions:
                for volatility in conditions[diversity]:
                    for run in conditions[diversity][volatility]:
                        for round_data in run['round_logs']:
                            total_rounds += 1
                            performance = (round_data['metrics']['coverage_rate'] -
                                        0.1 * round_data['metrics']['misallocation_penalty'] -
                                        0.1 * round_data['metrics']['response_delay'])
                            performance = (performance + 1) / 2  # Normalize to [0,1]
                            
                            deviation_data.append({
                                'deviation': round_data['metrics']['mean_deviation'],
                                'performance': performance,
                                'condition': f"{exp_type}_{diversity}_{volatility}"
                            })
        
        print(f"Processed {total_rounds} rounds of data")
        deviation_df = pd.DataFrame(deviation_data)
        
        # Summary dashboard
        print("\nAnalysis completed successfully!")
        print("=" * 50)
        print(f"\nResults saved to: {analysis_dir}")
        print(f"Generated files:")
        print(f"- Summary statistics: {stats_file}")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        raise
