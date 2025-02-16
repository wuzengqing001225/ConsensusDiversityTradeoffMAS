import os
import json
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

class ExperimentLogger:
    def __init__(self, base_dir: str = "experiment_logs"):
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, timestamp)
        
        # Create subdirectories
        self.runs_dir = os.path.join(self.exp_dir, "runs")
        self.plots_dir = os.path.join(self.exp_dir, "plots")
        self.metrics_dir = os.path.join(self.exp_dir, "metrics")
        
        for dir_path in [self.exp_dir, self.runs_dir, self.plots_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Initialize metric trackers
        self.deviation_metrics = []
        self.performance_metrics = []
        
        # Start experiment log
        self.log_experiment_start()
        
    def log_experiment_start(self):
        """Log experiment start with timestamp and basic info"""
        start_info = {
            'timestamp': datetime.now().isoformat(),
            'experiment_dir': self.exp_dir
        }
        
        with open(os.path.join(self.exp_dir, 'experiment_info.json'), 'w') as f:
            json.dump(start_info, f, indent=2)
            
    def log_run(self, 
                run_id: str,
                config: Dict,
                round_logs: List[Dict[str, Any]]):
        """Log a single experimental run"""
        
        run_dir = os.path.join(self.runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save run configuration
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
        # Save round-by-round agent interactions
        interaction_rows = []
        for round_idx, round_data in enumerate(round_logs):
            for agent_id, agent_data in round_data['agent_interactions'].items():
                row = {
                    'round': round_idx,
                    'agent_id': agent_id,
                    'role': agent_data['role'],
                    'prompt': agent_data['prompt'],
                    'analysis': agent_data['response']['analysis'],
                    'action': str(agent_data['response']['action']),
                    'message': agent_data['response']['message'],
                    'environment_state': round_data['environment_state']
                }
                interaction_rows.append(row)
                
        # Save interactions CSV
        interactions_df = pd.DataFrame(interaction_rows)
        interactions_df.to_csv(os.path.join(run_dir, "agent_interactions.csv"), 
                             index=False, encoding='utf-8')
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'round': range(len(round_logs)),
            'coverage_rate': [r['metrics']['coverage_rate'] for r in round_logs],
            'misallocation_penalty': [r['metrics']['misallocation_penalty'] for r in round_logs],
            'response_delay': [r['metrics']['response_delay'] for r in round_logs],
            'mean_deviation': [r['metrics']['mean_deviation'] for r in round_logs]
        })
        metrics_df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
        
        # Track deviation-performance relationship
        self.deviation_metrics.extend([r['metrics']['mean_deviation'] for r in round_logs])
        self.performance_metrics.extend([
            r['metrics']['coverage_rate'] - 
            0.1 * r['metrics']['misallocation_penalty'] - 
            0.1 * r['metrics']['response_delay'] 
            for r in round_logs
        ])
        
    def plot_deviation_performance(self):
        """Plot relationship between deviation and performance"""
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot with trend line
        sns.regplot(x=self.deviation_metrics, 
                   y=self.performance_metrics,
                   scatter_kws={'alpha':0.5},
                   line_kws={'color': 'red'})
        
        plt.xlabel('Mean Agent Deviation (d)')
        plt.ylabel('Performance (P)')
        plt.title('Relationship between Agent Deviation and System Performance')
        
        # Save plot
        plt.savefig(os.path.join(self.plots_dir, "deviation_performance.png"))
        plt.close()
        
        # Save raw data
        pd.DataFrame({
            'deviation': self.deviation_metrics,
            'performance': self.performance_metrics
        }).to_csv(os.path.join(self.metrics_dir, "deviation_performance.csv"), index=False)
        
    def plot_metrics_by_condition(self, results: Dict, 
                                condition_names: List[str] = ['diversity', 'volatility']):
        """Plot metrics for different experimental conditions"""
        metrics = ['coverage_rate', 'misallocation_penalty', 'response_delay']
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            data = []
            
            for cond1 in results:
                for cond2 in results[cond1]:
                    for run in results[cond1][cond2]:
                        # Access aggregate metrics from the run result
                        metric_value = run['aggregate_metrics'][metric]['mean']
                        data.append({
                            condition_names[0]: cond1,
                            condition_names[1]: cond2,
                            'value': metric_value
                        })
                    
            df = pd.DataFrame(data)
            sns.boxplot(data=df, x=condition_names[0], y='value', hue=condition_names[1])
            
            plt.title(f'{metric.replace("_", " ").title()} by {condition_names[0]} and {condition_names[1]}')
            plt.savefig(os.path.join(self.plots_dir, f"{metric}_by_condition.png"))
            plt.close()
            
    def save_final_metrics(self, aggregate_results: Dict):
        """Save final aggregate metrics"""
        with open(os.path.join(self.metrics_dir, 'aggregate_results.json'), 'w') as f:
            json.dump(aggregate_results, f, indent=2)