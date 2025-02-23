import os
import json
import csv
import pandas as pd
from datetime import datetime
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
                    'target_nodes': str(agent_data['response']['target_nodes']),
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
            'misinformation_spread': [r['metrics']['misinformation_spread'] for r in round_logs],
            'containment_time': [r['metrics']['containment_time'] for r in round_logs],
            'coverage_diversity': [r['metrics']['coverage_diversity'] for r in round_logs],
            'mean_deviation': [r['metrics']['mean_deviation'] for r in round_logs]
        })
        metrics_df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
        
        # Track deviation-performance relationship
        self.deviation_metrics.extend([r['metrics']['mean_deviation'] for r in round_logs])
        self.performance_metrics.extend([
            r['metrics']['coverage_diversity'] - 
            0.1 * r['metrics']['misinformation_spread'] - 
            0.1 * r['metrics']['containment_time'] 
            for r in round_logs
        ])
    
    def save_final_metrics(self, aggregate_results: Dict):
        """Save final aggregate metrics"""
        with open(os.path.join(self.metrics_dir, 'aggregate_results.json'), 'w') as f:
            json.dump(aggregate_results, f, indent=2)