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
        """Initialize logger with timestamped directory"""
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, timestamp)
        
        # Create subdirectories
        self.runs_dir = os.path.join(self.exp_dir, "runs")
        self.plots_dir = os.path.join(self.exp_dir, "plots")
        self.metrics_dir = os.path.join(self.exp_dir, "metrics")
        self.analysis_dir = os.path.join(self.exp_dir, "analysis")
        
        for dir_path in [self.exp_dir, self.runs_dir, self.plots_dir, 
                        self.metrics_dir, self.analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Initialize summary metrics
        self.summary_metrics = {
            'disaster_response': {},
            'info_spread': {},
            'public_goods': {}
        }
        
        # Start experiment log
        self.log_experiment_start()
        
    def log_experiment_start(self):
        """Log experiment start time and basic info"""
        start_info = {
            'timestamp': datetime.now().isoformat(),
            'experiment_dir': self.exp_dir,
            'scenarios': [
                'disaster_response',
                'info_spread',
                'public_goods'
            ]
        }
        
        with open(os.path.join(self.exp_dir, 'experiment_info.json'), 'w') as f:
            json.dump(start_info, f, indent=2)
            
    def log_run(self, 
                run_id: str,
                config: Dict,
                round_logs: List[Dict[str, Any]],
                scenario: str = None):
        """Log a single experimental run"""
        # Create run directory
        run_dir = os.path.join(self.runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save run configuration
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
        # Save round-by-round agent interactions
        interaction_rows = []
        for round_idx, round_data in enumerate(round_logs):
            agent_interactions = round_data.get('agent_interactions', {})
            
            for agent_id, agent_data in agent_interactions.items():
                row = {
                    'round': round_idx,
                    'agent_id': agent_id,
                    'role': agent_data.get('role', ''),
                    'prompt': agent_data.get('prompt', ''),
                    'response': json.dumps(agent_data.get('response', {})),
                    'environment_state': round_data.get('environment_state', '')
                }
                interaction_rows.append(row)
                
        interactions_df = pd.DataFrame(interaction_rows)
        interactions_df.to_csv(
            os.path.join(run_dir, "agent_interactions.csv"), 
            index=False,
            encoding='utf-8'
        )
        
        # Save metrics
        metrics_rows = []
        for round_idx, round_data in enumerate(round_logs):
            metrics = round_data.get('metrics', {})
            metrics['round'] = round_idx
            metrics_rows.append(metrics)
            
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(
            os.path.join(run_dir, "metrics.csv"),
            index=False
        )
        
        # Save scenario-specific data
        if scenario:
            scenario_data = self._extract_scenario_data(round_logs, scenario)
            with open(os.path.join(run_dir, f"{scenario}_data.json"), "w") as f:
                json.dump(scenario_data, f, indent=2)
                
    def _extract_scenario_data(self, 
                             round_logs: List[Dict],
                             scenario: str) -> Dict:
        """Extract scenario-specific data from round logs"""
        if scenario == 'disaster_response':
            return {
                'coverage_rates': [log['metrics']['coverage_rate'] 
                                 for log in round_logs],
                'misallocation_penalties': [log['metrics']['misallocation_penalty'] 
                                          for log in round_logs],
                'response_delays': [log['metrics']['response_delay'] 
                                  for log in round_logs]
            }
            
        elif scenario == 'info_spread':
            return {
                'misinformation_spread': [log['metrics']['misinformation_spread'] 
                                        for log in round_logs],
                'containment_times': [log['metrics']['containment_time'] 
                                    for log in round_logs],
                'coverage_diversity': [log['metrics']['coverage_diversity'] 
                                     for log in round_logs]
            }
            
        elif scenario == 'public_goods':
            return {
                'provision_rates': [log['metrics']['provision_rate'] 
                                  for log in round_logs],
                'total_welfare': [log['metrics']['total_welfare'] 
                                for log in round_logs],
                'contribution_disparity': [log['metrics']['contribution_disparity'] 
                                         for log in round_logs]
            }
            
        return {}
        
    def log_agent_thoughts(self,
                          run_id: str,
                          agent_id: int,
                          round_idx: int,
                          thoughts: Dict):
        """Log agent's thought process and chain-of-thought reasoning"""
        run_dir = os.path.join(self.runs_dir, run_id)
        thoughts_dir = os.path.join(run_dir, "agent_thoughts")
        os.makedirs(thoughts_dir, exist_ok=True)
        
        thought_file = os.path.join(thoughts_dir, 
                                  f"agent_{agent_id}_round_{round_idx}.json")
        with open(thought_file, "w") as f:
            json.dump(thoughts, f, indent=2)
            
    def save_analysis_results(self,
                            scenario: str,
                            analysis_data: Dict,
                            plots: Dict[str, str] = None):
        """Save analysis results and plots for a scenario"""
        # Save analysis data
        scenario_dir = os.path.join(self.analysis_dir, scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        
        with open(os.path.join(scenario_dir, "analysis.json"), "w") as f:
            json.dump(analysis_data, f, indent=2)
            
        # Save plots if provided
        if plots:
            plots_dir = os.path.join(scenario_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            for plot_name, plot_data in plots.items():
                plot_path = os.path.join(plots_dir, f"{plot_name}.png")
                plt.figure()
                # Assume plot_data contains the necessary plotting information
                plt.savefig(plot_path)
                plt.close()
                
    def update_summary_metrics(self,
                             scenario: str,
                             metrics: Dict):
        """Update summary metrics for a scenario"""
        self.summary_metrics[scenario].update(metrics)
        
        # Save updated summary
        with open(os.path.join(self.metrics_dir, "summary_metrics.json"), "w") as f:
            json.dump(self.summary_metrics, f, indent=2)
            
    def export_experiment_summary(self):
        """Export comprehensive experiment summary"""
        summary = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'directory': self.exp_dir,
                'total_runs': len(os.listdir(self.runs_dir))
            },
            'scenarios': {
                scenario: {
                    'metrics': metrics,
                    'analysis_path': os.path.join(self.analysis_dir, scenario)
                }
                for scenario, metrics in self.summary_metrics.items()
            }
        }
        
        # Save summary
        with open(os.path.join(self.exp_dir, "experiment_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
        return summary