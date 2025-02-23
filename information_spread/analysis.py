import os
import json
import argparse
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiment_data(exp_dir: str) -> Dict:
    """Load and reconstruct experiment data"""
    # Load results file
    results_file = os.path.join(exp_dir, 'experiment_results.json')
    if not os.path.exists(results_file):
        raise ValueError(f"Results file not found: {results_file}")
        
    with open(results_file, 'r') as f:
        return json.load(f)

def generate_network_analysis(results: Dict, save_dir: str):
    """Generate network-specific analysis plots"""
    plt.figure(figsize=(12, 6))
    
    # Analyze coverage patterns
    coverage_data = []
    for condition, condition_results in results.items():
        for diversity in condition_results:
            for volatility in condition_results[diversity]:
                for run in condition_results[diversity][volatility]:
                    for round_data in run['round_logs']:
                        coverage_data.append({
                            'condition': condition,
                            'diversity': diversity,
                            'volatility': volatility,
                            'round': round_data['round_idx'],
                            'coverage_diversity': round_data['metrics']['coverage_diversity'],
                            'influence_score': round_data['metrics']['influence_score']
                        })
    
    df = pd.DataFrame(coverage_data)
    
    # Plot coverage diversity vs influence score
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='coverage_diversity', y='influence_score', 
                   hue='condition', style='volatility')
    plt.title('Coverage Strategy Analysis')
    
    # Plot time evolution of coverage
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df[df['condition'] == 'main_experiment'],
                x='round', y='coverage_diversity', hue='diversity')
    plt.title('Coverage Evolution Over Time')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'network_analysis.png'))
    plt.close()

def analyze_misinformation_containment(results: Dict, save_dir: str):
    """Analyze misinformation containment effectiveness"""
    containment_data = []
    
    for condition, condition_results in results.items():
        for diversity in condition_results:
            for volatility in condition_results[diversity]:
                for run in condition_results[diversity][volatility]:
                    run_metrics = []
                    for round_data in run['round_logs']:
                        metrics = round_data['metrics']
                        run_metrics.append({
                            'condition': condition,
                            'diversity': diversity,
                            'volatility': volatility,
                            'round': round_data['round_idx'],
                            'spread': metrics['misinformation_spread'],
                            'containment_time': metrics['containment_time']
                        })
                    
                    # Calculate containment effectiveness
                    initial_spread = run_metrics[0]['spread']
                    final_spread = run_metrics[-1]['spread']
                    containment_rate = (initial_spread - final_spread) / initial_spread if initial_spread > 0 else 0
                    
                    containment_data.append({
                        'condition': condition,
                        'diversity': diversity,
                        'volatility': volatility,
                        'containment_rate': containment_rate,
                        'avg_containment_time': np.mean([m['containment_time'] for m in run_metrics])
                    })
    
    df = pd.DataFrame(containment_data)
    
    # Plot containment analysis
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, x='condition', y='containment_rate')
    plt.title('Containment Rate by Condition')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df[df['condition'] == 'main_experiment'],
                x='diversity', y='containment_rate', hue='volatility')
    plt.title('Impact of Diversity on Containment')
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x='containment_rate', y='avg_containment_time',
                   hue='condition', style='volatility')
    plt.title('Containment Rate vs Time')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'containment_analysis.png'))
    plt.close()

def analyze_consensus_patterns(results: Dict, save_dir: str):
    """Analyze consensus formation patterns"""
    consensus_data = []
    
    for condition, condition_results in results.items():
        for diversity in condition_results:
            for volatility in condition_results[diversity]:
                for run in condition_results[diversity][volatility]:
                    for round_data in run['round_logs']:
                        metrics = round_data['metrics']
                        consensus_data.append({
                            'condition': condition,
                            'diversity': diversity,
                            'volatility': volatility,
                            'round': round_data['round_idx'],
                            'mean_deviation': metrics['mean_deviation'],
                            'consensus_ratio': metrics['consensus_ratio'],
                            'network_diversity': metrics.get('network_diversity', 0)
                        })
    
    df = pd.DataFrame(consensus_data)
    
    # Plot consensus analysis
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, x='condition', y='consensus_ratio')
    plt.title('Consensus Ratio by Condition')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    time_evolution = df[df['condition'] == 'main_experiment'].groupby(
        ['round', 'diversity'])['consensus_ratio'].mean().reset_index()
    sns.lineplot(data=time_evolution, x='round', y='consensus_ratio', hue='diversity')
    plt.title('Consensus Evolution Over Time')
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x='mean_deviation', y='network_diversity',
                   hue='condition', style='volatility')
    plt.title('Deviation vs Network Diversity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'consensus_analysis.png'))
    plt.close()

def analyze_experiment(exp_dir: str):
    """Analyze experiment results and generate visualizations"""
    print(f"\nAnalyzing information spread experiment from: {exp_dir}")
    print("=" * 50)
    
    # Create analysis directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(exp_dir, f"analysis_{timestamp}")
    plots_dir = os.path.join(analysis_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Load results
        print("\nLoading experimental data...")
        results = load_experiment_data(exp_dir)
                
        # Generate network-specific analysis
        print("\nAnalyzing network patterns...")
        generate_network_analysis(results, plots_dir)
        
        # Analyze misinformation containment
        print("\nAnalyzing containment effectiveness...")
        analyze_misinformation_containment(results, plots_dir)
        
        # Analyze consensus patterns
        print("\nAnalyzing consensus patterns...")
        analyze_consensus_patterns(results, plots_dir)
        
        # Create comprehensive report
        report = {
            'experiment_summary': {
                'total_conditions': len(results),
                'diversity_levels': list(results['main_experiment'].keys()),
                'volatility_levels': list(results['main_experiment']['medium'].keys())
            },
            'key_findings': {
                condition: {
                    'avg_containment_rate': np.mean([
                        run['metrics']['misinformation_spread']
                        for div in results[condition]
                        for vol in results[condition][div]
                        for run in results[condition][div][vol]
                    ]),
                    'avg_coverage_diversity': np.mean([
                        run['metrics']['coverage_diversity']
                        for div in results[condition]
                        for vol in results[condition][div]
                        for run in results[condition][div][vol]
                    ])
                }
                for condition in results
            }
        }
        
        # Save report
        report_file = os.path.join(analysis_dir, 'analysis_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print("\nAnalysis completed successfully!")
        print("=" * 50)
        print(f"\nResults saved to: {analysis_dir}")
        print("Generated files:")
        print(f"- Analysis report: {report_file}")
        print(f"- Plots: {plots_dir}")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        raise
