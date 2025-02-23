import os
import json
import argparse
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_contribution_patterns(results: Dict, save_dir: str):
    """Analyze and visualize contribution patterns"""
    plt.figure(figsize=(15, 5))
    
    # 1. Individual contribution patterns
    contribution_data = []
    for condition, condition_results in results.items():
        for diversity in condition_results:
            for volatility in condition_results[diversity]:
                for run in condition_results[diversity][volatility]:
                    for round_data in run['round_logs']:
                        for i, contrib in enumerate(round_data['final_contributions']):
                            contribution_data.append({
                                'condition': condition,
                                'diversity': diversity,
                                'volatility': volatility,
                                'round': round_data['round_idx'],
                                'agent_id': i,
                                'contribution': contrib,
                                'threshold': round_data['round_results']['threshold']
                            })
    
    df = pd.DataFrame(contribution_data)
    
    # Plot contribution distribution
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, x='condition', y='contribution')
    plt.title('Contribution Distribution by Condition')
    plt.xticks(rotation=45)
    
    # Plot contribution vs threshold
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df[df['condition'] == 'main_experiment'],
                   x='threshold', y='contribution', hue='diversity')
    plt.title('Contribution vs Threshold')
    
    # Plot time evolution
    plt.subplot(1, 3, 3)
    time_evolution = df[df['condition'] == 'main_experiment'].groupby(
        ['round', 'diversity'])['contribution'].mean().reset_index()
    sns.lineplot(data=time_evolution, x='round', y='contribution', hue='diversity')
    plt.title('Contribution Evolution Over Time')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'contribution_patterns.png'))
    plt.close()

def analyze_free_riding(results: Dict, save_dir: str):
    """Analyze free-riding behavior"""
    plt.figure(figsize=(15, 5))
    
    # Calculate free-riding metrics
    freerider_data = []
    for condition, condition_results in results.items():
        for diversity in condition_results:
            for volatility in condition_results[diversity]:
                for run in condition_results[diversity][volatility]:
                    for round_data in run['round_logs']:
                        threshold = round_data['round_results']['threshold']
                        fair_share = threshold / len(round_data['final_contributions'])
                        
                        for contrib in round_data['final_contributions']:
                            freerider_score = max(0, 1 - contrib / fair_share)
                            freerider_data.append({
                                'condition': condition,
                                'diversity': diversity,
                                'volatility': volatility,
                                'freerider_score': freerider_score,
                                'funded': round_data['round_results']['funded']
                            })
    
    df = pd.DataFrame(freerider_data)
    
    # Plot free-riding analysis
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, x='condition', y='freerider_score')
    plt.title('Free-Riding Score by Condition')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df[df['condition'] == 'main_experiment'],
                x='diversity', y='freerider_score', hue='volatility')
    plt.title('Free-Riding Score by Diversity')
    
    plt.subplot(1, 3, 3)
    success_rate = df.groupby(['condition', 'freerider_score'])['funded'].mean().reset_index()
    sns.lineplot(data=success_rate, x='freerider_score', y='funded', hue='condition')
    plt.title('Success Rate vs Free-Riding')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'free_riding_analysis.png'))
    plt.close()

def analyze_adaptation(results: Dict, save_dir: str):
    """Analyze adaptation to changes"""
    plt.figure(figsize=(15, 5))
    
    # Calculate adaptation metrics
    adaptation_data = []
    for condition, condition_results in results.items():
        for diversity in condition_results:
            for volatility in condition_results[diversity]:
                for run in condition_results[diversity][volatility]:
                    for i in range(1, len(run['round_logs'])):
                        prev_round = run['round_logs'][i-1]
                        curr_round = run['round_logs'][i]
                        
                        # Calculate threshold change
                        thresh_change = (curr_round['round_results']['threshold'] - 
                                       prev_round['round_results']['threshold'])
                        
                        # Calculate contribution change
                        contrib_change = (sum(curr_round['final_contributions']) - 
                                        sum(prev_round['final_contributions']))
                        
                        # Calculate adaptation score
                        if abs(thresh_change) > 0:
                            adaptation_score = min(1, max(0, contrib_change / thresh_change 
                                                        if thresh_change > 0 
                                                        else -contrib_change / thresh_change))
                        else:
                            adaptation_score = 1
                            
                        adaptation_data.append({
                            'condition': condition,
                            'diversity': diversity,
                            'volatility': volatility,
                            'round': i,
                            'adaptation_score': adaptation_score,
                            'threshold_change': abs(thresh_change)
                        })
    
    df = pd.DataFrame(adaptation_data)
    
    # Plot adaptation analysis
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, x='condition', y='adaptation_score')
    plt.title('Adaptation Score by Condition')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df[df['condition'] == 'main_experiment'],
                x='diversity', y='adaptation_score', hue='volatility')
    plt.title('Adaptation Score by Diversity')
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x='threshold_change', y='adaptation_score',
                   hue='condition', alpha=0.5)
    plt.title('Adaptation vs Threshold Change')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'adaptation_analysis.png'))
    plt.close()

def analyze_consensus_formation(results: Dict, save_dir: str):
    """Analyze consensus formation patterns"""
    plt.figure(figsize=(15, 5))
    
    # Extract consensus metrics
    consensus_data = []
    for condition, condition_results in results.items():
        for diversity in condition_results:
            for volatility in condition_results[diversity]:
                for run in condition_results[diversity][volatility]:
                    for round_data in run['round_logs']:
                        consensus_data.append({
                            'condition': condition,
                            'diversity': diversity,
                            'volatility': volatility,
                            'round': round_data['round_idx'],
                            'mean_deviation': round_data['metrics']['mean_deviation'],
                            'consensus_ratio': round_data['metrics']['consensus_ratio'],
                            'threshold_alignment': round_data['metrics']['threshold_alignment']
                        })
    
    df = pd.DataFrame(consensus_data)
    
    # Plot consensus analysis
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
    sns.scatterplot(data=df, x='mean_deviation', y='threshold_alignment',
                   hue='condition', style='volatility')
    plt.title('Consensus vs Performance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'consensus_analysis.png'))
    plt.close()

def create_summary_report(results: Dict, save_dir: str):
    """Create comprehensive analysis report"""
    # Calculate key metrics
    summary = {
        'overall_metrics': {},
        'diversity_impact': {},
        'volatility_impact': {},
        'consensus_effectiveness': {}
    }
    
    # Calculate overall metrics
    for condition in results:
        metrics = []
        for diversity in results[condition]:
            for volatility in results[condition][diversity]:
                for run in results[condition][diversity][volatility]:
                    run_metrics = {
                        'provision_rate': np.mean([
                            round_data['round_results']['funded']
                            for round_data in run['round_logs']
                        ]),
                        'total_welfare': sum([
                            sum(round_data['round_results']['payoffs'].values())
                            for round_data in run['round_logs']
                        ]),
                        'avg_contribution': np.mean([
                            np.mean(round_data['final_contributions'])
                            for round_data in run['round_logs']
                        ])
                    }
                    metrics.append(run_metrics)
                    
        summary['overall_metrics'][condition] = {
            metric: {
                'mean': np.mean([m[metric] for m in metrics]),
                'std': np.std([m[metric] for m in metrics])
            }
            for metric in metrics[0].keys()
        }
    
    # Save summary report
    with open(os.path.join(save_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
        
    return summary

def analyze_experiment(exp_dir: str):
    """Analyze experiment results"""
    print(f"\nAnalyzing public goods experiment from: {exp_dir}")
    print("=" * 50)
    
    # Create analysis directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(exp_dir, f"analysis_{timestamp}")
    plots_dir = os.path.join(analysis_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Load results
        results_file = os.path.join(exp_dir, 'experiment_results.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        # Run all analyses
        print("\nAnalyzing contribution patterns...")
        analyze_contribution_patterns(results, plots_dir)
        
        print("Analyzing free-riding behavior...")
        analyze_free_riding(results, plots_dir)
        
        print("Analyzing adaptation capabilities...")
        analyze_adaptation(results, plots_dir)
        
        print("Analyzing consensus formation...")
        analyze_consensus_formation(results, plots_dir)
        
        print("Creating summary report...")
        summary = create_summary_report(results, analysis_dir)
        
        print("\nAnalysis completed successfully!")
        print("=" * 50)
        print(f"\nResults saved to: {analysis_dir}")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise
