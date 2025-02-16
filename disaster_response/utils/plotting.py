import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os

def plot_metrics_time_series(metrics_df: pd.DataFrame,
                           save_dir: str,
                           exp_condition: str = ''):
    """Plot time series of all metrics"""
    metrics = ['coverage_rate', 'misallocation_penalty', 'response_delay', 'mean_deviation']
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.lineplot(data=metrics_df, x='round', y=metric)
        plt.title(f'{metric.replace("_", " ").title()} over Time')
        plt.xlabel('Round')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'metrics_time_series_{exp_condition}.png'))
    plt.close()

def plot_performance_by_factors(results: Dict, save_dir: str):
    """Plot performance by diversity and volatility separately"""
    # Prepare data for both plots
    all_data = []
    
    for condition, condition_results in results.items():
        for diversity in condition_results:
            for volatility in condition_results[diversity]:
                runs = condition_results[diversity][volatility]
                for run in runs:
                    # Calculate average performance for each run
                    run_perfs = []
                    for round_data in run['round_logs']:
                        metrics = round_data['metrics']
                        perf = (metrics['coverage_rate'] - 
                               0.1 * metrics['misallocation_penalty'] - 
                               0.1 * metrics['response_delay'])
                        # Normalize performance to [0,1] range
                        perf = (perf + 1) / 2
                        run_perfs.append(perf)
                        
                    mean_perf = np.mean(run_perfs)
                    all_data.append({
                        'diversity_level': diversity,
                        'volatility_level': volatility,
                        'condition': condition,
                        'performance': mean_perf
                    })
    
    if not all_data:
        print("Warning: No data available for plotting")
        return
        
    df = pd.DataFrame(all_data)
    main_exp_data = df[df['condition'] == 'main_experiment']
    
    if main_exp_data.empty:
        print("Warning: No main experiment data available")
        return
    
    # 1. Plot by Diversity Level
    plt.figure(figsize=(10, 6))
    diversity_plot = main_exp_data.groupby('diversity_level')['performance'].agg(['mean', 'std']).reset_index()
    
    if not diversity_plot.empty:
        sns.barplot(data=diversity_plot, 
                    x='diversity_level', 
                    y='mean',
                    errorbar=('ci', 95))
        
        plt.title('Performance by Diversity Level')
        plt.xlabel('Diversity Level')
        plt.ylabel('Normalized Performance')
        
        # Add value labels
        ax = plt.gca()
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_by_diversity.png'))
    plt.close()
    
    # 2. Plot by Volatility Level
    plt.figure(figsize=(10, 6))
    volatility_plot = main_exp_data.groupby('volatility_level')['performance'].agg(['mean', 'std']).reset_index()
    
    if not volatility_plot.empty:
        sns.barplot(data=volatility_plot, 
                    x='volatility_level', 
                    y='mean',
                    errorbar=('ci', 95))
        
        plt.title('Performance by Volatility Level')
        plt.xlabel('Volatility Level')
        plt.ylabel('Normalized Performance')
        
        # Add value labels
        ax = plt.gca()
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_by_volatility.png'))
    plt.close()
    
    # 3. Create heatmap of diversity vs volatility
    plt.figure(figsize=(10, 6))
    heatmap_data = main_exp_data.pivot_table(
        values='performance',
        index='diversity_level',
        columns='volatility_level',
        aggfunc='mean'
    )
    
    if not heatmap_data.empty and heatmap_data.size > 0:
        sns.heatmap(heatmap_data, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Performance'})
        
        plt.title('Performance Heatmap: Diversity vs Volatility')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_heatmap.png'))
    plt.close()
    
    # 4. Print data summary
    print("\nData Summary:")
    print("\nDiversity Levels:", sorted(df['diversity_level'].unique()))
    print("Volatility Levels:", sorted(df['volatility_level'].unique()))
    print("Conditions:", sorted(df['condition'].unique()))
    print("\nNumber of data points:", len(df))
    print("Number of main experiment data points:", len(main_exp_data))

def plot_deviation_performance_relationship(deviation_data: pd.DataFrame,
                                          save_dir: str,
                                          exp_condition: str = ''):
    """Plot relationship between agent deviation and system performance with binning"""
    plt.figure(figsize=(10, 6))
    
    # Create bins for deviation values
    bins = np.linspace(0, deviation_data['deviation'].max(), 10)
    deviation_data['deviation_bin'] = pd.cut(deviation_data['deviation'], bins)
    
    # Calculate statistics for each bin
    bin_stats = deviation_data.groupby('deviation_bin').agg({
        'performance': ['mean', 'std', 'count']
    }).reset_index()
    bin_stats.columns = ['deviation_bin', 'mean_perf', 'std_perf', 'count']
    
    # Only plot bins with enough data points
    bin_stats = bin_stats[bin_stats['count'] >= 5]
    
    # Plot mean performance for each bin
    bin_centers = bin_stats['deviation_bin'].apply(lambda x: x.mid)
    plt.errorbar(bin_centers, bin_stats['mean_perf'],
                yerr=bin_stats['std_perf'],
                fmt='o-', capsize=5, markersize=8)
    
    # Add trend line
    z = np.polyfit(bin_centers, bin_stats['mean_perf'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(bin_centers.min(), bin_centers.max(), 100)
    plt.plot(x_trend, p(x_trend), 'r--', alpha=0.8)
    
    # Calculate correlation for the binned data
    corr = np.corrcoef(bin_centers, bin_stats['mean_perf'])[0,1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}',
             transform=plt.gca().transAxes)
    
    # Add sample size for each bin
    for i, (x, y, count) in enumerate(zip(bin_centers, 
                                        bin_stats['mean_perf'],
                                        bin_stats['count'])):
        plt.text(x, y + 0.02, f'n={count}',
                horizontalalignment='center', verticalalignment='bottom')
    
    plt.xlabel('Mean Agent Deviation (d)')
    plt.ylabel('System Performance (P)')
    plt.title('Deviation-Performance Relationship (Binned)')
    
    plt.savefig(os.path.join(save_dir, f'deviation_performance_{exp_condition}.png'))
    plt.close()

def plot_comparative_analysis(results: Dict,
                            save_dir: str,
                            metrics: List[str] = None):
    """Generate comparative plots for different metrics"""
    if metrics is None:
        metrics = ['coverage_rate', 'misallocation_penalty', 'response_delay']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        plot_data = []
        
        for condition, condition_results in results.items():
            for diversity in condition_results:
                for volatility in condition_results[diversity]:
                    # Calculate mean metric value for each run
                    for run in condition_results[diversity][volatility]:
                        run_values = [round_data['metrics'][metric] 
                                    for round_data in run['round_logs']]
                        mean_value = np.mean(run_values)
                        
                        plot_data.append({
                            'condition': condition,
                            'diversity_level': diversity,
                            'volatility_level': volatility,
                            'value': mean_value
                        })
        
        df = pd.DataFrame(plot_data)
        
        # Create grouped bar plot
        sns.barplot(data=df, 
                   x='condition', 
                   y='value',
                   hue='volatility_level',
                   errorbar=('ci', 95))
        
        plt.title(f'{metric.replace("_", " ").title()} by Condition and Volatility')
        plt.xlabel('Condition')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        # Add value labels
        ax = plt.gca()
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparative_{metric}.png'))
        plt.close()

def create_summary_dashboard(results: Dict, save_dir: str):
    """Create a comprehensive dashboard of key metrics"""
    plt.figure(figsize=(20, 15))
    
    # 1. Overall performance by condition
    plt.subplot(2, 2, 1)
    plot_data = []
    for condition, condition_results in results.items():
        performances = []
        for diversity in condition_results:
            for volatility in condition_results[diversity]:
                for run in condition_results[diversity][volatility]:
                    run_perfs = []
                    for round_data in run['round_logs']:
                        metrics = round_data['metrics']
                        perf = (metrics['coverage_rate'] - 
                               0.1 * metrics['misallocation_penalty'] - 
                               0.1 * metrics['response_delay'])
                        perf = (perf + 1) / 2  # Normalize to [0,1]
                        run_perfs.append(perf)
                    performances.append(np.mean(run_perfs))
        
        if performances:
            plot_data.append({
                'Condition': condition,
                'Performance': np.mean(performances),
                'Std': np.std(performances)
            })
    
    df = pd.DataFrame(plot_data)
    sns.barplot(data=df, x='Condition', y='Performance', errorbar='sd')
    plt.title('Overall Performance by Condition')
    plt.xticks(rotation=45)
    
    # Add other dashboard components as needed
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_dashboard.png'))
    plt.close()