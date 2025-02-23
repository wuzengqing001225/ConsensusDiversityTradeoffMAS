from typing import List, Dict
import numpy as np
from environment import PublicGoodsEnvironment

def calculate_provision_rate(env: PublicGoodsEnvironment) -> float:
    """Calculate rate of successful public good provision"""
    if not env.funded_history:
        return 0.0
    return sum(env.funded_history) / len(env.funded_history)

def calculate_total_welfare(env: PublicGoodsEnvironment) -> float:
    """Calculate total social welfare across all rounds"""
    total_welfare = 0
    
    for round_idx in range(env.time_step):
        round_contributions = [
            record for record in env.contributions
            if record.round == round_idx
        ]
        
        # Calculate total contribution for the round
        total_contribution = sum(record.amount for record in round_contributions)
        
        # Check if funded
        was_funded = env.funded_history[round_idx]
        
        if was_funded:
            # If funded, add benefit minus costs
            total_welfare += env.benefit - env.cost_factor * total_contribution
        else:
            # If not funded, subtract costs
            total_welfare -= env.cost_factor * total_contribution
            
    return total_welfare

def calculate_contribution_disparity(env: PublicGoodsEnvironment,
                                   round_idx: int = None) -> float:
    """Calculate Gini coefficient of contributions"""
    if round_idx is None:
        round_idx = env.time_step - 1
        
    round_contributions = [
        record for record in env.contributions
        if record.round == round_idx
    ]
    
    if not round_contributions:
        return 0.0
        
    # Calculate Gini coefficient
    amounts = np.array([record.amount for record in round_contributions])
    if np.sum(amounts) == 0:
        return 0.0
        
    n = len(amounts)
    diff_sum = sum(abs(x - y) for x in amounts for y in amounts)
    gini = diff_sum / (2 * n * np.sum(amounts))
    
    return gini

def calculate_adaptation_score(env: PublicGoodsEnvironment,
                             window_size: int = 3) -> float:
    """Calculate how well agents adapt to threshold changes"""
    if env.time_step < window_size:
        return 0.0
        
    adaptation_scores = []
    for t in range(window_size, env.time_step):
        # Get contributions before and after threshold change
        pre_contributions = [
            record for record in env.contributions
            if record.round == t - 1
        ]
        post_contributions = [
            record for record in env.contributions
            if record.round == t
        ]
        
        if not pre_contributions or not post_contributions:
            continue
            
        # Calculate adaptation score
        pre_total = sum(record.amount for record in pre_contributions)
        post_total = sum(record.amount for record in post_contributions)
        
        threshold_diff = env.threshold - pre_total
        contribution_change = post_total - pre_total
        
        if abs(threshold_diff) > 0:
            adaptation = min(1, max(0, 
                contribution_change / threshold_diff 
                if threshold_diff > 0 
                else -contribution_change / threshold_diff))
            adaptation_scores.append(adaptation)
            
    return np.mean(adaptation_scores) if adaptation_scores else 0.0

def evaluate_performance(env: PublicGoodsEnvironment) -> Dict[str, float]:
    """Calculate all performance metrics"""
    return {
        'provision_rate': calculate_provision_rate(env),
        'total_welfare': calculate_total_welfare(env),
        'contribution_disparity': calculate_contribution_disparity(env),
        'adaptation_score': calculate_adaptation_score(env)
    }