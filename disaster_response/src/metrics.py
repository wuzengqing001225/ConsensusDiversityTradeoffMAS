from typing import List, Tuple, Dict
import numpy as np
from src.environment import DisasterEnvironment

def calculate_coverage_rate(env: DisasterEnvironment, 
                          agent_actions: List[Tuple[int, int]]) -> float:
    """Calculate fraction of disasters contained within 2 rounds"""
    total_disasters = len(env.disasters)
    if total_disasters == 0:
        return 1.0
        
    covered = 0
    for disaster in env.disasters:
        if any(abs(disaster.location[0] - x) + abs(disaster.location[1] - y) <= 2 
               for x,y in agent_actions):
            covered += 1
            
    return covered / total_disasters

def calculate_misallocation_penalty(env: DisasterEnvironment, 
                                  agent_actions: List[Tuple[int, int]]) -> float:
    """Calculate penalty for multiple agents at same location while others uncovered"""
    action_counts = {}
    for action in agent_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
        
    penalty = 0
    for loc, count in action_counts.items():
        if count > 2:  # More than 2 agents at same location
            uncovered_disasters = sum(1 for d in env.disasters 
                                    if not any(abs(d.location[0] - x) + abs(d.location[1] - y) <= 2 
                                             for x,y in agent_actions))
            if uncovered_disasters > 0:
                penalty += 5 * (count - 2)  # Penalty increases with excess agents
            
    return penalty

def calculate_response_delay(env: DisasterEnvironment, 
                           agent_actions: List[Tuple[int, int]]) -> float:
    """Calculate average response delay to disasters"""
    if not env.disasters:
        return 0
        
    delays = []
    for disaster in env.disasters:
        min_delay = min(abs(disaster.location[0] - x) + abs(disaster.location[1] - y) 
                       for x,y in agent_actions)
        # Weight delay by disaster severity
        delays.append(min_delay * disaster.severity / 10)
        
    return sum(delays) / len(delays)

def calculate_performance_score(metrics: Dict[str, float]) -> float:
    """Calculate overall performance score from individual metrics"""
    coverage_weight = 1.0
    misallocation_weight = -0.1
    delay_weight = -0.1
    
    score = (coverage_weight * metrics['coverage_rate'] +
             misallocation_weight * metrics['misallocation_penalty'] +
             delay_weight * metrics['response_delay'])
             
    return score

def calculate_deviation_metrics(actions: List[Tuple[int, int]]) -> Dict[str, float]:
    """Calculate metrics about agent deviation from group behavior"""
    if not actions:
        return {
            'mean_deviation': 0.0,
            'max_deviation': 0.0,
            'deviation_std': 0.0
        }
        
    actions_array = np.array(actions)
    mean_action = np.mean(actions_array, axis=0)
    
    # Calculate Euclidean distances from mean
    deviations = np.sqrt(np.sum((actions_array - mean_action) ** 2, axis=1))
    
    return {
        'mean_deviation': float(np.mean(deviations)),
        'max_deviation': float(np.max(deviations)),
        'deviation_std': float(np.std(deviations))
    }