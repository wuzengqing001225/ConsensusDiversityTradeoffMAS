from typing import List, Tuple, Dict
import numpy as np

class ConsensusManager:
    """Manages consensus formation among agents"""
    
    def __init__(self, consensus_type: str = 'implicit'):
        self.consensus_type = consensus_type
        
    def get_final_actions(self, 
                         proposed_actions: List[Tuple[int, int]], 
                         messages: List[str]) -> List[Tuple[int, int]]:
        """Determine final actions based on consensus type"""
        if self.consensus_type == 'explicit':
            # Use majority voting
            action_counts = {}
            for action in proposed_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            majority_action = max(action_counts.items(), key=lambda x: x[1])[0]
            return [majority_action] * len(proposed_actions)
            
        else:  # implicit
            # Each agent keeps their proposed action
            return proposed_actions
            
    def calculate_consensus_metrics(self, 
                                  actions: List[Tuple[int, int]]) -> Dict[str, float]:
        """Calculate metrics about the degree of consensus"""
        if not actions:
            return {
                'mean_deviation': 0.0,
                'consensus_ratio': 1.0,
                'unique_actions': 0
            }
            
        # Convert to numpy array for easier calculation
        actions_array = np.array(actions)
        mean_action = np.mean(actions_array, axis=0)
        
        # Calculate Euclidean distances from mean
        deviations = np.sqrt(np.sum((actions_array - mean_action) ** 2, axis=1))
        
        # Count unique actions
        unique_actions = len(set(actions))
        
        # Calculate ratio of agents choosing most common action
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        max_count = max(action_counts.values())
        consensus_ratio = max_count / len(actions)
        
        return {
            'mean_deviation': float(np.mean(deviations)),
            'consensus_ratio': consensus_ratio,
            'unique_actions': unique_actions
        }