from typing import List, Dict
import numpy as np

class ConsensusManager:
    """Manages consensus formation among contributor agents"""
    
    def __init__(self, consensus_type: str = 'implicit'):
        self.consensus_type = consensus_type
        
    def get_final_actions(self, 
                         proposed_contributions: List[float], 
                         messages: List[str],
                         threshold: float = None) -> List[float]:
        """Determine final contributions based on consensus type"""
        if self.consensus_type == 'explicit':
            return self._get_explicit_consensus(proposed_contributions, threshold)
        else:  # implicit
            return proposed_contributions
            
    def _get_explicit_consensus(self, 
                              proposed_contributions: List[float],
                              threshold: float = None) -> List[float]:
        """Generate explicit consensus through group decision"""
        if not proposed_contributions:
            return []
            
        if threshold is None:
            # If no threshold provided, use average contribution
            consensus_amount = np.mean(proposed_contributions)
        else:
            # Try to meet threshold with equal contributions
            fair_share = threshold / len(proposed_contributions)
            consensus_amount = min(fair_share, max(proposed_contributions))
            
        # Return same consensus contribution for all agents
        return [consensus_amount] * len(proposed_contributions)
    
    def calculate_consensus_metrics(self, 
                                  contributions: List[float],
                                  threshold: float = None) -> Dict[str, float]:
        """Calculate metrics about the degree of consensus"""
        if not contributions:
            return {
                'mean_deviation': 0.0,
                'consensus_ratio': 1.0,
                'contribution_variance': 0.0,
                'threshold_alignment': 0.0
            }
            
        # Calculate basic statistics
        mean_contrib = np.mean(contributions)
        deviations = [abs(c - mean_contrib) for c in contributions]
        
        # Calculate consensus metrics
        metrics = {
            'mean_deviation': np.mean(deviations),
            'consensus_ratio': 1 - (np.std(contributions) / (mean_contrib + 1e-10)),
            'contribution_variance': np.var(contributions)
        }
        
        # Calculate threshold alignment if threshold is provided
        if threshold is not None:
            total_contribution = sum(contributions)
            metrics['threshold_alignment'] = min(1.0, total_contribution / threshold)
            
        return metrics