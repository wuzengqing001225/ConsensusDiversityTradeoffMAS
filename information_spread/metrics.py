from typing import List, Dict, Set
import numpy as np
from environment import InfoSpreadEnvironment

def calculate_misinformation_spread(env: InfoSpreadEnvironment) -> float:
    """Calculate current spread of misinformation"""
    total_misinformed = sum(1 for node in env.nodes.values() 
                           if node.state == 'misinformed')
    return total_misinformed / env.size

def calculate_containment_time(env: InfoSpreadEnvironment,
                             outbreak_start: Set[int],
                             current_spread: Set[int]) -> float:
    """Calculate average time to contain outbreak"""
    if not outbreak_start:
        return 0
        
    # Calculate how many of the initially infected nodes are now contained
    contained = outbreak_start - current_spread
    containment_ratio = len(contained) / len(outbreak_start)
    
    # Return containment time weighted by containment ratio
    return env.time_step * (1 - containment_ratio)

def calculate_coverage_diversity(env: InfoSpreadEnvironment,
                               agent_actions: List[List[int]]) -> float:
    """Calculate diversity of node coverage"""
    # Get unique nodes checked across all agents
    all_checked = set()
    for actions in agent_actions:
        all_checked.update(actions)
    
    # Calculate coverage ratio
    return len(all_checked) / env.size

def calculate_influence_score(env: InfoSpreadEnvironment,
                            fact_checked_nodes: Set[int]) -> float:
    """Calculate influence-weighted coverage score"""
    if not fact_checked_nodes:
        return 0
    
    # Calculate degree centrality for each node
    degrees = dict(env.graph.degree())
    max_degree = max(degrees.values())
    
    # Calculate normalized influence score
    influence_score = sum(degrees[node] / max_degree 
                         for node in fact_checked_nodes)
    
    return influence_score / env.size

def evaluate_performance(env: InfoSpreadEnvironment,
                        agent_actions: List[List[int]]) -> Dict[str, float]:
    """Calculate all performance metrics"""
    current_spread = {nid for nid, node in env.nodes.items() 
                     if node.state == 'misinformed'}
    fact_checked = {nid for nid, node in env.nodes.items() 
                    if node.fact_checked}
    
    return {
        'misinformation_spread': calculate_misinformation_spread(env),
        'containment_time': calculate_containment_time(
            env, 
            env.current_outbreak['initial_nodes'],
            current_spread
        ),
        'coverage_diversity': calculate_coverage_diversity(env, agent_actions),
        'influence_score': calculate_influence_score(env, fact_checked)
    }