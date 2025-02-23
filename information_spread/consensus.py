from typing import List, Dict
import numpy as np
import networkx as nx

class ConsensusManager:
    """Manages consensus formation among defender agents"""
    
    def __init__(self, consensus_type: str = 'implicit'):
        self.consensus_type = consensus_type
        
    def get_final_actions(self, 
                         proposed_actions: List[List[int]], 
                         messages: List[str],
                         graph: nx.Graph = None) -> List[List[int]]:
        """Determine final actions based on consensus type"""
        if self.consensus_type == 'explicit':
            return self._get_explicit_consensus(proposed_actions, graph)
        else:  # implicit
            return proposed_actions
            
    def _get_explicit_consensus(self, 
                              proposed_actions: List[List[int]], 
                              graph: nx.Graph) -> List[List[int]]:
        """Generate explicit consensus through voting and network analysis"""
        if not proposed_actions:
            return []
            
        # Count votes for each node
        node_votes = {}
        for actions in proposed_actions:
            for node in actions:
                node_votes[node] = node_votes.get(node, 0) + 1
                
        if not node_votes:
            return [[] for _ in proposed_actions]
            
        # Sort nodes by votes and degree centrality
        if graph is not None:
            degrees = dict(graph.degree())
            node_scores = {
                node: votes + 0.1 * degrees.get(node, 0)
                for node, votes in node_votes.items()
            }
        else:
            node_scores = node_votes
            
        sorted_nodes = sorted(node_scores.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
        
        # Take top nodes based on max nodes per agent
        max_nodes = max(len(actions) for actions in proposed_actions)
        consensus_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
        
        # Return same consensus for all agents
        return [consensus_nodes for _ in proposed_actions]
    
    def calculate_consensus_metrics(self, 
                                  actions: List[List[int]],
                                  graph: nx.Graph = None) -> Dict[str, float]:
        """Calculate metrics about the degree of consensus"""
        if not actions:
            return {
                'mean_deviation': 0.0,
                'consensus_ratio': 1.0,
                'coverage_overlap': 0.0,
                'network_diversity': 0.0
            }
            
        # Calculate pairwise Jaccard similarity between action sets
        similarities = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                set_i = set(actions[i])
                set_j = set(actions[j])
                if set_i or set_j:  # Avoid division by zero
                    similarity = len(set_i & set_j) / len(set_i | set_j)
                    similarities.append(similarity)
                    
        mean_similarity = np.mean(similarities) if similarities else 0.0
        
        # Calculate coverage overlap
        all_nodes = set()
        for action_set in actions:
            all_nodes.update(action_set)
        coverage_overlap = 1 - len(all_nodes) / (sum(len(a) for a in actions) + 1e-10)
        
        # Calculate network diversity if graph is provided
        network_diversity = 0.0
        if graph is not None and all_nodes:
            # Calculate average shortest path length between selected nodes
            selected_subgraph = graph.subgraph(all_nodes)
            try:
                avg_path_length = nx.average_shortest_path_length(selected_subgraph)
                network_diversity = avg_path_length / nx.diameter(graph)
            except (nx.NetworkXError, ZeroDivisionError):
                network_diversity = 0.0
        
        return {
            'mean_deviation': 1 - mean_similarity,
            'consensus_ratio': mean_similarity,
            'coverage_overlap': coverage_overlap,
            'network_diversity': network_diversity
        }