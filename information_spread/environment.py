import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import random

@dataclass
class InfoNode:
    """Represents a node in the information network"""
    id: int
    state: str  # 'unaware', 'informed', 'misinformed'
    fact_checked: bool = False
    infection_time: int = -1
    last_update: int = -1

class InfoSpreadEnvironment:
    """Environment for information spread and manipulation scenario"""
    def __init__(self, 
                 num_nodes: int = 50,
                 pspread: float = 0.2,
                 num_initial_infected: int = 3):
        # Initialize network
        self.size = num_nodes
        self.pspread = pspread
        self.time_step = 0
        
        # Create scale-free network
        self.graph = nx.barabasi_albert_graph(n=num_nodes, m=3)
        
        # Initialize node states
        self.nodes = {}
        for i in range(num_nodes):
            self.nodes[i] = InfoNode(id=i, state='unaware')
            
        # Track outbreak statistics
        self.current_outbreak = {
            'start_time': 0,
            'initial_nodes': set(),
            'spread_nodes': set()
        }
        
        # Set initial infected nodes
        self._initialize_infection(num_initial_infected)
        
    def _initialize_infection(self, num_infected: int):
        """Initialize infection in network"""
        # Choose high-degree nodes as initial infected
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        initial_infected = [node for node, _ in sorted_nodes[:num_infected]]
        
        for node_id in initial_infected:
            self.nodes[node_id].state = 'misinformed'
            self.nodes[node_id].infection_time = 0
            
        # Initialize outbreak tracking
        self.current_outbreak = {
            'start_time': 0,
            'initial_nodes': set(initial_infected),
            'spread_nodes': set(initial_infected)
        }
        
    def spread_misinformation(self):
        """Update misinformation spread in network"""
        new_infections = set()
        
        # Check each misinformed node
        for node_id, node in self.nodes.items():
            if node.state == 'misinformed':
                # Try to infect neighbors
                for neighbor in self.graph.neighbors(node_id):
                    if (self.nodes[neighbor].state == 'unaware' and 
                        not self.nodes[neighbor].fact_checked):
                        if random.random() < self.pspread:
                            new_infections.add(neighbor)
        
        # Update newly infected nodes
        for node_id in new_infections:
            self.nodes[node_id].state = 'misinformed'
            self.nodes[node_id].infection_time = self.time_step
            self.current_outbreak['spread_nodes'].add(node_id)
            
        return len(new_infections)
    
    def inject_misinformation(self, num_nodes: int = 2):
        """Adversary injects new misinformation"""
        potential_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.state == 'unaware' and not node.fact_checked
        ]
        
        if not potential_nodes:
            return []
            
        # Choose nodes based on degree centrality
        degrees = dict(self.graph.degree())
        potential_nodes.sort(key=lambda x: degrees[x], reverse=True)
        
        inject_nodes = potential_nodes[:num_nodes]
        for node_id in inject_nodes:
            self.nodes[node_id].state = 'misinformed'
            self.nodes[node_id].infection_time = self.time_step
            
            # Update outbreak tracking
            if not self.current_outbreak['spread_nodes']:  # If this is a new outbreak
                self.current_outbreak = {
                    'start_time': self.time_step,
                    'initial_nodes': set(inject_nodes),
                    'spread_nodes': set(inject_nodes)
                }
            else:  # Add to existing outbreak
                self.current_outbreak['spread_nodes'].add(node_id)
            
        return inject_nodes
    
    def fact_check_nodes(self, target_nodes: List[int]) -> Dict:
        """Fact check specified nodes"""
        results = {
            'corrected': [],
            'protected': [],
            'already_checked': []
        }
        
        for node_id in target_nodes:
            if node_id not in self.nodes:
                continue
                
            node = self.nodes[node_id]
            
            if node.fact_checked:
                results['already_checked'].append(node_id)
                continue
                
            # Mark as fact checked
            node.fact_checked = True
            node.last_update = self.time_step
            
            # If node was misinformed, correct it
            if node.state == 'misinformed':
                node.state = 'informed'
                results['corrected'].append(node_id)
            else:
                results['protected'].append(node_id)
                
        return results
    
    def get_state_description(self) -> str:
        """Generate textual description of current state"""
        misinformed = sum(1 for node in self.nodes.values() 
                         if node.state == 'misinformed')
        infected_ids = [nid for nid, node in self.nodes.items() 
                       if node.state == 'misinformed']
        fact_checked = sum(1 for node in self.nodes.values() 
                          if node.fact_checked)
        
        description = [
            f"Current time step: {self.time_step}",
            f"Total misinformed nodes: {misinformed} of {self.size}",
            f"Total fact-checked nodes: {fact_checked}",
        ]
        
        # Add information about recent infections
        new_infections = [nid for nid in infected_ids 
                         if self.nodes[nid].infection_time == self.time_step]
        if new_infections:
            description.append(f"New infections at nodes: {new_infections}")
            
        # Add suspicious activity alerts
        high_degree_infected = [
            nid for nid in infected_ids 
            if self.graph.degree(nid) > np.mean([self.graph.degree(n) for n in self.graph.nodes()])
        ]
        if high_degree_infected:
            description.append(
                f"Warning: High influence nodes {high_degree_infected} are spreading misinformation"
            )
            
        # Sometimes add misleading information
        if random.random() < 0.2:
            if random.random() < 0.5:
                fake_clear = random.choice(infected_ids) if infected_ids else random.randint(0, self.size-1)
                description.append(
                    f"Unconfirmed report: Node {fake_clear} may have been cleared of misinformation"
                )
            else:
                potential_fake = random.choice([
                    nid for nid in self.nodes if nid not in infected_ids
                ])
                description.append(
                    f"Rumors suggest Node {potential_fake} might be compromised"
                )
        
        return "\n".join(description)
    
    def get_observation(self) -> Dict:
        """Return current observation of environment"""
        return {
            'nodes': {nid: {
                'state': node.state,
                'fact_checked': node.fact_checked,
                'infection_time': node.infection_time,
                'last_update': node.last_update
            } for nid, node in self.nodes.items()},
            'graph': self.graph,
            'description': self.get_state_description(),
            'time_step': self.time_step,
            'outbreak': self.current_outbreak.copy()
        }
    
    def step(self, volatility: str = 'moderate'):
        """Advance environment one time step"""
        self.time_step += 1
        
        # Define volatility parameters
        volatility_params = {
            'low': {'spread_prob': 0.1, 'inject_prob': 0.1, 'max_inject': 1},
            'moderate': {'spread_prob': 0.2, 'inject_prob': 0.2, 'max_inject': 2},
            'high': {'spread_prob': 0.3, 'inject_prob': 0.3, 'max_inject': 3}
        }
        
        params = volatility_params[volatility]
        
        # Update spread probability
        self.pspread = params['spread_prob']
        
        # Spread existing misinformation
        spread_count = self.spread_misinformation()
        
        # Possibly inject new misinformation
        if random.random() < params['inject_prob']:
            self.inject_misinformation(random.randint(1, params['max_inject']))
            
        return spread_count
    
    def reset(self):
        """Reset environment to initial state"""
        # Reset node states
        for node in self.nodes.values():
            node.state = 'unaware'
            node.fact_checked = False
            node.infection_time = -1
            node.last_update = -1
            
        self.time_step = 0
        
        # Reset outbreak tracking
        self.current_outbreak = {
            'start_time': 0,
            'initial_nodes': set(),
            'spread_nodes': set()
        }
        
        # Initialize new infection
        self._initialize_infection(3)