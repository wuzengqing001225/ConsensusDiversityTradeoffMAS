from typing import Dict, List, Optional
import numpy as np
import networkx as nx
from environment import InfoSpreadEnvironment
from agents import DefenderAgent
from runner import InfoSpreadExperimentRunner

class BaselineAgent(DefenderAgent):
    """Base class for baseline agents"""
    def __init__(self, agent_id: int, max_nodes_per_round: int = 3):
        super().__init__(agent_id, "baseline", max_nodes_per_round=max_nodes_per_round)
        
    def decide_action(self, env_description: str, graph: nx.Graph, 
                     other_messages: List[str], consensus_type: str = 'implicit'):
        """Should be implemented by specific baseline agents"""
        raise NotImplementedError

class RandomAgent(BaselineAgent):
    """Agent that randomly selects nodes to fact-check"""
    def decide_action(self, env_description: str, graph: nx.Graph, 
                     other_messages: List[str], consensus_type: str = 'implicit'):
        nodes = list(graph.nodes())
        selected = np.random.choice(nodes, 
                                  size=min(self.max_nodes, len(nodes)), 
                                  replace=False)
        return list(selected), "Random selection"

class HighDegreeAgent(BaselineAgent):
    """Agent that selects highest degree nodes"""
    def decide_action(self, env_description: str, graph: nx.Graph, 
                     other_messages: List[str], consensus_type: str = 'implicit'):
        degrees = dict(graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        selected = [node for node, _ in sorted_nodes[:self.max_nodes]]
        return selected, "Selected high-degree nodes"

class LocalClusterAgent(BaselineAgent):
    """Agent that focuses on local clusters"""
    def decide_action(self, env_description: str, graph: nx.Graph, 
                     other_messages: List[str], consensus_type: str = 'implicit'):
        clustering = nx.clustering(graph)
        sorted_nodes = sorted(clustering.items(), key=lambda x: x[1], reverse=True)
        selected = [node for node, _ in sorted_nodes[:self.max_nodes]]
        return selected, "Selected high-clustering nodes"

def run_baseline_experiments(api_keys: Dict[str, str],
                           logger: Optional[object] = None,
                           role_config: Optional[Dict] = None,
                           seeds: List[int] = None) -> Dict:
    """Run all baseline experiments"""
    
    if seeds is None:
        seeds = list(range(5))
    
    # Baseline configurations
    baseline_configs = {
        'single_llm': {
            'num_agents': 1,
            'consensus_type': 'implicit',
            'diversity_level': 'none'
        },
        'no_interaction': {
            'num_agents': 10,
            'consensus_type': 'none',
            'diversity_level': 'none'
        },
        'explicit_consensus': {
            'num_agents': 10,
            'consensus_type': 'explicit',
            'diversity_level': 'medium'
        },
        'random_strategy': {
            'agent_class': RandomAgent,
            'num_agents': 10
        },
        'high_degree': {
            'agent_class': HighDegreeAgent,
            'num_agents': 10
        },
        'local_cluster': {
            'agent_class': LocalClusterAgent,
            'num_agents': 10
        }
    }
    
    diversity_levels = ['low', 'medium', 'high']
    volatility_levels = ['low', 'moderate', 'high']
    
    results = {}
    
    # Run experiments for each baseline
    for baseline, config in baseline_configs.items():
        print(f"\nRunning {baseline} baseline...")
        baseline_results = {div: {vol: [] for vol in volatility_levels} 
                          for div in diversity_levels}
        
        for diversity in diversity_levels:
            for volatility in volatility_levels:
                print(f"  {diversity} diversity, {volatility} volatility")
                
                for seed in seeds:
                    print(f"    Seed {seed}")
                    np.random.seed(seed)
                    
                    # Create unique run ID
                    run_id = f"{baseline}_{diversity}_{volatility}_seed{seed}"
                    
                    # Special handling for non-LLM baselines
                    if 'agent_class' in config:
                        runner = InfoSpreadExperimentRunner(
                            num_agents=config['num_agents'],
                            consensus_type='implicit',
                            diversity_level='none',
                            volatility=volatility,
                            api_keys=api_keys,
                            role_config=role_config
                        )
                        runner.agents = [
                            config['agent_class'](i) 
                            for i in range(config['num_agents'])
                        ]
                    else:
                        # Regular LLM-based experiments
                        runner = InfoSpreadExperimentRunner(
                            num_agents=config['num_agents'],
                            consensus_type=config.get('consensus_type', 'implicit'),
                            diversity_level=config.get('diversity_level', diversity),
                            volatility=volatility,
                            api_keys=api_keys,
                            role_config=role_config
                        )
                    
                    result = runner.run_experiment()
                    baseline_results[diversity][volatility].append(result)
                    
                    # Log run if logger is provided
                    if logger:
                        logger.log_run(run_id, {
                            'baseline': baseline,
                            'diversity_level': diversity,
                            'volatility': volatility,
                            'seed': seed,
                            **config
                        }, result['round_logs'])
                        
        results[baseline] = baseline_results
        
    return results