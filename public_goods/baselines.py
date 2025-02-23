from typing import Dict, List, Optional
import numpy as np
from agents import ContributorAgent
from runner import PublicGoodsExperimentRunner

class BaselineAgent(ContributorAgent):
    """Base class for baseline agents"""
    def __init__(self, agent_id: int, max_contribution: float = 20):
        super().__init__(agent_id, "baseline", max_contribution=max_contribution)
        
    def decide_action(self, env_description: str, other_messages: List[str], 
                     consensus_type: str = 'implicit'):
        """Should be implemented by specific baseline agents"""
        raise NotImplementedError

class FairShareAgent(BaselineAgent):
    """Agent that always contributes its fair share of the threshold"""
    def decide_action(self, env_description: str, other_messages: List[str], 
                     consensus_type: str = 'implicit'):
        # Parse threshold from description
        try:
            threshold_line = [line for line in env_description.split('\n') 
                            if 'threshold' in line.lower()][0]
            threshold = float(threshold_line.split(':')[1].strip().split()[0])
            fair_share = min(self.max_contribution, threshold / 5)  # Assuming 5 agents
        except:
            fair_share = self.max_contribution / 2
            
        return fair_share, "Contributing fair share"

class RandomAgent(BaselineAgent):
    """Agent that randomly contributes between 0 and max"""
    def decide_action(self, env_description: str, other_messages: List[str], 
                     consensus_type: str = 'implicit'):
        contribution = np.random.uniform(0, self.max_contribution)
        return contribution, "Random contribution"

class FreeriderAgent(BaselineAgent):
    """Agent that tries to free ride with occasional small contributions"""
    def decide_action(self, env_description: str, other_messages: List[str], 
                     consensus_type: str = 'implicit'):
        if np.random.random() < 0.2:  # 20% chance of contributing
            contribution = np.random.uniform(0, self.max_contribution / 4)
        else:
            contribution = 0
        return contribution, "Minimal contribution"

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
        'fair_share': {
            'agent_class': FairShareAgent,
            'num_agents': 10
        },
        'random': {
            'agent_class': RandomAgent,
            'num_agents': 10
        },
        'freerider': {
            'agent_class': FreeriderAgent,
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
                    
                    if 'agent_class' in config:
                        # Special handling for non-LLM baselines
                        runner = PublicGoodsExperimentRunner(
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
                        runner = PublicGoodsExperimentRunner(
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