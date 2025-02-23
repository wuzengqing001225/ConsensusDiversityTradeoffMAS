from typing import Dict, List, Optional
import numpy as np
from environment import PublicGoodsEnvironment
from agents import ContributorAgent
from consensus import ConsensusManager
from metrics import evaluate_performance

class PublicGoodsExperimentRunner:
    """Runs public goods provision experiments"""
    
    def __init__(self,
                 num_agents: int = 10,
                 num_rounds: int = 30,
                 consensus_type: str = 'implicit',
                 diversity_level: str = 'medium',  # 'low', 'medium', 'high'
                 volatility: str = 'moderate',     # 'low', 'moderate', 'high'
                 initial_threshold: float = 30,
                 initial_benefit: float = 100,
                 api_keys: Optional[Dict[str, str]] = None,
                 role_config: Optional[Dict] = None):
                 
        self.env = PublicGoodsEnvironment(
            num_agents=num_agents,
            initial_threshold=initial_threshold,
            initial_benefit=initial_benefit
        )
        self.consensus_manager = ConsensusManager(consensus_type)
        self.num_rounds = num_rounds
        self.volatility = volatility
        
        # Initialize agents based on diversity level
        self.agents = self.initialize_agents(
            num_agents=num_agents,
            diversity_level=diversity_level,
            api_keys=api_keys,
            role_config=role_config
        )
        
    def initialize_agents(self, 
                         num_agents: int,
                         diversity_level: str,
                         api_keys: Optional[Dict[str, str]] = None,
                         role_config: Optional[Dict] = None) -> List[ContributorAgent]:
        """Initialize agents with different roles based on diversity level"""
        if diversity_level == 'low':
            # All agents have same role (strategic)
            roles = ['strategic'] * num_agents
            
        elif diversity_level == 'medium':
            # Basic role distribution
            roles = [
                'altruistic',
                'strategic',
                'conservative'
            ] * (num_agents // 3 + 1)
            roles = roles[:num_agents]
            
        else:  # high
            # More specialized roles with potential conflicts
            roles = [
                'altruistic',
                'strategic',
                'conservative',
                'adaptive'
            ] * (num_agents // 4 + 1)
            roles = roles[:num_agents]
            
        # Create agents
        agents = []
        for i, role in enumerate(roles):
            llm_type = 'gpt'
            api_key = api_keys.get('anthropic' if llm_type == 'claude' else 'openai') if api_keys else None
            
            agent = ContributorAgent(
                agent_id=i,
                role=role,
                llm_type=llm_type,
                api_key=api_key,
                role_config=role_config.get(role) if role_config else None,
                max_contribution=self.env.max_contribution
            )
            agents.append(agent)
            
        return agents
    
    def run_single_round(self, round_idx: int) -> Dict:
        """Run a single round of the experiment"""
        # Update environment
        env_state = self.env.get_observation()
        self.env.step(self.volatility)
        
        # Get agent proposals
        proposed_contributions = []
        messages = []
        agent_interactions = {}
        
        for agent in self.agents:
            # Get agent decision
            prompt = agent._generate_prompt(
                env_state['description'],
                messages,
                self.consensus_manager.consensus_type
            )
            
            contribution, message = agent.decide_action(
                env_state['description'],
                messages,
                self.consensus_manager.consensus_type
            )
            
            proposed_contributions.append(contribution)
            messages.append(message)
            
            # Log agent interaction
            agent_interactions[agent.agent_id] = {
                'role': agent.role,
                'prompt': prompt,
                'response': agent.history[-1]['response']
            }
            
        # Get final contributions after consensus
        final_contributions = self.consensus_manager.get_final_actions(
            proposed_contributions,
            messages,
            env_state['threshold']
        )
        
        # Process contributions and get round results
        for agent_idx, contribution in enumerate(final_contributions):
            self.env.contribute(agent_idx, contribution)
            
        round_results = self.env.process_round()
        
        # Calculate metrics
        metrics = evaluate_performance(self.env)
        consensus_metrics = self.consensus_manager.calculate_consensus_metrics(
            final_contributions,
            env_state['threshold']
        )
        metrics.update(consensus_metrics)
        
        # Return round data
        return {
            'round_idx': round_idx,
            'environment_state': env_state['description'],
            'agent_interactions': agent_interactions,
            'proposed_contributions': proposed_contributions,
            'final_contributions': final_contributions,
            'round_results': round_results,
            'metrics': metrics
        }
        
    def run_experiment(self) -> Dict:
        """Run full experiment and return metrics"""
        self.env.reset()
        round_logs = []
        
        for round_idx in range(self.num_rounds):
            round_data = self.run_single_round(round_idx)
            round_logs.append(round_data)
            
        # Calculate aggregate metrics
        aggregate_metrics = {}
        for metric in round_logs[0]['metrics'].keys():
            values = [log['metrics'][metric] for log in round_logs]
            aggregate_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            
        return {
            'round_logs': round_logs,
            'aggregate_metrics': aggregate_metrics
        }