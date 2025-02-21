from typing import Dict, List, Optional
import numpy as np
from src.environment import DisasterEnvironment
from src.agents import LLMAgent
from src.consensus import ConsensusManager
from src.metrics import (calculate_coverage_rate, calculate_misallocation_penalty,
                        calculate_response_delay, calculate_performance_score)
from src.logger import ExperimentLogger

class ExperimentRunner:
    """Runs disaster response experiments"""
    
    def __init__(self,
                 env_size: int = 10,
                 num_agents: int = 10,
                 num_rounds: int = 20,
                 consensus_type: str = 'implicit',
                 diversity_level: str = 'medium',  # 'low', 'medium', 'high'
                 volatility: str = 'moderate',     # 'low', 'moderate', 'high'
                 api_keys: Optional[Dict[str, str]] = None,
                 logger: Optional[ExperimentLogger] = None,
                 role_config: Optional[Dict] = None):
        
        self.env = DisasterEnvironment(size=env_size)
        self.consensus_manager = ConsensusManager(consensus_type)
        self.num_rounds = num_rounds
        self.volatility = volatility
        self.logger = logger
        self.role_config = role_config or {}
        
        # Initialize agents based on diversity level
        self.agents = self.initialize_agents(num_agents, diversity_level, api_keys)
        
    def initialize_agents(self, 
                         num_agents: int, 
                         diversity_level: str,
                         api_keys: Optional[Dict[str, str]] = None) -> List[LLMAgent]:
        """Initialize agents with different roles based on diversity level"""
        
        if diversity_level == 'low':
            # All agents have same role
            roles = ['medical'] * num_agents
            
        elif diversity_level == 'medium':
            # Basic role distribution
            roles = ['medical', 'infrastructure', 'logistics'] * (num_agents // 3 + 1)
            roles = roles[:num_agents]
            
        else:  # high
            # More specialized roles with potential conflicts
            specialized_roles = [
                'medical_urgent',
                'medical_preventive',
                'infrastructure_power',
                'infrastructure_roads',
                'logistics_speed',
                'logistics_efficiency'
            ]
            roles = specialized_roles * (num_agents // len(specialized_roles) + 1)
            roles = roles[:num_agents]

        # Create agents with role-specific configs
        agents = []
        for i, role in enumerate(roles):
            role_specific_config = self.role_config.get(role, {})
            
            # Alternate between Claude and GPT for diversity
            # llm_type = 'claude' if i % 2 == 0 else 'gpt'
            llm_type = 'gpt'
            api_key = api_keys.get('anthropic' if llm_type == 'claude' else 'openai') if api_keys else None
            
            agent = LLMAgent(
                agent_id=i,
                role=role,
                llm_type=llm_type,
                api_key=api_key,
                role_config=role_specific_config
            )
            agents.append(agent)
            
        return agents

    def run_single_round(self, round_idx: int) -> Dict:
        """Run a single round of the experiment"""
        # Update environment
        self.env.step(self.volatility)
        env_state = self.env.get_observation()
        
        # Get agent proposals
        proposed_actions = []
        messages = []
        agent_interactions = {}
        
        for agent in self.agents:
            # Get agent decision
            prompt = agent._generate_prompt(
                env_state['description'],
                messages,
                self.consensus_manager.consensus_type
            )
            
            action, message = agent.decide_action(
                env_state['description'],
                messages,
                self.consensus_manager.consensus_type
            )
            
            proposed_actions.append(action)
            messages.append(message)
            
            # Log agent interaction
            agent_interactions[agent.agent_id] = {
                'role': agent.role,
                'prompt': prompt,
                'response': agent.history[-1]['response']
            }
            
        # Get final actions after consensus
        final_actions = self.consensus_manager.get_final_actions(
            proposed_actions, messages
        )
        
        # Calculate metrics
        metrics = {
            'coverage_rate': calculate_coverage_rate(self.env, final_actions),
            'misallocation_penalty': calculate_misallocation_penalty(self.env, final_actions),
            'response_delay': calculate_response_delay(self.env, final_actions),
            'mean_deviation': self.consensus_manager.calculate_consensus_metrics(final_actions)['mean_deviation']
        }
        
        # Return round data
        return {
            'round_idx': round_idx,
            'environment_state': env_state['description'],
            'agent_interactions': agent_interactions,
            'proposed_actions': proposed_actions,
            'final_actions': final_actions,
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
        for metric in ['coverage_rate', 'misallocation_penalty', 'response_delay', 'mean_deviation']:
            values = [log['metrics'][metric] for log in round_logs]
            aggregate_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            
        # Calculate overall performance score
        aggregate_metrics['performance_score'] = calculate_performance_score({
            metric: aggregate_metrics[metric]['mean']
            for metric in ['coverage_rate', 'misallocation_penalty', 'response_delay']
        })
        
        return {
            'round_logs': round_logs,
            'aggregate_metrics': aggregate_metrics
        }