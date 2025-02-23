import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random

@dataclass
class ContributionRecord:
    """Record of an agent's contribution"""
    agent_id: int
    amount: float
    round: int
    cumulative: float

class PublicGoodsEnvironment:
    """Environment for dynamic public goods provision scenario"""
    def __init__(self, 
                 num_agents: int = 10,
                 initial_threshold: float = 30,
                 initial_benefit: float = 100,
                 cost_factor: float = 1,
                 max_contribution: float = 20):
        self.num_agents = num_agents
        self.threshold = initial_threshold
        self.benefit = initial_benefit
        self.cost_factor = cost_factor
        self.max_contribution = max_contribution
        
        self.time_step = 0
        self.contributions = []  # History of all contributions
        self.funded_history = []  # Track if public good was funded each round
        
        # Initialize contribution records for each agent
        self.agent_records = {i: [] for i in range(num_agents)}
        
    def contribute(self, agent_id: int, amount: float) -> Dict:
        """Process an agent's contribution"""
        # Validate contribution
        amount = max(0, min(amount, self.max_contribution))
        
        # Record contribution
        contribution = ContributionRecord(
            agent_id=agent_id,
            amount=amount,
            round=self.time_step,
            cumulative=sum(record.amount for record in self.agent_records[agent_id]) + amount
        )
        
        self.agent_records[agent_id].append(contribution)
        self.contributions.append(contribution)
        
        return {
            'amount': amount,
            'cumulative': contribution.cumulative
        }
        
    def process_round(self) -> Dict:
        """Process end of round and determine outcomes"""
        # Calculate total contribution
        total_contribution = sum(
            record.amount 
            for record in self.contributions
            if record.round == self.time_step
        )
        
        # Check if public good is funded
        is_funded = total_contribution >= self.threshold
        self.funded_history.append(is_funded)
        
        # Calculate payoffs
        individual_payoffs = {}
        for agent_id in range(self.num_agents):
            contribution = next(
                (record.amount for record in self.contributions
                 if record.round == self.time_step and record.agent_id == agent_id),
                0
            )
            
            payoff = (self.benefit / self.num_agents - self.cost_factor * contribution) \
                    if is_funded else (-self.cost_factor * contribution)
                    
            individual_payoffs[agent_id] = payoff
            
        return {
            'total_contribution': total_contribution,
            'threshold': self.threshold,
            'funded': is_funded,
            'benefit': self.benefit,
            'payoffs': individual_payoffs
        }
        
    def update_threshold(self, delta: float):
        """Update the funding threshold"""
        self.threshold = max(0, self.threshold + delta)
        
    def update_benefit(self, delta: float):
        """Update the public good benefit"""
        self.benefit = max(0, self.benefit + delta)

    def get_state_description(self) -> str:
        """Generate textual description of current state"""
        # Basic state information
        description = [
            f"Current round: {self.time_step}",
            f"Funding threshold: {self.threshold:.1f}",
            f"Public good benefit: {self.benefit:.1f}",
            f"Maximum individual contribution: {self.max_contribution}"
        ]
        
        # Recent history
        if self.time_step > 0 and self.funded_history:
            last_round = [
                record for record in self.contributions
                if record.round == self.time_step - 1
            ]
            if last_round:  # Only add if there were contributions
                total_last_round = sum(record.amount for record in last_round)
                was_funded = self.funded_history[-1]
                
                description.append(
                    f"\nLast round summary:"
                    f"\n- Total contribution: {total_last_round:.1f}"
                    f"\n- Threshold: {self.threshold:.1f}"
                    f"\n- Outcome: {'Funded' if was_funded else 'Not funded'}"
                )
                
                # Add contribution pattern info if there were contributions
                min_contrib = min(record.amount for record in last_round)
                max_contrib = max(record.amount for record in last_round)
                description.append(
                    f"- Contribution range: {min_contrib:.1f} to {max_contrib:.1f}"
                )
        else:
            description.append("\nFirst round - no previous contributions.")

        # Add some uncertainty/rumors with moderate probability
        if random.random() < 0.2:
            if random.random() < 0.5:
                description.append(
                    "\nRumor: The threshold might increase next round due to new regulations."
                )
            else:
                description.append(
                    "\nAnalysts suggest the benefit might be overestimated."
                )
            
        return "\n".join(description)

    def get_observation(self) -> Dict:
        """Return current observation of environment"""
        return {
            'time_step': self.time_step,
            'threshold': self.threshold,
            'benefit': self.benefit,
            'description': self.get_state_description(),
            'history': {
                'contributions': self.contributions,
                'funded': self.funded_history
            }
        }
        
    def step(self, volatility: str = 'moderate'):
        """Advance environment one time step"""
        self.time_step += 1
        
        # Define volatility parameters
        volatility_params = {
            'low': {
                'threshold_change': (-5, 5),
                'benefit_change': (-10, 10),
                'change_prob': 0.1
            },
            'moderate': {
                'threshold_change': (-10, 10),
                'benefit_change': (-20, 20),
                'change_prob': 0.2
            },
            'high': {
                'threshold_change': (-15, 15),
                'benefit_change': (-30, 30),
                'change_prob': 0.3
            }
        }
        
        params = volatility_params[volatility]
        
        # Possibly update threshold
        if random.random() < params['change_prob']:
            delta = random.uniform(*params['threshold_change'])
            self.update_threshold(delta)
            
        # Possibly update benefit
        if random.random() < params['change_prob']:
            delta = random.uniform(*params['benefit_change'])
            self.update_benefit(delta)
            
    def reset(self):
        """Reset environment to initial state"""
        self.time_step = 0
        self.contributions = []
        self.funded_history = []
        self.agent_records = {i: [] for i in range(self.num_agents)}