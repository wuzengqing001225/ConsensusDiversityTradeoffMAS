import numpy as np
import json
import time
import random
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import openai
import anthropic
from enum import Enum

OPENAI_API_KEY = ''
Anthropic_API_KEY = ''

class ConsensusMode(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"

class DiversityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class VolatilityLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

@dataclass
class Disaster:
    x: int
    y: int
    severity: float
    rounds_active: int = 0
    
    def update_severity(self, change: float):
        self.severity = max(0, min(10, self.severity + change))
    
    def move(self, grid_size: int):
        """Random movement to neighboring cell"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dx, dy = random.choice(directions)
        self.x = max(0, min(grid_size - 1, self.x + dx))
        self.y = max(0, min(grid_size - 1, self.y + dy))

@dataclass
class AgentAction:
    agent_id: int
    x: int
    y: int
    confidence: float = 0.0

@dataclass
class RoundResult:
    round_num: int
    actions: List[AgentAction]
    disasters: List[Disaster]
    coverage_rate: float
    misallocation_penalty: float
    response_delay: float
    mean_deviation: float

class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        pass

class OpenAIInterface(LLMInterface):
    def __init__(self, model_name: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
    
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=256
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        return json.dumps({
            "analysis": "Unable to analyze due to API error",
            "action": [random.randint(0, 9), random.randint(0, 9)],
            "message": "Moving to random location due to system error"
        })

class AnthropicInterface(LLMInterface):
    def __init__(self, model_name: str = "claude-opus-4-20250514"):
        self.client = anthropic.Anthropic(api_key=Anthropic_API_KEY)
        self.model_name = model_name
    
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=256,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        return json.dumps({
            "analysis": "Unable to analyze due to API error",
            "action": [random.randint(0, 9), random.randint(0, 9)],
            "message": "Moving to random location due to system error"
        })

class DisasterAgent:
    ROLE_PROMPTS = {
        DiversityLevel.LOW: {
            "all": "You are a disaster response drone. Always address the highest severity zone. Focus on the most urgent disaster."
        },
        DiversityLevel.MEDIUM: {
            "medical": "Focus on rescuing casualties in highest-severity disaster zones for people.",
            "infrastructure": "Protect power lines and roads. Even if severity is high elsewhere, prioritize built structures.",
            "logistics": "Minimize travel cost. Quickly move to nearest active zone if severity is above 5."
        },
        DiversityLevel.HIGH: {
            "medical": "Focus exclusively on casualty evacuation. Ignore infrastructure damage.",
            "infrastructure": "Protect critical infrastructure at all costs. Casualties are secondary.",
            "logistics": "Optimize for minimum resource expenditure. Avoid high-cost operations.",
            "scout": "Gather intelligence on potential new disasters. Prevention over reaction.",
            "command": "Coordinate overall strategy. Balance all concerns but prioritize systematic coverage."
        }
    }
    
    def __init__(self, agent_id: int, llm_interface: LLMInterface, role: str, diversity_level: DiversityLevel):
        self.agent_id = agent_id
        self.llm = llm_interface
        self.role = role
        self.diversity_level = diversity_level
        self.message_history = []
        
    def get_role_prompt(self) -> str:
        if self.diversity_level == DiversityLevel.LOW:
            return self.ROLE_PROMPTS[self.diversity_level]["all"]
        else:
            return self.ROLE_PROMPTS[self.diversity_level].get(self.role, 
                self.ROLE_PROMPTS[DiversityLevel.MEDIUM]["medical"])
    
    def generate_action(self, disasters: List[Disaster], other_messages: List[str], 
                       round_num: int, mode: ConsensusMode) -> Tuple[AgentAction, str]:
        
        # Create situation description
        disaster_info = []
        for i, d in enumerate(disasters):
            disaster_info.append(f"Disaster {i}: location ({d.x},{d.y}), severity {d.severity:.1f}, active for {d.rounds_active} rounds")
        
        situation = f"Round {round_num}: {len(disasters)} active disasters.\n" + "\n".join(disaster_info)
        
        if random.random() < 0.2:
            fake_disaster = random.choice(disasters) if disasters else None
            if fake_disaster:
                if random.random() < 0.5:
                    situation += f"\nUnconfirmed report: Disaster at ({fake_disaster.x},{fake_disaster.y}) may be under control."
                else:
                    situation += f"\nWitness report: New casualties spotted near ({fake_disaster.x + 1},{fake_disaster.y + 1})."
        
        # Format other agents' messages
        others_msgs = "\n".join(other_messages) if other_messages else "No messages from other drones yet."
        
        prompt = f"""You are Drone {self.agent_id}, a {self.role} in a disaster response team.

Current situation: {situation}

Other drone messages: {others_msgs}

Your role instructions: {self.get_role_prompt()}

Based on the current situation and your role, provide:
1. Your analysis of the situation
2. Your proposed action as grid coordinates [x,y] (coordinates must be 0-9)
3. A brief message to share with other drones

Format your response as JSON exactly like this example:
{{"analysis": "My analysis of the situation...", "action": [3,4], "message": "My message to other drones..."}}

Do not include any other text in your response (e.g., '```json' is not allowed)."""

        response = self.llm.generate_response(prompt, temperature=0.7)
        print(response)
        
        try:
            parsed = json.loads(response.strip())
            action_coords = parsed.get("action", [random.randint(0, 9), random.randint(0, 9)])
            
            # Ensure coordinates are within bounds
            x = max(0, min(9, int(action_coords[0])))
            y = max(0, min(9, int(action_coords[1])))
            
            action = AgentAction(agent_id=self.agent_id, x=x, y=y)
            message = parsed.get("message", f"Moving to ({x},{y})")
            
            self.message_history.append(message)
            return action, message
            
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            print(f"Failed to parse agent {self.agent_id} response: {e}")
            # Fallback to random action
            x, y = random.randint(0, 9), random.randint(0, 9)
            action = AgentAction(agent_id=self.agent_id, x=x, y=y)
            message = f"Moving to ({x},{y}) due to parsing error"
            return action, message

class DisasterEnvironment:
    def __init__(self, grid_size: int = 10, max_disasters: int = 3):
        self.grid_size = grid_size
        self.max_disasters = max_disasters
        self.disasters: List[Disaster] = []
        self.round_num = 0
        
    def initialize_disasters(self):
        num_disasters = random.randint(1, min(3, self.max_disasters))
        for _ in range(num_disasters):
            x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            severity = random.uniform(3.0, 8.0)
            self.disasters.append(Disaster(x=x, y=y, severity=severity))
    
    def update_environment(self, volatility: VolatilityLevel):
        self.round_num += 1
        
        if volatility == VolatilityLevel.LOW:
            move_prob, severity_change, new_disaster_prob = 0.1, 1.0, 0.1
            move_freq, severity_freq = 3, 1
        elif volatility == VolatilityLevel.MODERATE:
            move_prob, severity_change, new_disaster_prob = 0.3, 2.0, 0.2
            move_freq, severity_freq = 2, 1
        else:
            move_prob, severity_change, new_disaster_prob = 0.5, 3.0, 0.3
            move_freq, severity_freq = 1, 1
        
        for disaster in self.disasters[:]:
            disaster.rounds_active += 1
            
            if self.round_num % move_freq == 0 and random.random() < move_prob:
                disaster.move(self.grid_size)
            
            if self.round_num % severity_freq == 0:
                change = random.uniform(-severity_change, severity_change)
                disaster.update_severity(change)
            
            if disaster.severity <= 0:
                self.disasters.remove(disaster)
        
        if (len(self.disasters) < self.max_disasters and 
            random.random() < new_disaster_prob):
            x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            severity = random.uniform(4.0, 9.0)
            self.disasters.append(Disaster(x=x, y=y, severity=severity))

class SimulationManager:
    def __init__(self, num_agents: int = 30, grid_size: int = 10, rounds: int = 20):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.rounds = rounds
        self.agents: List[DisasterAgent] = []
        self.environment = DisasterEnvironment(grid_size)
        self.results: List[RoundResult] = []
        
    def setup_agents(self, diversity_level: DiversityLevel, model_mix: str = "gpt4_claude"):
        self.agents = []
        
        if diversity_level == DiversityLevel.LOW:
            roles = ["all"] * self.num_agents
        elif diversity_level == DiversityLevel.MEDIUM:
            roles = ["medical", "infrastructure", "logistics"] * (self.num_agents // 3 + 1)
            roles = roles[:self.num_agents]
        else:
            base_roles = ["medical", "infrastructure", "logistics", "scout", "command"]
            roles = (base_roles * (self.num_agents // len(base_roles) + 1))[:self.num_agents]
        
        for i in range(self.num_agents):
            if model_mix == "gpt4_claude":
                if i % 2 == 0:
                    llm = OpenAIInterface("gpt-4o")
                else:
                    llm = AnthropicInterface("claude-opus-4-20250514")
            elif model_mix == "gpt4_only":
                llm = OpenAIInterface("gpt-4o")
            elif model_mix == "claude_only":
                llm = AnthropicInterface("claude-opus-4-20250514")
            else:
                llm = OpenAIInterface("gpt-4o-mini")  # fallback
            
            agent = DisasterAgent(i, llm, roles[i], diversity_level)
            self.agents.append(agent)
    
    def calculate_metrics(self, actions: List[AgentAction], disasters: List[Disaster]) -> Tuple[float, float, float, float]:
        if not disasters:
            return 1.0, 0.0, 0.0, 0.0
        
        covered_disasters = 0
        agent_positions = [(a.x, a.y) for a in actions]
        
        for disaster in disasters:
            if (disaster.x, disaster.y) in agent_positions:
                covered_disasters += 1
        
        coverage_rate = covered_disasters / len(disasters) if disasters else 1.0
        
        disaster_positions = [(d.x, d.y) for d in disasters]
        misallocated = 0
        
        for pos in agent_positions:
            if pos not in disaster_positions:
                misallocated += 1
        
        position_counts = {}
        for pos in agent_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        for pos, count in position_counts.items():
            if pos in disaster_positions and count > 2:
                misallocated += (count - 2)
        
        misallocation_penalty = misallocated / self.num_agents if self.num_agents > 0 else 0
        
        high_severity_disasters = [d for d in disasters if d.severity > 7.0]
        unattended_high_severity = 0
        
        for disaster in high_severity_disasters:
            if (disaster.x, disaster.y) not in agent_positions:
                unattended_high_severity += 1
        
        response_delay = unattended_high_severity / len(high_severity_disasters) if high_severity_disasters else 0
        
        if not actions:
            return coverage_rate, misallocation_penalty, response_delay, 0.0
        
        mean_x = sum(a.x for a in actions) / len(actions)
        mean_y = sum(a.y for a in actions) / len(actions)
        
        total_deviation = sum(abs(a.x - mean_x) + abs(a.y - mean_y) for a in actions)
        mean_deviation = total_deviation / len(actions)
        
        return coverage_rate, misallocation_penalty, response_delay, mean_deviation
    
    def run_explicit_consensus_round(self, round_num: int) -> Tuple[List[AgentAction], List[str]]:
        """Run one round with explicit consensus (majority voting)"""
        messages = []
        proposed_actions = []
        
        for agent in self.agents:
            action, message = agent.generate_action(
                self.environment.disasters, messages, round_num, ConsensusMode.EXPLICIT
            )
            proposed_actions.append(action)
            messages.append(f"Drone {agent.agent_id}: {message}")
        
        action_votes = {}
        for action in proposed_actions:
            coord = (action.x, action.y)
            action_votes[coord] = action_votes.get(coord, 0) + 1
        
        if action_votes:
            winning_coord = max(action_votes.keys(), key=lambda k: action_votes[k])
            final_actions = [AgentAction(i, winning_coord[0], winning_coord[1]) 
                           for i in range(self.num_agents)]
        else:
            final_actions = proposed_actions
        
        return final_actions, messages
    
    def run_implicit_consensus_round(self, round_num: int) -> Tuple[List[AgentAction], List[str]]:
        """Run one round with implicit consensus (individual decisions after discussion)"""
        messages = []
        
        initial_messages = []
        for agent in self.agents:
            _, message = agent.generate_action(
                self.environment.disasters, initial_messages, round_num, ConsensusMode.IMPLICIT
            )
            initial_messages.append(f"Drone {agent.agent_id}: {message}")
        
        final_actions = []
        for agent in self.agents:
            action, message = agent.generate_action(
                self.environment.disasters, initial_messages, round_num, ConsensusMode.IMPLICIT
            )
            final_actions.append(action)
            messages.append(f"Drone {agent.agent_id}: {message}")
        
        return final_actions, messages
    
    def run_simulation(self, mode: ConsensusMode, diversity: DiversityLevel, 
                      volatility: VolatilityLevel, model_mix: str = "gpt4_claude") -> List[RoundResult]:
        """Run complete simulation"""
        print(f"Starting simulation: {mode.value} consensus, {diversity.value} diversity, {volatility.value} volatility")
        
        self.setup_agents(diversity, model_mix)
        self.environment.initialize_disasters()
        self.results = []
        
        for round_num in range(self.rounds):
            print(f"Round {round_num + 1}/{self.rounds}")
            
            self.environment.update_environment(volatility)
            
            if mode == ConsensusMode.EXPLICIT:
                actions, messages = self.run_explicit_consensus_round(round_num)
            else:
                actions, messages = self.run_implicit_consensus_round(round_num)
            
            coverage_rate, misalloc_penalty, response_delay, mean_deviation = \
                self.calculate_metrics(actions, self.environment.disasters)
            
            result = RoundResult(
                round_num=round_num,
                actions=actions,
                disasters=self.environment.disasters.copy(),
                coverage_rate=coverage_rate,
                misallocation_penalty=misalloc_penalty,
                response_delay=response_delay,
                mean_deviation=mean_deviation
            )
            self.results.append(result)
            
            time.sleep(0.5)
        
        return self.results

def run_experimental_sweep():
    diversity_levels = [DiversityLevel.LOW, DiversityLevel.MEDIUM, DiversityLevel.HIGH]
    volatility_levels = [VolatilityLevel.LOW, VolatilityLevel.MODERATE, VolatilityLevel.HIGH]
    consensus_modes = [ConsensusMode.EXPLICIT, ConsensusMode.IMPLICIT]

    all_results = []
    
    for diversity in diversity_levels:
        for volatility in volatility_levels:
            for mode in consensus_modes:
                print(f"\n{'='*50}")
                print(f"Running: {mode.value} - {diversity.value} - {volatility.value}")
                
                for seed in range(10):
                    random.seed(seed)
                    np.random.seed(seed)
                    
                    sim = SimulationManager(num_agents=100, rounds=20)
                    results = sim.run_simulation(mode, diversity, volatility, "gpt4_only")
                    
                    avg_coverage = np.mean([r.coverage_rate for r in results])
                    avg_misalloc = np.mean([r.misallocation_penalty for r in results])
                    avg_delay = np.mean([r.response_delay for r in results])
                    avg_deviation = np.mean([r.mean_deviation for r in results])
                    
                    all_results.append({
                        'mode': mode.value,
                        'diversity': diversity.value,
                        'volatility': volatility.value,
                        'seed': seed,
                        'coverage_rate': avg_coverage,
                        'misallocation_penalty': avg_misalloc,
                        'response_delay': avg_delay,
                        'mean_deviation': avg_deviation
                    })
    
    return pd.DataFrame(all_results)

def plot_inverted_u_relationship(df: pd.DataFrame):    
    implicit_df = df[df['mode'] == 'implicit'].copy()
    
    deviation_bins = pd.cut(implicit_df['mean_deviation'], bins=10, include_lowest=True)
    binned_data = implicit_df.groupby(deviation_bins).agg({
        'coverage_rate': 'mean',
        'mean_deviation': 'mean'
    }).reset_index()
    
    binned_data = binned_data.dropna()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(binned_data['mean_deviation'], binned_data['coverage_rate'], 
               alpha=0.7, s=100, c='blue', label='Implicit Consensus')
    
    if len(binned_data) > 3:
        z = np.polyfit(binned_data['mean_deviation'], binned_data['coverage_rate'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(binned_data['mean_deviation'].min(), 
                            binned_data['mean_deviation'].max(), 100)
        plt.plot(x_trend, p(x_trend), 'r--', alpha=0.8, label='Trend Line')
    
    plt.xlabel('Mean Deviation')
    plt.ylabel('Coverage Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return binned_data

def create_performance_comparison_plot(df: pd.DataFrame):
    """Create performance comparison plots (Figures 3 & 4)"""
    
    grouped = df.groupby(['mode', 'diversity', 'volatility']).agg({
        'coverage_rate': 'mean',
        'misallocation_penalty': 'mean',
        'response_delay': 'mean'
    }).reset_index()
    
    overall = df.groupby('mode').agg({
        'coverage_rate': 'mean',
        'misallocation_penalty': 'mean',
        'response_delay': 'mean'
    })
    
    print("\nOverall Performance Comparison:")
    print(overall.round(3))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    modes = ['explicit', 'implicit']
    coverage_means = [overall.loc[mode, 'coverage_rate'] for mode in modes]
    
    axes[0].bar(modes, coverage_means, color=['red', 'blue'], alpha=0.7)
    axes[0].set_ylabel('Coverage Rate')
    axes[0].set_title('Performance Comparison between\nImplicit and Explicit Consensus')
    axes[0].set_ylim(0, 1.0)
    
    diversity_comparison = df.groupby(['mode', 'diversity'])['coverage_rate'].mean().unstack()
    diversity_comparison.plot(kind='bar', ax=axes[1], color=['lightcoral', 'lightblue', 'lightgreen'])
    axes[1].set_ylabel('Coverage Rate')
    axes[1].set_title('Performance by Diversity Level')
    axes[1].set_xlabel('Consensus Mode')
    axes[1].legend(title='Diversity Level')
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    return grouped

if __name__ == "__main__":
    print("Dynamic Disaster Response Simulation")
    print("Reproducing results from 'The Hidden Strength of Disagreement'")
    print("\nNote: Set up your API keys before running:")
    
    print("\nRunning quick test...")
    sim = SimulationManager(num_agents=100, rounds=20)
    results = sim.run_simulation(ConsensusMode.EXPLICIT, DiversityLevel.MEDIUM, VolatilityLevel.MODERATE, "gpt4_only")
    print(f"Test completed. Coverage rates: {[r.coverage_rate for r in results]}")
    
    print("\nRunning full experimental sweep...")
    df = run_experimental_sweep()
    print("\nResults summary:")
    print(df.groupby(['mode', 'diversity', 'volatility'])['coverage_rate'].mean().round(3))
