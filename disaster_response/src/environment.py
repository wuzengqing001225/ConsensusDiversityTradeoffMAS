import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random

@dataclass
class Disaster:
    location: Tuple[int, int]  # Grid coordinates
    severity: float  # 1-10 scale
    type: str  # 'fire', 'medical', etc.
    
class DisasterEnvironment:
    def __init__(self, size: int = 10, max_disasters: int = 3):
        self.size = size
        self.max_disasters = max_disasters
        self.grid = np.zeros((size, size))
        self.disasters: List[Disaster] = []
        self.time_step = 0
        self.pnew = 0.2  # Probability of new disaster
        
    def add_disaster(self, location: Tuple[int, int], severity: float, type: str):
        if len(self.disasters) < self.max_disasters:
            disaster = Disaster(location=location, severity=severity, type=type)
            self.disasters.append(disaster)
            self.grid[location] = severity
            return True
        return False
        
    def update_severity(self, disaster_idx: int, delta: float):
        """Update severity of a specific disaster"""
        disaster = self.disasters[disaster_idx]
        new_severity = max(0, min(10, disaster.severity + delta))
        disaster.severity = new_severity
        self.grid[disaster.location] = new_severity
        
    def move_disaster(self, disaster_idx: int):
        """Move disaster to adjacent cell randomly"""
        disaster = self.disasters[disaster_idx]
        x, y = disaster.location
        moves = [(0,1), (0,-1), (1,0), (-1,0)]
        valid_moves = []
        
        for dx, dy in moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                valid_moves.append((new_x, new_y))
                
        if valid_moves:
            self.grid[disaster.location] = 0
            new_loc = random.choice(valid_moves)
            disaster.location = new_loc
            self.grid[new_loc] = disaster.severity
            
    def step(self, volatility: str = 'moderate'):
        """Advance environment one time step"""
        self.time_step += 1
        
        # Define volatility parameters
        volatility_params = {
            'low': {'move_prob': 0.3, 'severity_range': (-1, 1)},
            'moderate': {'move_prob': 0.5, 'severity_range': (-2, 2)},
            'high': {'move_prob': 0.8, 'severity_range': (-3, 3)}
        }
        
        params = volatility_params[volatility]
            
        # Update existing disasters
        for i in range(len(self.disasters)):
            # Maybe move disaster
            if random.random() < params['move_prob']:
                self.move_disaster(i)
                
            # Maybe change severity
            if random.random() < 0.5:
                delta = random.uniform(*params['severity_range'])
                self.update_severity(i, delta)
                
        # Maybe add new disaster
        if len(self.disasters) < self.max_disasters and random.random() < self.pnew:
            x = random.randint(0, self.size-1)
            y = random.randint(0, self.size-1)
            severity = random.uniform(5, 8)
            self.add_disaster((x,y), severity, random.choice(['fire', 'medical']))
            
    def get_state_description(self) -> str:
        """Generate textual description of current state"""
        description = []
        for i, disaster in enumerate(self.disasters):
            x, y = disaster.location
            desc = f"A {disaster.type} at Sector ({x},{y}) has severity {disaster.severity:.1f}."
            
            # Add some uncertainty/noise to reports
            if random.random() < 0.2:
                if random.random() < 0.5:
                    desc += " Some witnesses claim it's spreading."
                else:
                    desc += " Local reports suggest it's under control."
                    
            description.append(desc)
            
        return "\n".join(description)
        
    def get_observation(self) -> Dict:
        """Return current observation of environment"""
        return {
            'grid': self.grid.copy(),
            'disasters': self.disasters.copy(),
            'description': self.get_state_description()
        }

    def reset(self):
        """Reset environment to initial state"""
        self.grid = np.zeros((self.size, self.size))
        self.disasters = []
        self.time_step = 0