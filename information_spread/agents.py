from typing import List, Tuple, Dict, Optional
import json
from anthropic import Anthropic
import openai
import networkx as nx
import numpy as np

class DefenderAgent:
    def __init__(self, 
                 agent_id: int,
                 role: str,
                 llm_type: str = 'gpt',
                 api_key: Optional[str] = None,
                 role_config: Optional[Dict] = None,
                 max_nodes_per_round: int = 3):
        self.agent_id = agent_id
        self.role = role
        self.llm_type = llm_type
        self.history = []
        self.role_config = role_config or {}
        self.max_nodes = max_nodes_per_round
        
        # Initialize LLM client
        if llm_type == 'claude':
            self.client = Anthropic(api_key=api_key)
        else:  # GPT
            openai.api_key = api_key
            
    def _get_default_role_instructions(self) -> str:
        """Get default role-specific instructions"""
        role_instructions = {
            'proactive_defender': """You are a proactive defender focused on preventing misinformation spread.
- Prioritize checking high-influence nodes before they get infected
- Focus on nodes that could become major spread points
- Pay special attention to network structure and node centrality
- Aim to create "firebreaks" in the network""",

            'reactive_defender': """You are a reactive defender focused on containing active misinformation.
- Prioritize fact-checking currently infected nodes
- Target nodes that are actively spreading misinformation
- Focus on reducing current spread rather than prevention
- Pay attention to recent infection patterns""",

            'network_analyzer': """You are a network analyzer focused on understanding spread patterns.
- Study network structure and identify critical nodes
- Track infection patterns and predict likely spread paths
- Balance between high-centrality and bridge nodes
- Share strategic insights with other defenders""",

            'rapid_responder': """You are a rapid responder focused on early containment.
- Quickly respond to new infections
- Prioritize nodes that were recently infected
- Focus on containing new outbreaks before they spread
- Monitor suspicious activity patterns"""
        }
        
        return role_instructions.get(self.role.lower(), 
                                  "Focus on fact-checking nodes to prevent misinformation spread.")
            
    def _generate_prompt(self,
                        env_description: str,
                        graph: nx.Graph,
                        other_messages: List[str],
                        consensus_type: str) -> str:
        """Generate prompt for LLM based on current state"""
        
        role_instruction = self.role_config.get('prompt_template', 
                                              self._get_default_role_instructions())
        
        # Add network structure information
        network_info = (
            f"Network has {len(graph)} nodes. "
            f"Average degree: {np.mean([d for n, d in graph.degree()]):.1f}. "
            f"You can check up to {self.max_nodes} nodes per round."
        )
        
        prompt = f"""You are Defender {self.agent_id}, a {self.role} in an information manipulation defense team.

Current situation:
{env_description}

Network information:
{network_info}

Other defender messages:
{chr(10).join(other_messages)}

Your role specific instructions:
{role_instruction}

Consensus type: {consensus_type}

Based on the current situation and your role, provide:
1. Your analysis of the situation and strategy
2. Your proposed nodes to fact-check (maximum {self.max_nodes} nodes) as a list of integers
3. A brief message to share with other defenders

Format your response as JSON exactly like this example:
{{
    "analysis": "My analysis of the situation...",
    "target_nodes": [1, 4, 7],
    "message": "My message to other defenders..."
}}

Remember:
- Choose nodes based on your role and strategy
- Consider network structure and infection patterns
- Coordinate with other defenders through messages
- Stay within the {self.max_nodes} nodes per round limit"""

        return prompt
    
    def _parse_llm_response(self, response: str) -> dict:
        """Extract structured response from LLM output"""
        try:
            # Clean up the response
            response = response.strip()
            
            # Find JSON block
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")
                
            json_str = response[start:end]
            
            # Try to parse JSON
            result = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['analysis', 'target_nodes', 'message']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
                    
            # Validate target_nodes format and limit
            if not isinstance(result['target_nodes'], list):
                raise ValueError("target_nodes must be a list")
                
            if len(result['target_nodes']) > self.max_nodes:
                result['target_nodes'] = result['target_nodes'][:self.max_nodes]
                
            return result
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Original response: {response}")
            
            # Return safe default
            return {
                "analysis": "Error parsing response",
                "target_nodes": [],
                "message": f"Error parsing response: {str(e)}"
            }
            
    def decide_action(self,
                     env_description: str,
                     graph: nx.Graph,
                     other_messages: List[str],
                     consensus_type: str = 'implicit') -> Tuple[List[int], str]:
        """Get next action and message from LLM"""
        
        prompt = self._generate_prompt(env_description, graph, 
                                     other_messages, consensus_type)
        
        try:
            if self.llm_type == 'claude':
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=256,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                result = self._parse_llm_response(response.content[0].text)
                
            else:  # GPT
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    max_tokens=256
                )
                result = self._parse_llm_response(response.choices[0].message.content)
                
            # Store in history
            self.history.append({
                'prompt': prompt,
                'response': result
            })
            
            return result['target_nodes'], result['message']
            
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return [], "Error occurred"