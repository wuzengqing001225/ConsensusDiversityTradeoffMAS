from typing import List, Tuple, Optional, Dict
import json
from anthropic import Anthropic
import openai

class LLMAgent:
    def __init__(self, 
                 agent_id: int,
                 role: str,
                 llm_type: str = 'claude',
                 api_key: Optional[str] = None,
                 role_config: Optional[Dict] = None):
        self.agent_id = agent_id
        self.role = role
        self.llm_type = llm_type
        self.history = []
        self.role_config = role_config or {}
        
        # Initialize LLM client
        if llm_type == 'claude':
            self.client = Anthropic(api_key=api_key)
        elif llm_type == 'gpt':
            openai.api_key = api_key
            
    def _generate_prompt(self, 
                        env_description: str,
                        other_messages: List[str],
                        consensus_type: str) -> str:
        """Generate prompt for LLM based on current state"""
        
        role_instruction = self.role_config.get('prompt_template', 
                                              self._get_default_role_instructions())
        
        prompt = f"""You are Drone {self.agent_id}, a {self.role} in a disaster response team.

Current situation:
{env_description}

Other drone messages:
{chr(10).join(other_messages)}

Your role specific instructions:
{role_instruction}

Consensus type: {consensus_type}

Based on the current situation and your role, provide:
1. Your analysis of the situation
2. Your proposed action as grid coordinates in format [x,y] (where x and y are integers from 0-9)
3. A brief message to share with other drones

Format your response as JSON exactly like this example:
{{
    "analysis": "My analysis of the situation...",
    "action": [3,4],
    "message": "My message to other drones..."
}}

Remember:
- Keep coordinates within 0-9 range
- Use square brackets for action coordinates
- Make action a pair of integers
- Consider other drones' perspectives in your analysis and think step by step"""

        return prompt
        
    def _get_default_role_instructions(self) -> str:
        """Get default role-specific instructions"""
        role_instructions = {
            'medical': "Focus on rescuing casualties in highest-severity disaster zones for people.",
            'infrastructure': "Protect power lines and roads. Even if severity is high elsewhere, prioritize built structures.",
            'logistics': "Minimize travel cost. Quickly move to nearest active zone if severity is above 5."
        }
        return role_instructions.get(self.role.lower(), "Respond to disasters based on severity and location.")
            
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
            required_fields = ['analysis', 'action', 'message']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
                    
            # Validate action format
            action = result['action']
            if not isinstance(action, (list, tuple)) or len(action) != 2:
                raise ValueError(f"Invalid action format: {action}")
                
            return result
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Original response: {response}")
            
            # Return safe default
            return {
                "analysis": "Error parsing response",
                "action": [0,0],
                "message": f"Error parsing response: {str(e)}"
            }
            
    def decide_action(self,
                     env_description: str,
                     other_messages: List[str],
                     consensus_type: str = 'implicit') -> Tuple[Tuple[int, int], str]:
        """Get next action and message from LLM"""
        
        prompt = self._generate_prompt(env_description, other_messages, consensus_type)
        
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
                    model="gpt-4",
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
            
            # Parse action more robustly
            try:
                action = result['action']
                if isinstance(action, str):
                    action = action.strip('()[]').split(',')
                    action = tuple(int(x.strip()) for x in action)
                elif isinstance(action, (list, tuple)):
                    action = tuple(int(x) for x in action)
                else:
                    raise ValueError(f"Unexpected action format: {action}")
                
                # Validate coordinates are within grid
                if not (0 <= action[0] < 10 and 0 <= action[1] < 10):
                    raise ValueError("Coordinates out of bounds")
                    
                return action, result['message']
            except Exception as e:
                print(f"Error parsing action {result['action']}: {e}")
                return (0,0), "Error parsing action"
            
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return (0,0), "Error occurred"