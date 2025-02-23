from typing import List, Tuple, Dict, Optional
import json
from anthropic import Anthropic
import openai
import numpy as np

class ContributorAgent:
    def __init__(self,
                 agent_id: int,
                 role: str,
                 llm_type: str = 'claude',
                 api_key: Optional[str] = None,
                 role_config: Optional[Dict] = None,
                 max_contribution: float = 20):
        self.agent_id = agent_id
        self.role = role
        self.llm_type = llm_type
        self.history = []
        self.role_config = role_config or {}
        self.max_contribution = max_contribution
        
        # Initialize LLM client
        if llm_type == 'claude':
            self.client = Anthropic(api_key=api_key)
        else:  # GPT
            openai.api_key = api_key
            
    def _get_default_role_instructions(self) -> str:
        """Get default role-specific instructions"""
        role_instructions = {
            'altruistic': """You are an altruistic contributor focused on ensuring public good provision.
- Prioritize meeting the threshold to ensure the public good is funded
- Willing to contribute more than your fair share if needed
- Consider long-term group welfare over individual gains
- Encourage others to contribute through positive messaging""",

            'strategic': """You are a strategic contributor focused on optimal resource allocation.
- Balance personal costs against public benefits
- Adjust contributions based on threshold and others' behavior
- Aim for fair cost distribution among participants
- Share strategic insights about optimal contribution levels""",

            'conservative': """You are a conservative contributor focused on minimizing risks.
- Prefer smaller, safer contributions
- Carefully evaluate threshold changes and volatility
- Focus on sustainable long-term participation
- Express concerns about high-risk situations""",

            'adaptive': """You are an adaptive contributor focused on responding to changes.
- Quickly adjust to threshold and benefit changes
- Learn from past rounds' outcomes
- Share observations about environmental changes
- Help group adapt to new conditions"""
        }
        
        return role_instructions.get(self.role.lower(),
                                  "Contribute to help fund the public good.")
    
    def _generate_prompt(self,
                        env_description: str,
                        other_messages: List[str],
                        consensus_type: str) -> str:
        """Generate prompt for LLM based on current state"""
        
        role_instruction = self.role_config.get('prompt_template',
                                              self._get_default_role_instructions())
        
        prompt = f"""You are Contributor {self.agent_id}, a {self.role} in a public goods provision team.

Current situation:
{env_description}

Other contributor messages:
{chr(10).join(other_messages)}

Your role specific instructions:
{role_instruction}

Consensus type: {consensus_type}

Based on the current situation and your role, provide:
1. Your analysis of the situation and strategy
2. Your proposed contribution amount (between 0 and {self.max_contribution})
3. A brief message to share with other contributors

Format your response as JSON exactly like this example:
{{
    "analysis": "My analysis of the situation...",
    "contribution": 10.5,
    "message": "My message to other contributors..."
}}

Remember:
- Consider the threshold and benefit values
- Keep contributions within [0, {self.max_contribution}]
- Think about group dynamics and coordination
- Balance individual and collective interests"""

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
            required_fields = ['analysis', 'contribution', 'message']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
                    
            # Validate contribution amount
            if not isinstance(result['contribution'], (int, float)):
                raise ValueError("Contribution must be a number")
                
            result['contribution'] = max(0, min(float(result['contribution']), 
                                              self.max_contribution))
                
            return result
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Original response: {response}")
            
            # Return safe default
            return {
                "analysis": "Error parsing response",
                "contribution": 0.0,
                "message": f"Error parsing response: {str(e)}"
            }
            
    def decide_action(self,
                     env_description: str,
                     other_messages: List[str],
                     consensus_type: str = 'implicit') -> Tuple[float, str]:
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
            
            return result['contribution'], result['message']
            
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return 0.0, "Error occurred"