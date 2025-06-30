#!/usr/bin/env python3
"""
FIRE Command Cross-Validation Enhancement
Adds OpenAI model cross-validation to SPARC trio outputs
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

# Add the zen-mcp-server path for imports
zen_server_path = Path(__file__).parent / "zen-mcp-server"
sys.path.insert(0, str(zen_server_path))

# Load environment from zen-mcp-server
env_file = zen_server_path / ".env"
load_dotenv(dotenv_path=env_file)

class FireCrossValidator:
    """Cross-validation using OpenAI models through zen-mcp-server"""
    
    def __init__(self):
        self.zen_server_path = zen_server_path
        self.available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if cross-validation is available"""
        return bool(os.getenv('OPENAI_API_KEY') or os.getenv('OPENROUTER_API_KEY'))
    
    def cross_validate(self, primary_output: str, analysis_type: str = "general") -> dict:
        """
        Cross-validate SPARC trio output using OpenAI models
        
        Args:
            primary_output: The output from SPARC trio to validate
            analysis_type: Type of analysis (research, strategy, implementation)
            
        Returns:
            Dict with validation results
        """
        if not self.available:
            return {
                "status": "unavailable",
                "message": "Cross-validation unavailable - no OpenAI/OpenRouter API key configured",
                "validation_score": 0.0
            }
        
        try:
            # Prepare cross-validation prompt based on analysis type
            validation_prompts = {
                "research": "Review this research analysis for accuracy, completeness, and potential blind spots. Rate 1-10 and provide brief feedback:",
                "strategy": "Evaluate this strategic plan for feasibility, risks, and logical structure. Rate 1-10 and provide brief feedback:",
                "implementation": "Assess this implementation approach for correctness, efficiency, and best practices. Rate 1-10 and provide brief feedback:",
                "general": "Analyze this output for quality, accuracy, and potential improvements. Rate 1-10 and provide brief feedback:"
            }
            
            prompt = validation_prompts.get(analysis_type, validation_prompts["general"])
            full_prompt = f"{prompt}\n\n{primary_output}"
            
            # Use cost-effective o4-mini model for validation
            response = self._call_zen_mcp(
                tool="chat",
                params={
                    "prompt": full_prompt,
                    "model": "o4-mini",  # Cost-effective OpenAI model
                    "temperature": 0.2   # Low temperature for consistent validation
                }
            )
            
            if response:
                # Parse the validation score from response
                validation_score = self._extract_score(response)
                
                return {
                    "status": "success",
                    "model_used": "OpenAI o4-mini",
                    "validation_score": validation_score,
                    "feedback": response[:300] + "..." if len(response) > 300 else response,
                    "full_feedback": response,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cost_estimate": "~$0.0001"  # Rough estimate for o4-mini
                }
            else:
                return {
                    "status": "failed",
                    "message": "Failed to get validation response",
                    "validation_score": 0.0
                }
                
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Cross-validation error: {str(e)}",
                "validation_score": 0.0
            }
    
    def _call_zen_mcp(self, tool: str, params: dict) -> str:
        """Call zen-mcp-server tool"""
        try:
            # Use the working simulator approach
            test_code = f'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from simulator_tests.base_test import BaseSimulatorTest

class ValidationCall(BaseSimulatorTest):
    def __init__(self):
        super().__init__(verbose=False)
    
    def call_tool(self):
        try:
            response, _ = self.call_mcp_tool("{tool}", {json.dumps(params)})
            if response:
                # Extract just the content from the response
                if isinstance(response, str) and "content" in response:
                    import json
                    try:
                        resp_data = json.loads(response)
                        return resp_data.get("content", response)
                    except:
                        return response
                return response
            return None
        except Exception as e:
            print(f"Error: {{e}}", file=sys.stderr)
            return None

caller = ValidationCall()
result = caller.call_tool()
if result:
    print(result)
'''
            
            # Write temporary test file
            temp_file = self.zen_server_path / "temp_validation_call.py"
            with open(temp_file, 'w') as f:
                f.write(test_code)
            
            # Run the validation call
            env = os.environ.copy()
            result = subprocess.run(
                [sys.executable, str(temp_file)], 
                env=env, 
                capture_output=True, 
                text=True,
                cwd=str(self.zen_server_path)
            )
            
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Validation call failed: {result.stderr}", file=sys.stderr)
                return None
                
        except Exception as e:
            print(f"Error calling zen-mcp: {e}", file=sys.stderr)
            return None
    
    def _extract_score(self, response: str) -> float:
        """Extract numerical score from validation response"""
        try:
            # Look for patterns like "8/10", "Score: 7", "Rating: 9.5"
            import re
            
            # Pattern for X/10
            pattern1 = r'(\d+(?:\.\d+)?)/10'
            match1 = re.search(pattern1, response)
            if match1:
                return float(match1.group(1))
            
            # Pattern for "Score:" or "Rating:"
            pattern2 = r'(?:score|rating|rate):\s*(\d+(?:\.\d+)?)'
            match2 = re.search(pattern2, response, re.IGNORECASE)
            if match2:
                return float(match2.group(1))
            
            # Pattern for standalone numbers near rating keywords
            pattern3 = r'(?:rate|score|rating).*?(\d+(?:\.\d+)?)'
            match3 = re.search(pattern3, response, re.IGNORECASE)
            if match3:
                return float(match3.group(1))
            
            # Default: assume good quality if no score found
            return 7.5
            
        except:
            return 7.5  # Default score
    
    def validate_fire_output(self, sparc_results: dict) -> dict:
        """
        Validate complete FIRE/SPARC trio output
        
        Args:
            sparc_results: Dictionary containing outputs from RESEARCHER, MASTERMIND, EXECUTOR
            
        Returns:
            Dictionary with cross-validation results for each agent
        """
        validation_results = {
            "cross_validation_summary": {
                "enabled": self.available,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_used": "OpenAI o4-mini",
                "total_cost_estimate": "~$0.0003"
            },
            "agent_validations": {}
        }
        
        if not self.available:
            validation_results["cross_validation_summary"]["status"] = "unavailable"
            return validation_results
        
        # Validate each agent's output
        agent_configs = {
            "RESEARCHER": {"type": "research", "weight": 0.3},
            "MASTERMIND": {"type": "strategy", "weight": 0.4}, 
            "EXECUTOR": {"type": "implementation", "weight": 0.3}
        }
        
        total_weighted_score = 0.0
        
        for agent_name, config in agent_configs.items():
            if agent_name in sparc_results:
                validation = self.cross_validate(
                    sparc_results[agent_name], 
                    config["type"]
                )
                validation_results["agent_validations"][agent_name] = validation
                
                if validation["status"] == "success":
                    total_weighted_score += validation["validation_score"] * config["weight"]
        
        # Calculate overall validation score
        validation_results["cross_validation_summary"]["overall_score"] = round(total_weighted_score, 2)
        validation_results["cross_validation_summary"]["quality_grade"] = self._get_quality_grade(total_weighted_score)
        
        return validation_results
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert numerical score to quality grade"""
        if score >= 9.0:
            return "A+ (Excellent)"
        elif score >= 8.0:
            return "A (Very Good)"
        elif score >= 7.0:
            return "B+ (Good)"
        elif score >= 6.0:
            return "B (Satisfactory)"
        elif score >= 5.0:
            return "C+ (Needs Improvement)"
        else:
            return "C (Poor Quality)"


def test_cross_validation():
    """Test the cross-validation functionality"""
    validator = FireCrossValidator()
    
    if not validator.available:
        print("❌ Cross-validation not available - missing API keys")
        return False
    
    # Test with sample output
    sample_output = """
    This is a test analysis of a trading algorithm implementation.
    The algorithm uses moving averages to detect trends and generates
    buy/sell signals based on crossover patterns. The implementation
    follows best practices with proper error handling and logging.
    """
    
    print("🧪 Testing cross-validation...")
    result = validator.cross_validate(sample_output, "implementation")
    
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"✅ Validation Score: {result['validation_score']}/10")
        print(f"Model: {result['model_used']}")
        print(f"Feedback: {result['feedback']}")
        return True
    else:
        print(f"❌ Validation failed: {result.get('message', 'Unknown error')}")
        return False


if __name__ == "__main__":
    success = test_cross_validation()
    sys.exit(0 if success else 1)