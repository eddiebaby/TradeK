#!/usr/bin/env python3
"""
Integrate Ollama models with the agent system
"""
import subprocess
import json
import requests
import time

class OllamaAgentIntegration:
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"
        self.available_models = []
        
    def check_ollama_status(self):
        """Check if Ollama is running and get available models"""
        try:
            # Check if Ollama service is running
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Ollama is running")
                
                # Parse available models
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        self.available_models.append(model_name)
                
                print(f"📦 Available models: {', '.join(self.available_models)}")
                return True
            else:
                print("❌ Ollama is not running")
                return False
        except Exception as e:
            print(f"❌ Error checking Ollama: {e}")
            return False
    
    def test_model(self, model_name, prompt="Hello, introduce yourself briefly"):
        """Test a specific model with a simple prompt"""
        try:
            print(f"\n🧪 Testing {model_name}...")
            
            # Use ollama run command for testing
            result = subprocess.run([
                'ollama', 'run', model_name, prompt
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                print(f"✅ {model_name} responded successfully")
                print(f"📝 Response preview: {response[:100]}...")
                return True
            else:
                print(f"❌ {model_name} failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏱️ {model_name} timed out")
            return False
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            return False
    
    def create_agent_model_config(self):
        """Create configuration for agents to use Ollama models"""
        
        config = {
            "ollama_integration": {
                "enabled": True,
                "base_url": self.ollama_base_url,
                "available_models": self.available_models,
                "model_assignments": {
                    "researcher": "qwen3:8b" if "qwen3:8b" in self.available_models else "qwen3:4b",
                    "mastermind": "qwen3:8b" if "qwen3:8b" in self.available_models else "qwen3:4b", 
                    "executor": "qwen3:8b" if "qwen3:8b" in self.available_models else "qwen3:4b"
                },
                "fallback_model": self.available_models[0] if self.available_models else None,
                "timeout": 30,
                "max_tokens": 2048
            }
        }
        
        # Save config
        config_file = "/home/scottschweizer/TradeKnowledge/agents/config/ollama_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Saved Ollama config to {config_file}")
        return config
    
    def create_simple_agent_wrapper(self):
        """Create a simple wrapper to use Ollama with agents"""
        
        wrapper_code = '''#!/usr/bin/env python3
"""
Simple Ollama-powered agent for stock analysis
"""
import subprocess
import json

class SimpleOllamaAgent:
    def __init__(self, model_name="qwen3:4b"):
        self.model_name = model_name
        
    def analyze_stock(self, ticker):
        """Analyze a stock using Ollama model"""
        
        prompt = f"""
        As a financial analyst, provide a comprehensive analysis of {ticker} stock including:
        
        1. Company Overview and Business Model
        2. Recent Financial Performance 
        3. Key Strengths and Competitive Advantages
        4. Risk Factors and Challenges
        5. Market Position and Growth Prospects
        6. Investment Recommendation (Buy/Hold/Sell)
        7. Price Target and Risk Assessment
        
        Please provide specific, actionable insights for investment decisions.
        Format your response with clear sections and bullet points.
        """
        
        try:
            print(f"🤖 Using {self.model_name} to analyze {ticker}...")
            
            result = subprocess.run([
                'ollama', 'run', self.model_name, prompt
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return f"Analysis timed out for {ticker}"
        except Exception as e:
            return f"Error analyzing {ticker}: {e}"
    
    def quick_insight(self, ticker):
        """Get quick investment insight"""
        
        prompt = f"""
        Provide a brief investment insight for {ticker} stock in 3-4 sentences:
        1. Current investment thesis
        2. Key catalyst or risk
        3. Recommendation with confidence level
        """
        
        try:
            result = subprocess.run([
                'ollama', 'run', self.model_name, prompt
            ], capture_output=True, text=True, timeout=60)
            
            return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr}"
            
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    import sys
    
    agent = SimpleOllamaAgent()
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print(f"📈 Stock Analysis for {ticker}")
    print("=" * 50)
    
    analysis = agent.analyze_stock(ticker)
    print(analysis)
    
    print("\\n" + "=" * 50)
    print("⚠️ This is AI-generated analysis for research only")
'''
        
        wrapper_file = "/home/scottschweizer/TradeKnowledge/simple_ollama_agent.py"
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_code)
        
        # Make executable
        subprocess.run(['chmod', '+x', wrapper_file])
        
        print(f"🤖 Created simple Ollama agent: {wrapper_file}")
        return wrapper_file

def main():
    print("🔗 Ollama-Agent Integration Setup")
    print("=" * 40)
    
    integrator = OllamaAgentIntegration()
    
    # Check Ollama status
    if not integrator.check_ollama_status():
        print("❌ Please start Ollama first: ollama serve")
        return
    
    # Test available models
    working_models = []
    for model in integrator.available_models:
        if integrator.test_model(model):
            working_models.append(model)
    
    if not working_models:
        print("❌ No working models found")
        return
    
    print(f"\n✅ Working models: {', '.join(working_models)}")
    
    # Create configuration
    config = integrator.create_agent_model_config()
    
    # Create simple agent wrapper
    wrapper_file = integrator.create_simple_agent_wrapper()
    
    print(f"\n🎉 Integration complete!")
    print(f"🚀 Test the agent: python {wrapper_file} WMT")

if __name__ == "__main__":
    main()