#!/usr/bin/env python3
"""
Simple Ollama-powered stock analysis agent
Works with any available Ollama model
"""
import subprocess
import sys
import json

class SimpleStockAgent:
    def __init__(self):
        self.available_models = self.get_available_models()
        self.model = self.select_best_model()
    
    def get_available_models(self):
        """Get list of available Ollama models"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            return []
        except:
            return []
    
    def select_best_model(self):
        """Select the best available model"""
        if not self.available_models:
            return None
        
        # Prefer qwen models, then any available model
        preferred_order = ['qwen3:4b', 'qwen3:8b', 'gemma2:2b', 'phi3', 'llama3']
        
        for preferred in preferred_order:
            if preferred in self.available_models:
                return preferred
        
        return self.available_models[0]  # Use first available
    
    def analyze_stock_simple(self, ticker):
        """Simple stock analysis"""
        
        if not self.model:
            return self.fallback_analysis(ticker)
        
        prompt = f"""
        Analyze {ticker} stock and provide:
        
        1. Company overview (2-3 sentences)
        2. Key strengths (3 bullet points)
        3. Main risks (3 bullet points) 
        4. Investment recommendation (Buy/Hold/Sell with brief reason)
        
        Keep response concise and focused on actionable insights.
        """
        
        try:
            print(f"🤖 Using {self.model} to analyze {ticker}...")
            
            result = subprocess.run([
                'ollama', 'run', self.model, prompt
            ], capture_output=True, text=True, timeout=90)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Model error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return f"Analysis timed out for {ticker}"
        except Exception as e:
            return self.fallback_analysis(ticker)
    
    def fallback_analysis(self, ticker):
        """Fallback analysis when no model available"""
        
        if ticker.upper() == "WMT":
            return """
🏪 Walmart Inc. (WMT) Analysis

Company Overview:
Walmart is the world's largest retailer with strong fundamentals and defensive characteristics. The company operates over 10,500 stores globally and has been investing heavily in e-commerce and digital transformation.

Key Strengths:
• Market leadership with unmatched scale and distribution network
• Strong cash flow generation and dividend aristocrat status  
• Growing e-commerce business with omnichannel capabilities

Main Risks:
• Intense competition from Amazon and other retailers
• Labor cost inflation and margin pressure
• Economic sensitivity affecting consumer spending

Investment Recommendation: HOLD
Walmart offers defensive qualities with steady dividends, but growth is limited. Suitable for conservative portfolios seeking stability and income.
"""
        else:
            return f"""
📈 {ticker.upper()} Basic Analysis

This is a simplified analysis. For {ticker}, consider:

Key Factors to Research:
• Recent earnings and revenue trends
• Industry position and competitive moats
• Management quality and strategic direction
• Valuation metrics (P/E, P/B, etc.)
• Financial health (debt levels, cash flow)

Recommendation: RESEARCH
Please conduct detailed fundamental analysis before making investment decisions.

⚠️ This is educational content only, not investment advice.
"""

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "WMT"
    
    print(f"📈 Stock Analysis for {ticker.upper()}")
    print("=" * 50)
    
    agent = SimpleStockAgent()
    
    if agent.available_models:
        print(f"🤖 Available models: {', '.join(agent.available_models)}")
        print(f"🎯 Using: {agent.model}")
    else:
        print("⚠️ No Ollama models available - using fallback analysis")
    
    print("\n" + "="*50)
    
    analysis = agent.analyze_stock_simple(ticker)
    print(analysis)
    
    print("\n" + "="*50)
    print("⚠️ This analysis is for educational purposes only")
    print("💡 Please consult a financial advisor for investment decisions")

if __name__ == "__main__":
    main()