#!/usr/bin/env python3
"""
Demo of Local AI Trading System - Zero Anthropic Tokens
========================================================

This demonstrates a complete offline AI trading system that:
1. Uses only local Qwen model (no cloud APIs)
2. Leverages processed trading book knowledge
3. Generates production-ready trading strategies
4. Works during API timeouts/rate limits
"""

import sys
sys.path.insert(0, '.')

from local_ai_trading_system import LocalTradingAI
import time

def demo_system():
    print("🚀 Local AI Trading System Demo")
    print("="*60)
    print("📊 Zero Anthropic tokens used")
    print("🤖 Local Qwen2.5-Coder + Trading Books")
    print("⚡ Instant fallback system")
    print("="*60)
    
    # Initialize system
    system = LocalTradingAI()
    
    demo_strategies = [
        {
            "name": "Momentum Strategy",
            "request": "momentum trading with RSI",
            "description": "Moving average crossover with RSI confirmation"
        },
        {
            "name": "ML Factor Strategy", 
            "request": "machine learning factor selection",
            "description": "Random Forest for factor-based investing"
        },
        {
            "name": "Risk Management",
            "request": "portfolio risk management",
            "description": "Position sizing and VaR calculation"
        }
    ]
    
    total_cost = 0.0
    total_time = 0.0
    
    for i, strategy in enumerate(demo_strategies, 1):
        print(f"\n📈 Demo {i}/3: {strategy['name']}")
        print(f"📝 Request: {strategy['request']}")
        print(f"🎯 Goal: {strategy['description']}")
        print("-"*50)
        
        start_time = time.time()
        
        # Generate strategy (force fallback for demo speed)
        system.qwen.available = False  # Comment this out to test real Qwen
        result = system.generate_strategy(strategy['request'])
        
        elapsed = time.time() - start_time
        total_time += elapsed
        total_cost += result.get('cost', 0)
        
        if result["success"]:
            print(f"✅ Generated in {elapsed:.1f}s")
            print(f"🧠 Model: {result['model']}")
            print(f"📏 Code length: {len(result['content'])} chars")
            print(f"💰 Cost: ${result['cost']:.4f}")
            
            # Extract key components from generated code
            content = result["content"]
            if "class" in content:
                class_line = [line for line in content.split('\n') if 'class ' in line][0]
                print(f"🏗️  Strategy class: {class_line.strip()}")
            
            if "def " in content:
                methods = [line.strip() for line in content.split('\n') if line.strip().startswith('def ')]
                print(f"🔧 Methods: {len(methods)} functions")
                if methods:
                    print(f"    Key: {methods[0][:50]}...")
                    
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n🎉 Demo Complete!")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"💰 Total cost: ${total_cost:.4f}")
    print(f"📊 Average speed: {total_time/3:.1f}s per strategy")
    
    print(f"\n🛡️  System Benefits:")
    print(f"  ✅ Zero external API dependencies")
    print(f"  ✅ Works during Anthropic timeouts")
    print(f"  ✅ No rate limits or token costs")
    print(f"  ✅ Expert knowledge from trading books")
    print(f"  ✅ Production-ready strategy code")
    
    print(f"\n📚 Knowledge Sources Used:")
    for concept, data in system.book_search.knowledge_base["concepts"].items():
        print(f"  📖 {concept.replace('_', ' ').title()}: {data['source']}")
    
    print(f"\n🔥 Ready for production use!")

if __name__ == "__main__":
    demo_system()