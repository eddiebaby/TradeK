#!/usr/bin/env python3
"""
Test script for Qwen2.5-Coder ML Trading Strategy capabilities
"""

import requests
import json
import time

def test_qwen_coding():
    """Test Qwen2.5-Coder with ML trading strategy prompts"""
    
    ollama_url = "http://localhost:11434/api/generate"
    
    test_prompts = [
        {
            "name": "Simple Moving Average Strategy",
            "prompt": """Create a Python function to implement a simple moving average crossover trading strategy using pandas. Include parameters for short_window, long_window, and return buy/sell signals."""
        },
        {
            "name": "RSI Indicator",
            "prompt": """Write a Python function to calculate the Relative Strength Index (RSI) for a given price series using pandas and numpy. Include proper error handling."""
        },
        {
            "name": "Backtesting Framework",
            "prompt": """Create a basic backtesting class in Python that can test trading strategies on historical data, calculate returns, and generate performance metrics."""
        }
    ]
    
    results = {}
    
    for test in test_prompts:
        print(f"\n{'='*50}")
        print(f"Testing: {test['name']}")
        print(f"{'='*50}")
        
        payload = {
            "model": "qwen2.5-coder:7b",
            "prompt": test['prompt'],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent code
                "top_p": 0.9
            }
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(ollama_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                elapsed_time = time.time() - start_time
                
                print(f"Response time: {elapsed_time:.2f} seconds")
                print(f"Response preview: {result['response'][:200]}...")
                
                results[test['name']] = {
                    "success": True,
                    "response_time": elapsed_time,
                    "response_length": len(result['response'])
                }
            else:
                print(f"Error: {response.status_code} - {response.text}")
                results[test['name']] = {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"Exception: {e}")
            results[test['name']] = {"success": False, "error": str(e)}
    
    return results

if __name__ == "__main__":
    print("Testing Qwen2.5-Coder for ML Trading Strategy Development")
    print("="*60)
    
    # Wait for model to be ready
    print("Checking if model is loaded...")
    
    results = test_qwen_coding()
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result.get("success") else "❌ FAIL"
        if result.get("success"):
            print(f"{status} {test_name}: {result['response_time']:.2f}s")
        else:
            print(f"{status} {test_name}: {result.get('error', 'Unknown error')}")