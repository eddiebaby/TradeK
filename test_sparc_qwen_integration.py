#!/usr/bin/env python3
"""
Test SPARC trio integration with Qwen2.5-Coder
"""

import sys
import os
sys.path.append('/home/scott/TradeKnowledge/agents')
sys.path.append('/home/scott/TradeKnowledge/agents/core')

from model_router import HybridModelRouter, TaskContext

def test_model_routing():
    """Test that Qwen2.5-Coder is properly selected for coding tasks"""
    
    router = HybridModelRouter()
    
    test_cases = [
        {
            "name": "Python ML Trading Strategy",
            "task": TaskContext(
                agent_name="EXECUTOR",
                operation="implementation",
                description="Create a Python ML trading strategy using pandas and numpy",
                expected_output_length=800
            ),
            "expected_model": "qwen2.5-coder:7b"
        },
        {
            "name": "Algorithm Development",
            "task": TaskContext(
                agent_name="EXECUTOR", 
                operation="algorithm_development",
                description="Implement a momentum trading algorithm in Python",
                expected_output_length=600
            ),
            "expected_model": "qwen2.5-coder:7b"
        },
        {
            "name": "Backtesting Framework",
            "task": TaskContext(
                agent_name="EXECUTOR",
                operation="testing",
                description="Create backtesting framework for trading strategies",
                expected_output_length=1000
            ),
            "expected_model": "qwen2.5-coder:7b"
        },
        {
            "name": "Strategic Planning",
            "task": TaskContext(
                agent_name="MASTERMIND",
                operation="strategic_planning", 
                description="Design architecture for trading system",
                expected_output_length=1200
            ),
            "expected_model": "mixtral:8x7b"  # Should use mixtral for strategy
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {test_case['name']}")
        print(f"{'='*50}")
        
        task = test_case['task']
        choice = router.route_task(task)
        selected_model = router.select_model(task, choice)
        
        print(f"Task: {task.description}")
        print(f"Operation: {task.operation}")
        print(f"Routing Choice: {choice}")
        print(f"Selected Model: {selected_model}")
        print(f"Expected Model: {test_case['expected_model']}")
        
        success = selected_model == test_case['expected_model']
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"Result: {status}")
        
        results[test_case['name']] = {
            "success": success,
            "selected_model": selected_model,
            "expected_model": test_case['expected_model']
        }
    
    return results

if __name__ == "__main__":
    print("Testing SPARC Trio + Qwen2.5-Coder Integration")
    print("="*60)
    
    results = test_model_routing()
    
    print(f"\n{'='*60}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result['success']:
            print(f"    Expected: {result['expected_model']}")
            print(f"    Got: {result['selected_model']}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Qwen2.5-Coder integration successful!")
    else:
        print("⚠️ Some tests failed. Check model routing logic.")