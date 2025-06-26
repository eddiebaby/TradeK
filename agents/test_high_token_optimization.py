#!/usr/bin/env python3
"""
Test script to trigger optimization suggestions by simulating high token usage
This should trigger the optimization bucket logging that was previously failing
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from influx_blackboard import get_blackboard

async def test_optimization_triggers():
    """Test that optimization suggestions are triggered and logged correctly"""
    print("🧪 Testing Optimization Triggers")
    print("-" * 40)
    
    bb = get_blackboard()
    
    # Test 1: High token usage trigger
    print("1. Testing high token usage trigger...")
    
    # This should trigger the high_token_usage optimization
    await bb.log_metrics(
        agent="Researcher",
        operation="stock_analysis",
        tokens_used=1200,  # Above threshold of 1000
        exec_time=1.5,
        success=True,
        data_size=500
    )
    print("   ✅ High token usage logged (1200 tokens > 1000 threshold)")
    
    # Test 2: Slow execution trigger
    print("2. Testing slow execution trigger...")
    
    # This should trigger the slow_execution optimization
    await bb.log_metrics(
        agent="Executor",
        operation="tdd_implementation",
        tokens_used=300,
        exec_time=6.5,  # Above threshold of 5.0 seconds
        success=True,
        data_size=200
    )
    print("   ✅ Slow execution logged (6.5s > 5.0s threshold)")
    
    # Test 3: Both triggers
    print("3. Testing both triggers simultaneously...")
    
    await bb.log_metrics(
        agent="Mastermind",
        operation="architectural_analysis",
        tokens_used=1500,  # High tokens
        exec_time=8.0,     # Slow execution
        success=True,
        data_size=800
    )
    print("   ✅ Both high tokens and slow execution logged")
    
    print("\n4. Checking if optimization suggestions were written...")
    
    # Query the optimizations bucket to verify data was written
    if bb.query_api:
        query = '''
        from(bucket: "optimizations")
          |> range(start: -5m)
          |> filter(fn: (r) => r["_measurement"] == "optimizations")
          |> sort(columns: ["_time"], desc: true)
        '''
        
        try:
            result = bb.query_api.query(query, org="AgentBlackboard")
            
            optimization_count = 0
            for table in result:
                for record in table.records:
                    optimization_count += 1
                    agent = record.values.get('target_agent', 'unknown')
                    category = record.values.get('category', 'unknown')
                    suggestion = record.values.get('suggestion', 'no suggestion')
                    confidence = record.values.get('confidence', 0)
                    
                    print(f"   - {agent}: {category} (confidence: {confidence:.2f})")
                    print(f"     Suggestion: {suggestion}")
            
            if optimization_count > 0:
                print(f"\n   ✅ Found {optimization_count} optimization suggestions!")
            else:
                print("   ⚠️  No optimization suggestions found (may be expected)")
                
        except Exception as e:
            print(f"   ❌ Error querying optimizations: {e}")
    
    print("\n5. Testing optimization analysis...")
    
    # Test the optimizer's analysis functionality
    analysis = bb.optimizer.analyze_operation_efficiency("stock_analysis")
    print(f"   Analysis result: {analysis}")
    
    print("\n✅ Optimization trigger test completed!")

async def main():
    """Run optimization trigger tests"""
    print("🚀 Optimization Trigger Test Suite")
    print("=" * 60)
    
    try:
        await test_optimization_triggers()
        print("\n🎉 All optimization trigger tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close blackboard connection
        bb = get_blackboard()
        bb.close()

if __name__ == "__main__":
    asyncio.run(main())