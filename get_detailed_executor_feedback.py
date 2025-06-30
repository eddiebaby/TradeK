#!/usr/bin/env python3
"""
Get Detailed EXECUTOR Feedback Analysis
Understand why score decreased from 8.0 to 7.0 and identify specific improvements needed
"""

import asyncio
from test_improved_executor_simple import ImprovedExecutorTester


async def get_detailed_feedback():
    """Get detailed feedback for the improved EXECUTOR output"""
    
    print("🔍 Getting Detailed EXECUTOR Feedback Analysis")
    print("=" * 60)
    
    tester = ImprovedExecutorTester()
    
    if not tester.cross_validator.available:
        print("❌ Cross-validation not available - missing API keys")
        return
    
    # Generate the improved output
    improved_output = tester.generate_improved_executor_output("Algorithmic Trading Platform")
    
    # Get detailed feedback
    detailed_feedback_prompt = f"""
Please provide detailed analysis of this EXECUTOR agent output and explain why it scored 7/10 instead of 9.5+/10.

EXECUTOR OUTPUT:
{improved_output[:3000]}...

Please analyze:

1. SPECIFIC QUALITY GAPS (why not 9.5+/10):
   - What critical elements are missing?
   - Which sections lack depth or detail?
   - What implementation specifics are unclear?

2. TECHNICAL DEFICIENCIES:
   - Missing architectural components?
   - Insufficient technical specifications?
   - Unclear implementation details?

3. OPERATIONAL GAPS:
   - Missing operational procedures?
   - Insufficient monitoring details?
   - Unclear disaster recovery specifics?

4. QUALITY IMPROVEMENTS NEEDED:
   - What would make this 9.5+/10 quality?
   - Which sections need the most work?
   - What industry standards are missing?

5. ACTIONABLE FIXES (top 5 priorities):
   - Most impactful improvements
   - Specific additions needed
   - Critical gaps to address

Focus on specific, actionable feedback that would elevate this from 7/10 to 9.5+/10.
"""
    
    try:
        detailed_response = tester.cross_validator._call_zen_mcp(
            tool="chat",
            params={
                "prompt": detailed_feedback_prompt,
                "model": "o4-mini",
                "temperature": 0.1
            }
        )
        
        if detailed_response and len(detailed_response) > 50:
            print("✅ Detailed feedback received")
            print("=" * 60)
            print(detailed_response)
            print("=" * 60)
            
            # Save detailed feedback to file
            with open('/home/scottschweizer/TradeKnowledge/executor_detailed_feedback.md', 'w') as f:
                f.write(f"""# EXECUTOR Agent Detailed Feedback Analysis
**Current Score**: 7.0/10
**Target Score**: 9.5/10
**Gap**: 2.5 points

## Analysis Date
{datetime.now().isoformat()}

## Detailed OpenAI Feedback

{detailed_response}

## Next Steps
Based on this feedback, implement the top 5 priority improvements to target 9.5+/10 quality.
""")
            
            print(f"\n📄 Detailed feedback saved to: executor_detailed_feedback.md")
            print(f"🎯 Ready to implement targeted improvements based on specific feedback")
            
        else:
            print(f"❌ Failed to get detailed feedback: {detailed_response}")
            
    except Exception as e:
        print(f"❌ Error getting detailed feedback: {e}")


if __name__ == "__main__":
    asyncio.run(get_detailed_feedback())