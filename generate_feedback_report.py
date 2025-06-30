#!/usr/bin/env python3
"""
Generate Comprehensive Feedback Report for SPARC Agent Improvements
Extract and analyze detailed OpenAI feedback to create targeted improvement strategies
"""

import asyncio
from analyze_feedback import FeedbackAnalyzer


async def generate_comprehensive_report():
    """Generate full feedback report with detailed improvement strategies"""
    
    analyzer = FeedbackAnalyzer()
    
    print("📋 Generating Comprehensive Feedback Report")
    print("=" * 60)
    
    # Get detailed feedback for all agents
    results = await analyzer.analyze_all_agents()
    
    report = """
# SPARC Agent Improvement Report - Phase 3: Cross-Validation Feedback Analysis
**Target**: Improve from 7.0/10 to 9.5+/10 through targeted enhancements

## Executive Summary
Current scores after Phase 2 enhanced contexts:
- RESEARCHER: 6.0/10 → Target: 9.5/10 (3.5 points needed)
- MASTERMIND: 7.0/10 → Target: 9.5/10 (2.5 points needed)  
- EXECUTOR: 8.0/10 → Target: 9.5/10 (1.5 points needed)

## Detailed Feedback Analysis

"""
    
    # Process feedback for each agent
    for agent_name, analysis in results['agents'].items():
        if analysis['status'] == 'success':
            feedback = analysis['detailed_feedback']
            
            report += f"""
### {agent_name} Agent - Detailed Improvement Analysis

**Current Score**: {6.0 if agent_name == 'RESEARCHER' else 7.0 if agent_name == 'MASTERMIND' else 8.0}/10
**Target Score**: 9.5/10

#### OpenAI Feedback Analysis:
```
{feedback}
```

---
"""
        else:
            report += f"""
### {agent_name} Agent - Analysis Failed
**Error**: {analysis.get('error', 'Unknown error')}

---
"""
    
    report += """
## Implementation Strategy for 9.5+/10 Quality

### Phase 3a: Immediate Improvements
1. **EXECUTOR Agent** (8.0→9.5): Highest ROI, needs 1.5 points
   - Add missing architectural diagrams and technical specifications
   - Include comprehensive monitoring and alerting implementation
   - Add disaster recovery and operational procedures

2. **MASTERMIND Agent** (7.0→9.5): Medium complexity, needs 2.5 points  
   - Enhance business context and success metrics
   - Add detailed risk assessment with quantified impacts
   - Include technology alternatives with decision matrices

3. **RESEARCHER Agent** (6.0→9.5): Most complex, needs 3.5 points
   - Increase source diversity and credibility
   - Add quantitative analysis and confidence scoring
   - Include competitive intelligence and market positioning

### Phase 3b: Quality Enhancement Cycle
1. Implement targeted improvements based on specific feedback
2. Re-test with enhanced outputs
3. Analyze new feedback and iterate
4. Continue until 9.5+/10 achieved consistently

### Success Metrics
- Individual agent scores ≥ 9.5/10
- Overall average score ≥ 9.5/10  
- Consistent quality across multiple test scenarios
- Production-ready SPARC trio workflow
"""
    
    # Write report to file
    with open('/home/scottschweizer/TradeKnowledge/feedback_report.md', 'w') as f:
        f.write(report)
    
    print("✅ Comprehensive feedback report generated")
    print("📄 Report saved to: feedback_report.md")
    
    # Print summary of key improvements needed
    print(f"\n🎯 Key Improvement Areas Identified:")
    
    if results['agents']['RESEARCHER']['status'] == 'success':
        print(f"   📊 RESEARCHER: Source diversity, quantitative analysis, confidence scoring")
    
    if results['agents']['MASTERMIND']['status'] == 'success':
        print(f"   🏗️ MASTERMIND: Business context, risk quantification, decision matrices")
        
    if results['agents']['EXECUTOR']['status'] == 'success':
        print(f"   ⚡ EXECUTOR: Architectural detail, monitoring implementation, operations")
    
    print(f"\n🚀 Ready for Phase 3a: Targeted Quality Improvements")
    
    return results


if __name__ == "__main__":
    asyncio.run(generate_comprehensive_report())