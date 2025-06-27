#!/usr/bin/env python3
"""
IWM Institutional Analysis Demo

This demonstrates comprehensive analysis specifically tailored for IWM
(iShares Russell 2000 ETF) with ETF-specific insights and metrics.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.comprehensive_analyzer import ComprehensiveStockAnalyzer
from src.fundamental.models import *

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    rprint = print


class IWMInstitutionalAnalyzer(ComprehensiveStockAnalyzer):
    """Specialized analyzer for IWM with ETF-specific insights."""
    
    def _create_iwm_company_profile(self, symbol: str) -> CompanyProfile:
        """Create detailed IWM profile."""
        return CompanyProfile(
            symbol="IWM",
            company_name="iShares Russell 2000 ETF",
            sector="Financial Services - ETF",
            industry="Exchange Traded Fund",
            market_cap=Decimal("60500000000"),
            description="""The iShares Russell 2000 ETF seeks to track the Russell 2000 Index, 
            which represents approximately 2000 small-capitalization U.S. companies. As the 
            premier small-cap ETF, IWM provides broad exposure to the small-cap segment of 
            the U.S. equity market. Small-cap companies typically have market capitalizations 
            between $300M - $2B and represent approximately 10% of total U.S. market 
            capitalization. IWM is widely used by institutional and retail investors for 
            small-cap exposure, sector rotation strategies, and as a barometer of domestic 
            economic health and risk sentiment.""",
            exchange="NYSE Arca"
        )
    
    def _generate_iwm_investment_thesis(self, symbol: str, **kwargs) -> InvestmentThesis:
        """Generate IWM-specific investment thesis."""
        
        market_data = kwargs.get('market_data')
        current_price = market_data.current_price if market_data else Decimal('215.00')
        
        return InvestmentThesis(
            symbol="IWM",
            rating=InvestmentRating.BUY,
            price_target=current_price * Decimal('1.12'),  # 12% upside
            bull_case_target=current_price * Decimal('1.25'),  # 25% upside
            bear_case_target=current_price * Decimal('0.88'),  # 12% downside
            investment_rationale="""IWM represents a compelling opportunity to gain exposure 
            to U.S. small-cap companies, which historically outperform during economic expansion 
            phases and benefit from domestic economic growth. The Russell 2000 companies are 
            primarily domestically focused, providing pure-play exposure to U.S. economic trends 
            without significant international currency or geopolitical risks.""",
            bull_case_points=[
                "Economic Recovery: Small-caps typically outperform during economic expansion cycles",
                "Domestic Focus: Less exposure to international trade wars and currency fluctuations", 
                "Rate Environment: Small-caps benefit from falling interest rates and credit expansion",
                "M&A Activity: Small-caps are frequent acquisition targets driving premiums",
                "Innovation Premium: Small companies often lead in emerging technologies and business models",
                "Liquidity Premium: IWM provides excellent liquidity for small-cap exposure",
                "Diversification: 2000+ holdings provide broad small-cap market exposure"
            ],
            bear_case_points=[
                "Economic Sensitivity: Small-caps are more vulnerable to economic downturns",
                "Interest Rate Risk: Higher sensitivity to rising rates and credit tightening",
                "Volatility Risk: Higher volatility than large-cap alternatives",
                "Liquidity Risk: Underlying companies may have lower liquidity during stress",
                "Execution Risk: Small companies have higher business execution risks",
                "Market Concentration: Potential for sector concentration in certain market cycles"
            ],
            key_catalysts=[
                "Federal Reserve policy shifts toward accommodation",
                "Infrastructure spending and domestic investment",
                "Economic data showing domestic strength",
                "Credit market expansion and small business lending",
                "Merger and acquisition activity in small-cap space"
            ],
            key_risks=[
                "Economic recession or slowdown",
                "Federal Reserve interest rate hikes",
                "Credit market tightening",
                "Large-cap outperformance trend",
                "International trade disruptions affecting domestic companies"
            ],
            short_term_outlook="""Near-term performance will depend on Federal Reserve policy, 
            economic data trends, and relative performance vs large-caps. Key focus on interest 
            rate environment and credit conditions.""",
            medium_term_outlook="""Medium-term returns driven by economic cycle positioning, 
            small-cap vs large-cap relative performance, and domestic economic growth trends. 
            Potential for significant outperformance in right economic environment.""",
            long_term_outlook="""Long-term value from small-cap premium, innovation leadership, 
            and acquisition activity. Historical small-cap outperformance over long periods 
            supports strategic allocation.""",
            monitoring_points=[
                "Federal Reserve policy announcements and rate decisions",
                "Economic indicators: GDP, employment, consumer spending",
                "Small-cap vs large-cap relative performance (IWM vs SPY)",
                "Credit spreads and small business lending conditions", 
                "Russell 2000 earnings growth and margin trends",
                "Merger and acquisition activity in small-cap space",
                "Sector rotation patterns and risk-on/risk-off sentiment",
                "IWM expense ratio and tracking efficiency",
                "Options activity and institutional flows",
                "International economic developments affecting domestic focus"
            ],
            portfolio_allocation="5-15% allocation for core equity portfolios, up to 25% for small-cap focused strategies",
            risk_level=RiskLevel.MODERATE_HIGH,
            time_horizon="3-7 year investment cycle to capture small-cap premium",
            portfolio_fit="Essential for diversified equity exposure, tactical allocation tool, domestic economic play"
        )
    
    def _assess_iwm_risks(self, **kwargs) -> RiskAssessment:
        """Assess IWM-specific risks."""
        return RiskAssessment(
            symbol="IWM",
            overall_risk_level=RiskLevel.MODERATE_HIGH,
            regulatory_risks=[
                "ETF Structure Risk: Changes in ETF regulations or tax treatment",
                "Index Methodology Risk: Changes to Russell 2000 index construction",
                "Securities Lending Risk: Revenue and counterparty risks from securities lending",
                "Liquidity Risk: Market maker and authorized participant concentration",
                "Tracking Error Risk: Deviation from underlying index performance"
            ],
            business_risks=[
                "Market Risk: Broad small-cap market decline and volatility",
                "Economic Cycle Risk: Small-cap sensitivity to economic downturns",
                "Interest Rate Risk: Small-cap sensitivity to rising rates",
                "Credit Risk: Small company financing and refinancing risks",
                "Sector Concentration Risk: Overweight in cyclical sectors",
                "Size Factor Risk: Small-cap underperformance vs large-cap"
            ],
            growth_catalysts=[
                "Economic Expansion: Small-cap outperformance during growth cycles",
                "Rate Cuts: Benefit from lower interest rates and credit expansion",
                "Domestic Policy: Infrastructure and domestic investment initiatives",
                "Innovation Premium: Small company technology and business model innovation",
                "M&A Activity: Acquisition premiums and takeout activity",
                "Liquidity Expansion: Increased institutional allocation to small-caps"
            ],
            beta=Decimal("1.2"),  # Small-caps typically have higher beta
        )

    async def generate_iwm_report(self) -> str:
        """Generate specialized IWM institutional report."""
        
        if HAS_RICH:
            console.print(f"\n[bold cyan]🏛️ IWM Institutional Analysis[/bold cyan]")
            console.print("[yellow]Generating ETF-specific comprehensive report...[/yellow]")
        else:
            print("\n🏛️ IWM Institutional Analysis")
            print("Generating ETF-specific comprehensive report...")
        
        # Override specific components for IWM
        analysis = await self.analyze_stock("IWM")
        
        # Customize for IWM
        analysis.company_profile = self._create_iwm_company_profile("IWM")
        analysis.investment_thesis = self._generate_iwm_investment_thesis("IWM", market_data=analysis.market_data)
        analysis.risk_assessment = self._assess_iwm_risks()
        
        # Generate custom report
        report = await self.generate_report(analysis)
        
        # Add IWM-specific sections
        iwm_addendum = self._generate_iwm_addendum(analysis)
        full_report = report + "\n\n" + iwm_addendum
        
        return full_report
    
    def _generate_iwm_addendum(self, analysis) -> str:
        """Generate IWM-specific analysis addendum."""
        
        current_price = float(analysis.market_data.current_price)
        
        addendum = f"""
# 📊 IWM ETF-Specific Analysis Addendum

## 🎯 Small-Cap Market Analysis

### Russell 2000 Index Characteristics
- **Index Components**: ~2000 small-cap U.S. companies
- **Market Cap Range**: Typically $300M - $2B per company
- **Sector Diversification**: Broad exposure across all sectors
- **Geographic Focus**: Primarily domestic U.S. companies
- **Rebalancing**: Annual reconstitution in June

### Current Market Environment
- **Current Price**: ${current_price:.2f}
- **Small-Cap Premium**: Historical 2-3% annual outperformance vs large-caps
- **Economic Sensitivity**: 1.2-1.5x beta to broader market
- **Interest Rate Sensitivity**: Higher than large-caps due to financing needs

## 📈 ETF Mechanics & Efficiency

### Fund Characteristics
- **Expense Ratio**: 0.19% (competitive for small-cap exposure)
- **AUM**: ~$60 billion (excellent liquidity and scale)
- **Daily Volume**: High institutional and retail trading volume
- **Tracking Error**: Typically <0.25% annual vs Russell 2000 Index
- **Tax Efficiency**: ETF structure provides tax advantages vs mutual funds

### Institutional Considerations
- **Liquidity**: Excellent for large block trades via authorized participants
- **Options Market**: Robust options market for hedging and income strategies
- **Securities Lending**: Additional revenue from securities lending program
- **Dividend Policy**: Quarterly distributions reflecting underlying holdings

## 🔍 Small-Cap Investment Framework

### Optimal Market Conditions for IWM
1. **Economic Expansion**: Early-to-mid cycle economic growth
2. **Falling Rates**: Accommodative monetary policy
3. **Credit Expansion**: Easy access to growth capital
4. **Domestic Focus**: Strong U.S. economic fundamentals
5. **Risk-On Sentiment**: Investor appetite for growth and volatility

### Warning Signals for IWM
1. **Yield Curve Inversion**: Recession signals
2. **Credit Tightening**: Higher financing costs
3. **Dollar Strength**: Can hurt domestically focused companies
4. **Large-Cap Outperformance**: Factor rotation away from small-caps
5. **Geopolitical Stress**: Flight to quality assets

## 💼 Portfolio Implementation Strategies

### Core Holding Strategy
- **Allocation**: 5-10% of equity portfolio
- **Rationale**: Diversification and small-cap exposure
- **Rebalancing**: Annual or based on relative performance

### Tactical Allocation Strategy
- **Allocation**: 0-20% based on economic cycle
- **Signals**: Economic indicators, yield curve, credit spreads
- **Risk Management**: Stop-loss or options hedging

### Pairs Trading Strategy
- **Long IWM / Short SPY**: Pure small-cap vs large-cap play
- **Sector Rotation**: IWM vs sector ETFs based on cycle
- **International**: IWM vs international small-cap ETFs

## 📊 Key Performance Metrics

### Historical Performance (10-Year Averages)
- **Annual Return**: ~11-12% (varies by period)
- **Volatility**: ~20-25% annual
- **Sharpe Ratio**: ~0.5-0.6
- **Maximum Drawdown**: ~50-60% during major corrections
- **Beta**: ~1.2-1.3 vs S&P 500

### Relative Performance Drivers
- **Economic Growth**: Positive correlation with GDP growth
- **Interest Rates**: Negative correlation with rate changes
- **Credit Spreads**: Performance deteriorates with wider spreads
- **Dollar Strength**: Generally negative for domestic small-caps

## 🎯 Current Investment Recommendation

### Rating: **BUY** 📈
### Rationale
Given current economic conditions, interest rate environment, and small-cap valuations,
IWM presents an attractive opportunity for:

1. **Diversification**: Complement large-cap holdings
2. **Economic Recovery Play**: Benefit from domestic growth
3. **Rate Sensitivity**: Positive from potential rate cuts
4. **Valuation**: Reasonable valuations vs historical levels

### Position Sizing Recommendations
- **Conservative**: 3-5% portfolio allocation
- **Moderate**: 5-10% portfolio allocation  
- **Aggressive**: 10-15% portfolio allocation
- **Tactical**: Up to 20% during favorable conditions

### Risk Management
- **Stop Loss**: 15-20% below entry for tactical positions
- **Hedging**: Consider put options during high volatility
- **Diversification**: Combine with international and large-cap exposure

================================================================================
✅ IWM ETF-SPECIFIC ANALYSIS COMPLETE
🎯 Position IWM as core small-cap allocation with tactical overlay capability
📊 Monitor economic cycle, rate environment, and relative performance trends
⚠️ Maintain appropriate position sizing given inherent small-cap volatility
================================================================================
"""
        
        return addendum


async def main():
    """Run IWM institutional analysis."""
    
    if HAS_RICH:
        console.print("[bold green]🏛️ IWM Institutional-Grade Analysis[/bold green]")
        console.print("[dim]Specialized ETF analysis with institutional insights[/dim]\n")
    else:
        print("🏛️ IWM Institutional-Grade Analysis")
        print("Specialized ETF analysis with institutional insights\n")
    
    try:
        analyzer = IWMInstitutionalAnalyzer()
        
        # Generate comprehensive IWM report
        report = await analyzer.generate_iwm_report()
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"IWM_institutional_analysis_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        if HAS_RICH:
            console.print(f"[green]✅ Institutional IWM report generated: {filename}[/green]")
            console.print(f"[dim]Report length: {len(report):,} characters[/dim]")
            
            # Show key insights
            table = Table(title="🎯 IWM Key Insights")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Implication", style="yellow")
            
            table.add_row("Asset Class", "Small-Cap ETF", "Domestic Economic Exposure")
            table.add_row("Risk Level", "Moderate-High", "Higher Volatility Expected")
            table.add_row("Beta", "~1.2", "More Volatile than Market")
            table.add_row("Best Environment", "Economic Expansion", "Early-Cycle Outperformance")
            table.add_row("Rate Sensitivity", "High", "Benefits from Rate Cuts")
            
            console.print(table)
        else:
            print(f"✅ Institutional IWM report generated: {filename}")
            print(f"Report length: {len(report):,} characters")
        
        print(f"\n🎉 IWM institutional analysis complete!")
        return filename
        
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Analysis failed: {e}[/red]")
        else:
            print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())