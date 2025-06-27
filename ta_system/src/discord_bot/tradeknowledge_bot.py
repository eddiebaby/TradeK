#!/usr/bin/env python3
"""
TradeKnowledge Discord Bot - Main Bot Implementation

Discord-first interface for systematic options trading analysis.
Designed for technical professionals like aerospace engineers who trade systematically.

Features:
- Options chain analysis with neural network confidence intervals
- Diagonal spreads strategy identification  
- Real-time market alerts and notifications
- Multi-agent blackboard system integration
- Free-to-paid conversion tracking
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Any

import discord
from discord.ext import commands, tasks
import pandas as pd

from ..comprehensive_analyzer import ComprehensiveStockAnalyzer
from .subscription_manager import SubscriptionManager
from .options_analyzer import OptionsAnalyzer
from .neural_agents import NeuralNetworkAgent
from .blackboard_system import BlackboardSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeKnowledgeBot(commands.Bot):
    """
    TradeKnowledge Discord Bot for systematic options trading.
    
    Provides Bloomberg-quality analysis through Discord interface
    at a fraction of the cost with engineer-friendly UX.
    """
    
    def __init__(self):
        """Initialize the TradeKnowledge bot with required intents."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        
        super().__init__(
            command_prefix='!tk ',
            intents=intents,
            description="TradeKnowledge - Professional options trading intelligence"
        )
        
        # Core systems
        self.analyzer = ComprehensiveStockAnalyzer()
        self.subscription_manager = SubscriptionManager()
        self.options_analyzer = OptionsAnalyzer()
        self.neural_agent = NeuralNetworkAgent()
        self.blackboard = BlackboardSystem()
        
        # Bot state
        self.startup_time = datetime.now()
        self.command_count = 0
        self.active_users = set()
        
        # Load configuration
        self.free_tier_daily_limit = 5
        self.engineer_tier_limit = 100
        self.rocket_scientist_unlimited = True

    async def setup_hook(self):
        """Initialize bot systems and sync slash commands."""
        logger.info("🚀 Initializing TradeKnowledge Bot systems...")
        
        # Initialize subsystems
        await self.blackboard.initialize()
        await self.subscription_manager.initialize()
        
        # Sync slash commands
        try:
            synced = await self.tree.sync()
            logger.info(f"✅ Synced {len(synced)} slash commands")
        except Exception as e:
            logger.error(f"❌ Failed to sync commands: {e}")
        
        # Start background tasks
        self.market_monitor.start()
        self.alert_system.start()
        
        logger.info("🎯 TradeKnowledge Bot ready for systematic trading!")

    async def on_ready(self):
        """Bot ready event handler."""
        logger.info(f"📡 {self.user} is online and ready!")
        logger.info(f"🔗 Connected to {len(self.guilds)} servers")
        
        # Set bot status
        activity = discord.Activity(
            type=discord.ActivityType.watching, 
            name="SPX options | /tk-help"
        )
        await self.change_presence(status=discord.Status.online, activity=activity)

    async def on_command_error(self, ctx, error):
        """Global error handler for bot commands."""
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f"⏰ Command on cooldown. Try again in {error.retry_after:.1f}s")
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f"❌ Missing required argument: {error.param}")
        else:
            logger.error(f"Command error in {ctx.command}: {error}")
            await ctx.send("⚠️ An error occurred processing your command.")

    # Core Options Trading Commands

    @discord.app_commands.command(name="options-scan", description="Scan SPX options for trading opportunities")
    @discord.app_commands.describe(
        strategy="Trading strategy to scan for",
        expiry_range="Days to expiration range (e.g., '30-45')",
        confidence="Neural network confidence level (90, 99, or 99.5)"
    )
    async def options_scan(
        self, 
        interaction: discord.Interaction, 
        strategy: str = "diagonal", 
        expiry_range: str = "30-45",
        confidence: float = 99.0
    ):
        """Scan SPX options for optimal trading setups."""
        await interaction.response.defer()
        
        # Check user subscription and limits
        user_tier = await self.subscription_manager.get_user_tier(interaction.user.id)
        if not await self._check_usage_limits(interaction.user.id, user_tier):
            await interaction.followup.send(
                "📊 **Daily limit reached!** Upgrade to Engineer tier for unlimited scans.\n"
                "Use `/tk-upgrade` to see subscription options."
            )
            return
        
        try:
            # Track command usage
            await self._track_command_usage(interaction.user.id, "options_scan")
            
            # Parse expiry range
            min_dte, max_dte = map(int, expiry_range.split('-'))
            
            # Run options analysis
            analysis_results = await self.options_analyzer.scan_opportunities(
                symbol="SPX",
                strategy=strategy,
                min_dte=min_dte,
                max_dte=max_dte,
                confidence_level=confidence
            )
            
            # Create embed response
            embed = discord.Embed(
                title=f"📊 SPX {strategy.title()} Options Scan",
                description=f"Confidence: {confidence}% | DTE: {expiry_range} days",
                color=0x00ff00 if analysis_results.get('opportunities') else 0xff9900,
                timestamp=datetime.now()
            )
            
            if analysis_results.get('opportunities'):
                for i, opp in enumerate(analysis_results['opportunities'][:5]):
                    embed.add_field(
                        name=f"🎯 Opportunity {i+1}",
                        value=(
                            f"**Strategy**: {opp['strategy']}\n"
                            f"**Entry**: ${opp['entry_price']:.2f}\n"
                            f"**Max Profit**: ${opp['max_profit']:.2f}\n"
                            f"**Probability**: {opp['success_probability']:.1f}%"
                        ),
                        inline=True
                    )
            else:
                embed.add_field(
                    name="📈 Market Status", 
                    value="No optimal opportunities at current confidence level", 
                    inline=False
                )
            
            # Add user tier info
            embed.set_footer(
                text=f"Tier: {user_tier} | Powered by TradeKnowledge Neural Networks"
            )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Options scan error: {e}")
            await interaction.followup.send("⚠️ Error running options scan. Please try again.")

    @discord.app_commands.command(name="diagonal-setup", description="Analyze specific diagonal spread setup")
    @discord.app_commands.describe(
        long_strike="Long option strike price",
        short_strike="Short option strike price", 
        long_expiry="Long option expiry (YYYY-MM-DD)",
        short_expiry="Short option expiry (YYYY-MM-DD)"
    )
    async def diagonal_setup(
        self, 
        interaction: discord.Interaction,
        long_strike: float,
        short_strike: float,
        long_expiry: str,
        short_expiry: str
    ):
        """Analyze a specific diagonal spread configuration."""
        await interaction.response.defer()
        
        user_tier = await self.subscription_manager.get_user_tier(interaction.user.id)
        if not await self._check_usage_limits(interaction.user.id, user_tier):
            await interaction.followup.send("📊 Daily limit reached! Upgrade for unlimited analysis.")
            return
        
        try:
            # Track usage
            await self._track_command_usage(interaction.user.id, "diagonal_setup")
            
            # Analyze diagonal spread
            analysis = await self.options_analyzer.analyze_diagonal_spread(
                symbol="SPX",
                long_strike=long_strike,
                short_strike=short_strike,
                long_expiry=long_expiry,
                short_expiry=short_expiry
            )
            
            # Create detailed embed
            embed = discord.Embed(
                title="🔄 Diagonal Spread Analysis",
                description=f"SPX {long_strike}/{short_strike} diagonal",
                color=0x0099ff,
                timestamp=datetime.now()
            )
            
            embed.add_field(
                name="💰 Financial Metrics",
                value=(
                    f"**Net Debit**: ${analysis['net_debit']:.2f}\n"
                    f"**Max Profit**: ${analysis['max_profit']:.2f}\n"
                    f"**Max Loss**: ${analysis['max_loss']:.2f}\n"
                    f"**ROI**: {analysis['roi']:.1f}%"
                ),
                inline=True
            )
            
            embed.add_field(
                name="📊 Greeks & Risk",
                value=(
                    f"**Delta**: {analysis['delta']:.3f}\n"
                    f"**Gamma**: {analysis['gamma']:.4f}\n"
                    f"**Theta**: {analysis['theta']:.2f}\n"
                    f"**Vega**: {analysis['vega']:.2f}"
                ),
                inline=True
            )
            
            embed.add_field(
                name="🎯 Probabilities",
                value=(
                    f"**Profit Prob**: {analysis['profit_probability']:.1f}%\n"
                    f"**Max Profit**: {analysis['max_profit_probability']:.1f}%\n"
                    f"**Breakeven**: ${analysis['breakeven']:.2f}"
                ),
                inline=True
            )
            
            # Add recommendation
            recommendation_color = 0x00ff00 if analysis['recommendation'] == 'BUY' else 0xff0000
            embed.add_field(
                name="🚀 Recommendation",
                value=f"**{analysis['recommendation']}** - {analysis['reasoning']}",
                inline=False
            )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Diagonal setup analysis error: {e}")
            await interaction.followup.send("⚠️ Error analyzing diagonal spread.")

    @discord.app_commands.command(name="neural-analyze", description="Run neural network analysis with confidence intervals")
    @discord.app_commands.describe(
        symbol="Stock/ETF symbol to analyze",
        confidence="Confidence level (90, 99, or 99.5)",
        timeframe="Analysis timeframe (1d, 1w, 1m)"
    )
    async def neural_analyze(
        self, 
        interaction: discord.Interaction,
        symbol: str = "SPX",
        confidence: float = 99.0,
        timeframe: str = "1w"
    ):
        """Advanced neural network analysis with statistical confidence intervals."""
        await interaction.response.defer()
        
        user_tier = await self.subscription_manager.get_user_tier(interaction.user.id)
        
        # Premium feature check
        if confidence > 90.0 and user_tier == "free":
            await interaction.followup.send(
                "🧠 **Neural Network Premium Feature**\n"
                "99%+ confidence analysis requires Engineer tier or higher.\n"
                "Upgrade with `/tk-upgrade` for advanced neural analysis!"
            )
            return
            
        if not await self._check_usage_limits(interaction.user.id, user_tier):
            await interaction.followup.send("📊 Daily limit reached!")
            return
        
        try:
            await self._track_command_usage(interaction.user.id, "neural_analyze")
            
            # Run neural network analysis
            neural_results = await self.neural_agent.analyze_with_confidence(
                symbol=symbol,
                confidence_level=confidence,
                timeframe=timeframe
            )
            
            embed = discord.Embed(
                title=f"🧠 Neural Network Analysis - {symbol}",
                description=f"Confidence: {confidence}% | Timeframe: {timeframe}",
                color=0x8a2be2,
                timestamp=datetime.now()
            )
            
            embed.add_field(
                name="📈 Price Prediction",
                value=(
                    f"**Direction**: {neural_results['direction']}\n"
                    f"**Target**: ${neural_results['target_price']:.2f}\n"
                    f"**Confidence**: {neural_results['confidence']:.1f}%\n"
                    f"**Timeframe**: {neural_results['timeframe']}"
                ),
                inline=True
            )
            
            embed.add_field(
                name="📊 Statistical Metrics", 
                value=(
                    f"**Model Accuracy**: {neural_results['model_accuracy']:.1f}%\n"
                    f"**Sample Size**: {neural_results['sample_size']:,}\n"
                    f"**R²**: {neural_results['r_squared']:.3f}\n"
                    f"**Sharpe**: {neural_results['sharpe_ratio']:.2f}"
                ),
                inline=True
            )
            
            embed.add_field(
                name="⚠️ Risk Assessment",
                value=(
                    f"**VaR (95%)**: {neural_results['var_95']:.2f}%\n"
                    f"**Max Drawdown**: {neural_results['max_drawdown']:.2f}%\n"
                    f"**Volatility**: {neural_results['volatility']:.2f}%"
                ),
                inline=True
            )
            
            # Add feature access note
            if user_tier == "free":
                embed.set_footer(text="💡 Unlock 99%+ confidence analysis with Engineer tier")
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Neural analysis error: {e}")
            await interaction.followup.send("⚠️ Neural network analysis failed.")

    # Subscription and User Management Commands

    @discord.app_commands.command(name="tk-status", description="Check your TradeKnowledge account status")
    async def tk_status(self, interaction: discord.Interaction):
        """Display user's subscription status and usage statistics."""
        await interaction.response.defer(ephemeral=True)
        
        try:
            user_data = await self.subscription_manager.get_user_data(interaction.user.id)
            
            embed = discord.Embed(
                title="📊 Your TradeKnowledge Account",
                color=0x00ff88,
                timestamp=datetime.now()
            )
            
            embed.add_field(
                name="💎 Subscription Tier",
                value=f"**{user_data['tier'].title()}**",
                inline=True
            )
            
            embed.add_field(
                name="📈 Today's Usage",
                value=f"{user_data['daily_usage']}/{user_data['daily_limit']} commands",
                inline=True
            )
            
            embed.add_field(
                name="📅 Member Since",
                value=user_data['join_date'].strftime("%B %d, %Y"),
                inline=True
            )
            
            if user_data['tier'] != "free":
                embed.add_field(
                    name="💳 Subscription Status",
                    value=f"Active until {user_data['subscription_end'].strftime('%Y-%m-%d')}",
                    inline=False
                )
            
            embed.add_field(
                name="🏆 Total Commands",
                value=f"{user_data['total_commands']:,} commands executed",
                inline=True
            )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Status check error: {e}")
            await interaction.followup.send("⚠️ Error retrieving account status.")

    @discord.app_commands.command(name="tk-upgrade", description="View subscription upgrade options")
    async def tk_upgrade(self, interaction: discord.Interaction):
        """Display subscription tiers and upgrade options."""
        embed = discord.Embed(
            title="🚀 TradeKnowledge Subscription Tiers",
            description="Bloomberg-quality analysis at engineering-friendly prices",
            color=0xffd700,
            timestamp=datetime.now()
        )
        
        embed.add_field(
            name="🆓 Free Tier",
            value=(
                "• 5 daily queries\n"
                "• Basic options scanning\n"
                "• 90% confidence analysis\n"
                "• Community Discord access"
            ),
            inline=True
        )
        
        embed.add_field(
            name="⚙️ Engineer - $49/month",
            value=(
                "• Unlimited queries\n"
                "• Full confidence spectrum (90%/99%/99.5%)\n"
                "• Advanced options strategies\n"
                "• PostgreSQL MCP access\n"
                "• Priority support"
            ),
            inline=True
        )
        
        embed.add_field(
            name="🚀 Rocket Scientist - $149/month",
            value=(
                "• Multi-agent blackboard access\n"
                "• Custom neural network training\n"
                "• Real-time market microstructure\n"
                "• API access for algo trading\n"
                "• Private channels with experts"
            ),
            inline=True
        )
        
        embed.add_field(
            name="🎯 Mission Control - $349/month",
            value=(
                "• White-label Discord bot\n"
                "• Custom agent development\n"
                "• Enterprise data feeds\n"
                "• 1-on-1 consulting sessions\n"
                "• Custom integrations"
            ),
            inline=False
        )
        
        embed.set_footer(text="💡 Compare: Bloomberg Terminal costs $2,000/month")
        
        await interaction.response.send_message(embed=embed)

    # Utility and Background Tasks

    async def _check_usage_limits(self, user_id: int, tier: str) -> bool:
        """Check if user has remaining usage for their tier."""
        usage_data = await self.subscription_manager.get_daily_usage(user_id)
        
        limits = {
            "free": self.free_tier_daily_limit,
            "engineer": self.engineer_tier_limit,
            "rocket_scientist": float('inf'),
            "mission_control": float('inf')
        }
        
        return usage_data['count'] < limits.get(tier, 0)

    async def _track_command_usage(self, user_id: int, command_name: str):
        """Track command usage for analytics and billing."""
        await self.subscription_manager.increment_usage(user_id, command_name)
        self.command_count += 1
        self.active_users.add(user_id)

    @tasks.loop(minutes=15)
    async def market_monitor(self):
        """Background task for market monitoring and alerts."""
        try:
            # Check for significant market moves
            market_data = await self.analyzer.analyze_stock("SPX")
            current_price = float(market_data.market_data.current_price)
            
            # Post alerts to alert channels for subscribed users
            # Implementation depends on alert configuration
            pass
            
        except Exception as e:
            logger.error(f"Market monitor error: {e}")

    @tasks.loop(hours=1)
    async def alert_system(self):
        """Process and send trading alerts to users."""
        try:
            # Check blackboard for new opportunities
            opportunities = await self.blackboard.get_new_opportunities()
            
            for opportunity in opportunities:
                # Send to relevant users based on their subscriptions
                await self._send_opportunity_alert(opportunity)
                
        except Exception as e:
            logger.error(f"Alert system error: {e}")

    async def _send_opportunity_alert(self, opportunity: Dict[str, Any]):
        """Send trading opportunity alert to subscribed users."""
        embed = discord.Embed(
            title="🚨 Trading Opportunity Alert",
            description=opportunity['description'],
            color=0xff6600,
            timestamp=datetime.now()
        )
        
        embed.add_field(
            name="📊 Details",
            value=(
                f"**Symbol**: {opportunity['symbol']}\n"
                f"**Strategy**: {opportunity['strategy']}\n"
                f"**Confidence**: {opportunity['confidence']}%\n"
                f"**Expected Return**: {opportunity['expected_return']:.1f}%"
            ),
            inline=False
        )
        
        # Send to users with appropriate subscription tiers
        # Implementation depends on user notification preferences


# Bot initialization and startup
def create_bot() -> TradeKnowledgeBot:
    """Create and configure the TradeKnowledge Discord bot."""
    return TradeKnowledgeBot()


if __name__ == "__main__":
    # Load bot token from environment
    TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    if not TOKEN:
        logger.error("❌ DISCORD_BOT_TOKEN environment variable not set")
        exit(1)
    
    # Create and run bot
    bot = create_bot()
    bot.run(TOKEN)