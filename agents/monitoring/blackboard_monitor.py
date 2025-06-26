#!/usr/bin/env python3
"""
Agent Blackboard Monitoring and Efficiency Dashboard
Real-time monitoring of token usage, performance, and optimization opportunities
"""

import asyncio
import time
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from influx_blackboard import get_blackboard

@dataclass
class MonitoringAlert:
    """Monitoring alert definition"""
    severity: str  # low, medium, high, critical
    category: str
    message: str
    agent: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class BlackboardMonitor:
    """Real-time monitoring system for agent blackboard"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent / "config" / "blackboard_influx.yaml"
        self.config = self._load_config()
        self.blackboard = get_blackboard()
        self.alerts = []
        self.monitoring_active = False
        
        # Monitoring thresholds
        self.thresholds = self.config.get('monitoring', {})
        
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        if not self.config_path.exists():
            return {}
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start continuous monitoring"""
        print("🚀 Starting Blackboard Monitoring System")
        print("=" * 60)
        
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                await self._check_all_metrics()
                await self._display_dashboard()
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
    
    async def _check_all_metrics(self):
        """Check all monitoring metrics and generate alerts"""
        current_time = time.time()
        
        # Check token usage
        await self._check_token_usage()
        
        # Check performance metrics
        await self._check_performance_metrics()
        
        # Check agent health
        await self._check_agent_health()
        
        # Check optimization opportunities
        await self._check_optimization_opportunities()
        
        # Clean old alerts
        self._clean_old_alerts(max_age=3600)  # 1 hour
    
    async def _check_token_usage(self):
        """Monitor token usage patterns"""
        if not self.blackboard.query_api:
            return
        
        # Query recent token usage
        query = '''
        from(bucket: "metrics")
          |> range(start: -1h)
          |> filter(fn: (r) => r._measurement == "metrics")
          |> group(columns: ["agent"])
          |> sum(column: "tokens_used")
        '''
        
        try:
            tables = self.blackboard.query_api.query(query)
            
            high_usage_threshold = self.thresholds.get('high_token_usage', 1000)
            
            for table in tables:
                for record in table.records:
                    agent = record.values.get('agent')
                    tokens_used = record.values.get('_value', 0)
                    
                    if tokens_used > high_usage_threshold:
                        alert = MonitoringAlert(
                            severity="high",
                            category="token_usage",
                            message=f"High token usage detected for {agent}: {tokens_used} tokens in last hour",
                            agent=agent,
                            metric_value=tokens_used,
                            threshold=high_usage_threshold
                        )
                        self.alerts.append(alert)
                        
        except Exception as e:
            print(f"⚠️ Error checking token usage: {e}")
    
    async def _check_performance_metrics(self):
        """Monitor agent performance metrics"""
        if not self.blackboard.query_api:
            return
        
        # Query recent performance
        query = '''
        from(bucket: "metrics")
          |> range(start: -30m)
          |> filter(fn: (r) => r._measurement == "metrics")
          |> group(columns: ["agent"])
          |> mean(column: "exec_time")
        '''
        
        try:
            tables = self.blackboard.query_api.query(query)
            
            slow_threshold = self.thresholds.get('slow_execution', 5.0)
            
            for table in tables:
                for record in table.records:
                    agent = record.values.get('agent')
                    avg_exec_time = record.values.get('_value', 0)
                    
                    if avg_exec_time > slow_threshold:
                        alert = MonitoringAlert(
                            severity="medium",
                            category="performance",
                            message=f"Slow execution detected for {agent}: {avg_exec_time:.2f}s average",
                            agent=agent,
                            metric_value=avg_exec_time,
                            threshold=slow_threshold
                        )
                        self.alerts.append(alert)
                        
        except Exception as e:
            print(f"⚠️ Error checking performance metrics: {e}")
    
    async def _check_agent_health(self):
        """Monitor agent health and activity"""
        if not self.blackboard.query_api:
            return
        
        # Check for recent activity
        query = '''
        from(bucket: "metrics")
          |> range(start: -10m)
          |> filter(fn: (r) => r._measurement == "metrics")
          |> group(columns: ["agent"])
          |> count()
        '''
        
        try:
            tables = self.blackboard.query_api.query(query)
            active_agents = set()
            
            for table in tables:
                for record in table.records:
                    agent = record.values.get('agent')
                    if agent:
                        active_agents.add(agent)
            
            # Expected agents
            expected_agents = {'R', 'M', 'E'}  # Researcher, Mastermind, Executor
            inactive_agents = expected_agents - active_agents
            
            for agent in inactive_agents:
                alert = MonitoringAlert(
                    severity="medium",
                    category="agent_health",
                    message=f"Agent {agent} has been inactive for >10 minutes",
                    agent=agent
                )
                self.alerts.append(alert)
                
        except Exception as e:
            print(f"⚠️ Error checking agent health: {e}")
    
    async def _check_optimization_opportunities(self):
        """Check for optimization opportunities"""
        if not self.blackboard.query_api:
            return
        
        # Query optimization suggestions
        query = '''
        from(bucket: "optimizations")
          |> range(start: -1h)
          |> filter(fn: (r) => r._measurement == "optimizations")
          |> filter(fn: (r) => r.implemented == "false")
          |> count()
        '''
        
        try:
            tables = self.blackboard.query_api.query(query)
            
            for table in tables:
                for record in table.records:
                    count = record.values.get('_value', 0)
                    
                    if count > 5:  # More than 5 pending optimizations
                        alert = MonitoringAlert(
                            severity="low",
                            category="optimization",
                            message=f"{count} optimization opportunities available",
                            metric_value=count
                        )
                        self.alerts.append(alert)
                        
        except Exception as e:
            print(f"⚠️ Error checking optimization opportunities: {e}")
    
    def _clean_old_alerts(self, max_age: float):
        """Remove old alerts"""
        cutoff_time = time.time() - max_age
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    async def _display_dashboard(self):
        """Display real-time dashboard"""
        # Clear screen
        print("\033[2J\033[H")
        
        print("🖥️  AGENT BLACKBOARD MONITORING DASHBOARD")
        print("=" * 80)
        print(f"⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Get efficiency report
        try:
            report = await self.blackboard.generate_efficiency_report(period_hours=1)
            
            if "error" not in report:
                await self._display_efficiency_summary(report)
            else:
                print(f"⚠️ Could not generate efficiency report: {report['error']}")
        except Exception as e:
            print(f"⚠️ Dashboard error: {e}")
        
        # Display alerts
        await self._display_alerts()
        
        # Display agent status
        await self._display_agent_status()
        
        print("-" * 80)
        print("📊 Press Ctrl+C to stop monitoring")
    
    async def _display_efficiency_summary(self, report: Dict[str, Any]):
        """Display efficiency summary"""
        print("📈 EFFICIENCY SUMMARY (Last Hour)")
        print("-" * 40)
        print(f"Total Tokens Used: {report.get('total_tokens', 0):,}")
        print(f"Total Operations: {report.get('total_operations', 0):,}")
        
        if report.get('total_operations', 0) > 0:
            avg_tokens = report['total_tokens'] / report['total_operations']
            print(f"Average Tokens/Operation: {avg_tokens:.1f}")
        
        print()
        
        # Agent breakdown
        agents_data = report.get('agents', {})
        if agents_data:
            print("🤖 AGENT PERFORMANCE")
            print("-" * 40)
            
            for agent, stats in agents_data.items():
                efficiency_icon = self._get_efficiency_icon(stats.get('efficiency_score', 0))
                success_icon = "✅" if stats.get('success_rate', 0) > 0.9 else "⚠️" if stats.get('success_rate', 0) > 0.7 else "❌"
                
                print(f"{agent}: {efficiency_icon} {success_icon}")
                print(f"  Tokens: {stats.get('tokens_used', 0):,} | Ops: {stats.get('operations', 0)} | Success: {stats.get('success_rate', 0):.1%}")
                print(f"  Avg Time: {stats.get('avg_exec_time', 0):.2f}s | Efficiency: {stats.get('efficiency_score', 0):.3f}")
                print()
    
    def _get_efficiency_icon(self, score: float) -> str:
        """Get icon based on efficiency score"""
        if score >= 0.8:
            return "🟢"
        elif score >= 0.6:
            return "🟡"
        else:
            return "🔴"
    
    async def _display_alerts(self):
        """Display current alerts"""
        if not self.alerts:
            print("✅ NO ACTIVE ALERTS")
            print()
            return
        
        print("🚨 ACTIVE ALERTS")
        print("-" * 40)
        
        # Group alerts by severity
        alerts_by_severity = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        for alert in self.alerts[-10:]:  # Show last 10 alerts
            alerts_by_severity[alert.severity].append(alert)
        
        for severity in ['critical', 'high', 'medium', 'low']:
            alerts = alerts_by_severity[severity]
            if alerts:
                severity_icon = {
                    'critical': '🚨',
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }[severity]
                
                print(f"{severity_icon} {severity.upper()} ({len(alerts)})")
                for alert in alerts[-3:]:  # Show last 3 of each severity
                    time_str = datetime.fromtimestamp(alert.timestamp).strftime('%H:%M:%S')
                    print(f"  [{time_str}] {alert.message}")
                print()
    
    async def _display_agent_status(self):
        """Display real-time agent status"""
        print("👥 AGENT STATUS")
        print("-" * 40)
        
        # Query recent activity for each agent
        agents = ['R', 'M', 'E']
        
        for agent in agents:
            try:
                context = await self.blackboard.get_agent_context(agent, lookback_hours=1)
                
                if "error" in context:
                    status_icon = "❓"
                    status_text = "Unknown"
                else:
                    # Determine status based on activity
                    last_activity = context.get('last_activity', '')
                    tasks_total = context.get('tasks_total', 0)
                    success_rate = context.get('success_rate', 0)
                    
                    if tasks_total > 0 and success_rate > 0.8:
                        status_icon = "🟢"
                        status_text = "Active"
                    elif tasks_total > 0:
                        status_icon = "🟡"
                        status_text = "Issues"
                    else:
                        status_icon = "⚪"
                        status_text = "Idle"
                
                agent_name = {'R': 'Researcher', 'M': 'Mastermind', 'E': 'Executor'}[agent]
                
                print(f"{status_icon} {agent_name}: {status_text}")
                
                if "error" not in context:
                    print(f"   Tasks: {context.get('tasks_total', 0)} | Completed: {context.get('tasks_completed', 0)}")
                    print(f"   Avg Tokens: {context.get('avg_tokens', 0):.0f} | Success Rate: {context.get('success_rate', 0):.1%}")
                
            except Exception as e:
                print(f"❌ {agent}: Error getting status - {e}")
        
        print()

async def generate_efficiency_report():
    """Generate and display detailed efficiency report"""
    print("📊 GENERATING DETAILED EFFICIENCY REPORT")
    print("=" * 60)
    
    monitor = BlackboardMonitor()
    
    # Generate reports for different time periods
    periods = [
        (1, "Last Hour"),
        (6, "Last 6 Hours"), 
        (24, "Last 24 Hours")
    ]
    
    for hours, title in periods:
        print(f"\n{title}")
        print("-" * 30)
        
        try:
            report = await monitor.blackboard.generate_efficiency_report(period_hours=hours)
            
            if "error" in report:
                print(f"❌ Error: {report['error']}")
                continue
            
            print(f"Total Tokens: {report.get('total_tokens', 0):,}")
            print(f"Total Operations: {report.get('total_operations', 0):,}")
            
            if report.get('total_operations', 0) > 0:
                avg_tokens = report['total_tokens'] / report['total_operations']
                print(f"Efficiency: {avg_tokens:.1f} tokens/operation")
            
            # Agent performance
            agents_data = report.get('agents', {})
            if agents_data:
                print("\nAgent Performance:")
                
                for agent, stats in agents_data.items():
                    agent_name = {'R': 'Researcher', 'M': 'Mastermind', 'E': 'Executor'}.get(agent, agent)
                    efficiency = stats.get('efficiency_score', 0)
                    success = stats.get('success_rate', 0)
                    
                    print(f"  {agent_name}: {efficiency:.3f} efficiency, {success:.1%} success")
        
        except Exception as e:
            print(f"❌ Error generating report for {title}: {e}")
    
    monitor.blackboard.close()

async def main():
    """Main monitoring function"""
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        await generate_efficiency_report()
    else:
        monitor = BlackboardMonitor()
        try:
            await monitor.start_monitoring()
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
        finally:
            monitor.blackboard.close()

if __name__ == "__main__":
    print("🖥️  Agent Blackboard Monitor")
    print("Usage:")
    print("  python blackboard_monitor.py       - Start real-time monitoring")
    print("  python blackboard_monitor.py report - Generate efficiency report")
    print()
    
    asyncio.run(main())