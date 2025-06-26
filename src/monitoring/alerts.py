"""
Alert Handlers and Notification System
Handles triggered alerts and sends notifications through various channels
"""

import asyncio
import json
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import structlog

from ..core.config import get_config
from ..core.monitoring_service import Alert

logger = structlog.get_logger(__name__)


@dataclass
class NotificationChannel:
    """Configuration for a notification channel"""

    name: str
    type: str  # email, webhook, slack, etc.
    config: dict[str, Any]
    enabled: bool = True


class EmailNotificationHandler:
    """Handle email notifications for alerts"""

    def __init__(self, smtp_config: dict[str, Any]):
        self.smtp_host = smtp_config.get("host", "localhost")
        self.smtp_port = smtp_config.get("port", 587)
        self.smtp_user = smtp_config.get("username")
        self.smtp_password = smtp_config.get("password")
        self.from_email = smtp_config.get("from_email", "alerts@tradeknowledge.com")
        self.to_emails = smtp_config.get("to_emails", [])
        self.use_tls = smtp_config.get("use_tls", True)

    async def send_alert_notification(self, alert: Alert):
        """Send email notification for an alert"""
        try:
            if not self.to_emails:
                logger.warning("No email recipients configured for alerts")
                return

            subject = f"[TradeKnowledge] {alert.severity.upper()}: {alert.name}"

            # Create email content
            html_content = self._create_alert_email_html(alert)
            text_content = self._create_alert_email_text(alert)

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)

            # Add text and HTML parts
            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            # Send email
            await self._send_email(msg)

            logger.info(
                "Alert email notification sent",
                alert_id=alert.id,
                recipients=len(self.to_emails),
            )

        except Exception as e:
            logger.error(
                "Failed to send email notification", alert_id=alert.id, error=str(e)
            )

    async def _send_email(self, message: MIMEMultipart):
        """Send email using SMTP"""

        def send_smtp():
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(message)

        # Run SMTP operation in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(None, send_smtp)

    def _create_alert_email_html(self, alert: Alert) -> str:
        """Create HTML email content for alert"""
        severity_color = {
            "critical": "#dc3545",
            "warning": "#ffc107",
            "info": "#17a2b8",
        }.get(alert.severity, "#6c757d")

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px;">
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px;">
                <h2 style="color: {severity_color}; margin-top: 0;">
                    🚨 {alert.name}
                </h2>
                
                <div style="background-color: white; padding: 15px; border-radius: 3px; margin: 15px 0;">
                    <h3>Alert Details</h3>
                    <p><strong>Severity:</strong> <span style="color: {severity_color};">{alert.severity.upper()}</span></p>
                    <p><strong>Description:</strong> {alert.description}</p>
                    <p><strong>Condition:</strong> {alert.condition}</p>
                    <p><strong>Threshold:</strong> {alert.threshold}</p>
                    <p><strong>Triggered At:</strong> {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC') if alert.triggered_at else 'N/A'}</p>
                </div>
                
                <div style="background-color: #e9ecef; padding: 10px; border-radius: 3px; margin: 15px 0;">
                    <small>
                        <strong>Alert ID:</strong> {alert.id}<br>
                        <strong>System:</strong> TradeKnowledge Monitoring<br>
                        <strong>Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
                    </small>
                </div>
            </div>
        </body>
        </html>
        """

    def _create_alert_email_text(self, alert: Alert) -> str:
        """Create plain text email content for alert"""
        return f"""
TradeKnowledge Alert: {alert.name}

SEVERITY: {alert.severity.upper()}
DESCRIPTION: {alert.description}
CONDITION: {alert.condition}
THRESHOLD: {alert.threshold}
TRIGGERED AT: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC') if alert.triggered_at else 'N/A'}

Alert ID: {alert.id}
System: TradeKnowledge Monitoring
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

Please check the monitoring dashboard for more details.
        """


class WebhookNotificationHandler:
    """Handle webhook notifications for alerts"""

    def __init__(self, webhook_config: dict[str, Any]):
        self.webhook_url = webhook_config.get("url")
        self.headers = webhook_config.get("headers", {})
        self.timeout = webhook_config.get("timeout", 30)

    async def send_alert_notification(self, alert: Alert):
        """Send webhook notification for an alert"""
        try:
            if not self.webhook_url:
                logger.warning("No webhook URL configured for alerts")
                return

            # Prepare webhook payload
            payload = {
                "alert_id": alert.id,
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity,
                "condition": alert.condition,
                "threshold": alert.threshold,
                "status": alert.status,
                "triggered_at": (
                    alert.triggered_at.isoformat() if alert.triggered_at else None
                ),
                "system": "TradeKnowledge",
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Send webhook
            import aiohttp

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    self.webhook_url, json=payload, headers=self.headers
                ) as response:
                    if response.status < 400:
                        logger.info(
                            "Alert webhook notification sent",
                            alert_id=alert.id,
                            webhook_url=self.webhook_url,
                            status_code=response.status,
                        )
                    else:
                        logger.error(
                            "Webhook notification failed",
                            alert_id=alert.id,
                            status_code=response.status,
                            response_text=await response.text(),
                        )

        except Exception as e:
            logger.error(
                "Failed to send webhook notification", alert_id=alert.id, error=str(e)
            )


class SlackNotificationHandler:
    """Handle Slack notifications for alerts"""

    def __init__(self, slack_config: dict[str, Any]):
        self.webhook_url = slack_config.get("webhook_url")
        self.channel = slack_config.get("channel", "#alerts")
        self.username = slack_config.get("username", "TradeKnowledge Bot")

    async def send_alert_notification(self, alert: Alert):
        """Send Slack notification for an alert"""
        try:
            if not self.webhook_url:
                logger.warning("No Slack webhook URL configured for alerts")
                return

            # Determine emoji and color based on severity
            severity_config = {
                "critical": {"emoji": "🚨", "color": "danger"},
                "warning": {"emoji": "⚠️", "color": "warning"},
                "info": {"emoji": "ℹ️", "color": "good"},
            }

            config = severity_config.get(
                alert.severity, {"emoji": "❓", "color": "#808080"}
            )

            # Create Slack message
            payload = {
                "channel": self.channel,
                "username": self.username,
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": config["color"],
                        "title": f"{config['emoji']} {alert.name}",
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.upper(),
                                "short": True,
                            },
                            {
                                "title": "Condition",
                                "value": alert.condition,
                                "short": True,
                            },
                            {
                                "title": "Threshold",
                                "value": str(alert.threshold),
                                "short": True,
                            },
                            {
                                "title": "Triggered At",
                                "value": (
                                    alert.triggered_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                                    if alert.triggered_at
                                    else "N/A"
                                ),
                                "short": True,
                            },
                        ],
                        "footer": "TradeKnowledge Monitoring",
                        "ts": int(datetime.utcnow().timestamp()),
                    }
                ],
            }

            # Send to Slack
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(
                            "Alert Slack notification sent",
                            alert_id=alert.id,
                            channel=self.channel,
                        )
                    else:
                        logger.error(
                            "Slack notification failed",
                            alert_id=alert.id,
                            status_code=response.status,
                        )

        except Exception as e:
            logger.error(
                "Failed to send Slack notification", alert_id=alert.id, error=str(e)
            )


class FileLogNotificationHandler:
    """Handle file-based alert logging"""

    def __init__(self, log_config: dict[str, Any]):
        self.log_file = Path(log_config.get("file_path", "data/alerts.log"))
        self.max_file_size = log_config.get("max_file_size_mb", 10) * 1024 * 1024
        self.backup_count = log_config.get("backup_count", 5)

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    async def send_alert_notification(self, alert: Alert):
        """Log alert to file"""
        try:
            # Rotate log file if needed
            await self._rotate_log_if_needed()

            # Create log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "alert_id": alert.id,
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity,
                "condition": alert.condition,
                "threshold": alert.threshold,
                "status": alert.status,
                "triggered_at": (
                    alert.triggered_at.isoformat() if alert.triggered_at else None
                ),
            }

            # Write to log file
            def write_log():
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            await asyncio.get_event_loop().run_in_executor(None, write_log)

            logger.debug(
                "Alert logged to file", alert_id=alert.id, log_file=str(self.log_file)
            )

        except Exception as e:
            logger.error("Failed to log alert to file", alert_id=alert.id, error=str(e))

    async def _rotate_log_if_needed(self):
        """Rotate log file if it exceeds size limit"""
        try:
            if not self.log_file.exists():
                return

            if self.log_file.stat().st_size > self.max_file_size:
                # Rotate files
                for i in range(self.backup_count - 1, 0, -1):
                    old_file = self.log_file.with_suffix(f".{i}")
                    new_file = self.log_file.with_suffix(f".{i + 1}")

                    if old_file.exists():
                        old_file.rename(new_file)

                # Move current log to .1
                backup_file = self.log_file.with_suffix(".1")
                self.log_file.rename(backup_file)

                logger.info(
                    "Alert log file rotated",
                    original_size_mb=self.max_file_size / (1024 * 1024),
                )
        except Exception as e:
            logger.warning("Failed to rotate alert log file", error=str(e))


class AlertNotificationManager:
    """Manages alert notifications across multiple channels"""

    def __init__(self):
        self.config = get_config()
        self.handlers: list[Any] = []
        self.channels: dict[str, NotificationChannel] = {}

    async def initialize(self):
        """Initialize notification handlers"""
        try:
            logger.info("Initializing alert notification system")

            # Load notification configuration
            await self._load_notification_config()

            # Initialize handlers based on configuration
            await self._initialize_handlers()

            logger.info(
                "✅ Alert notification system initialized",
                handlers_count=len(self.handlers),
                channels_count=len(self.channels),
            )

        except Exception as e:
            logger.error("Failed to initialize alert notification system", error=str(e))
            # Don't raise - alerts should work even if notifications fail

    async def _load_notification_config(self):
        """Load notification configuration"""
        # Default configuration - can be extended to read from config file
        default_channels = [
            NotificationChannel(
                name="file_log",
                type="file",
                config={
                    "file_path": "data/alerts.log",
                    "max_file_size_mb": 10,
                    "backup_count": 5,
                },
                enabled=True,
            )
        ]

        # Add email if configured
        if hasattr(self.config, "email") and self.config.email.get("enabled", False):
            default_channels.append(
                NotificationChannel(
                    name="email", type="email", config=self.config.email, enabled=True
                )
            )

        # Add webhook if configured
        if hasattr(self.config, "webhook") and self.config.webhook.get(
            "enabled", False
        ):
            default_channels.append(
                NotificationChannel(
                    name="webhook",
                    type="webhook",
                    config=self.config.webhook,
                    enabled=True,
                )
            )

        # Add Slack if configured
        if hasattr(self.config, "slack") and self.config.slack.get("enabled", False):
            default_channels.append(
                NotificationChannel(
                    name="slack", type="slack", config=self.config.slack, enabled=True
                )
            )

        self.channels = {channel.name: channel for channel in default_channels}

    async def _initialize_handlers(self):
        """Initialize notification handlers"""
        for channel in self.channels.values():
            if not channel.enabled:
                continue

            try:
                if channel.type == "email":
                    handler = EmailNotificationHandler(channel.config)
                    self.handlers.append(handler)

                elif channel.type == "webhook":
                    handler = WebhookNotificationHandler(channel.config)
                    self.handlers.append(handler)

                elif channel.type == "slack":
                    handler = SlackNotificationHandler(channel.config)
                    self.handlers.append(handler)

                elif channel.type == "file":
                    handler = FileLogNotificationHandler(channel.config)
                    self.handlers.append(handler)

                logger.info(
                    "Initialized notification handler",
                    channel_name=channel.name,
                    channel_type=channel.type,
                )

            except Exception as e:
                logger.error(
                    "Failed to initialize notification handler",
                    channel_name=channel.name,
                    error=str(e),
                )

    async def handle_alert(self, alert: Alert):
        """Handle an alert by sending notifications through all configured channels"""
        if not self.handlers:
            logger.warning("No notification handlers configured")
            return

        # Send notifications in parallel
        notification_tasks = []
        for handler in self.handlers:
            task = asyncio.create_task(handler.send_alert_notification(alert))
            notification_tasks.append(task)

        # Wait for all notifications to complete (with individual error handling)
        await asyncio.gather(*notification_tasks, return_exceptions=True)

        logger.info(
            "Alert notifications processed",
            alert_id=alert.id,
            handlers_count=len(self.handlers),
        )

    def add_custom_handler(self, handler: Any):
        """Add a custom notification handler"""
        if hasattr(handler, "send_alert_notification"):
            self.handlers.append(handler)
            logger.info(
                "Added custom notification handler", handler_type=type(handler).__name__
            )
        else:
            raise ValueError("Handler must implement send_alert_notification method")


# Global notification manager
notification_manager = AlertNotificationManager()


async def get_notification_manager() -> AlertNotificationManager:
    """Get the global notification manager"""
    return notification_manager


async def initialize_alert_notifications():
    """Initialize alert notifications globally"""
    await notification_manager.initialize()


# Default alert handler function for the monitoring service
async def default_alert_handler(alert: Alert):
    """Default alert handler that sends notifications"""
    await notification_manager.handle_alert(alert)
