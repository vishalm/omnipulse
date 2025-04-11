"""
Alert management module for OmniPulse monitoring dashboard.

This module provides functionality for creating, managing, and triggering alerts
based on metric thresholds and conditions. It supports various notification types
and alert severity levels.
"""

import streamlit as st
import pandas as pd
import numpy as np
import smtplib
import requests
import json
import logging
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged" 
    RESOLVED = "resolved"


class Alert:
    """Class representing an alert instance"""
    
    def __init__(
        self,
        alert_id: str,
        metric_name: str,
        severity: AlertSeverity,
        threshold: float,
        current_value: float,
        message: str,
        component: str,
        timestamp: datetime = None
    ):
        """
        Initialize an alert.
        
        Args:
            alert_id: Unique identifier for the alert
            metric_name: Name of the metric that triggered the alert
            severity: Severity level of the alert
            threshold: Threshold value that was exceeded
            current_value: Current value of the metric
            message: Alert message
            component: Component that triggered the alert (e.g., 'ollama', 'system')
            timestamp: Alert creation time (defaults to current time)
        """
        self.alert_id = alert_id
        self.metric_name = metric_name
        self.severity = severity
        self.threshold = threshold
        self.current_value = current_value
        self.message = message
        self.component = component
        self.timestamp = timestamp or datetime.now()
        self.status = AlertStatus.ACTIVE
        self.acknowledged_at = None
        self.resolved_at = None
        self.acknowledged_by = None
        self.resolution_message = None
    
    def acknowledge(self, user: str = "admin"):
        """
        Acknowledge the alert.
        
        Args:
            user: User who acknowledged the alert
        """
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now()
        self.acknowledged_by = user
    
    def resolve(self, resolution_message: str = ""):
        """
        Resolve the alert.
        
        Args:
            resolution_message: Message describing the resolution
        """
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()
        self.resolution_message = resolution_message
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert alert to dictionary for storage or transmission.
        
        Returns:
            Dictionary representation of the alert
        """
        return {
            "alert_id": self.alert_id,
            "metric_name": self.metric_name,
            "severity": self.severity.value,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "message": self.message,
            "component": self.component,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolution_message": self.resolution_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """
        Create an alert instance from a dictionary.
        
        Args:
            data: Dictionary with alert data
            
        Returns:
            Alert instance
        """
        alert = cls(
            alert_id=data["alert_id"],
            metric_name=data["metric_name"],
            severity=AlertSeverity(data["severity"]),
            threshold=data["threshold"],
            current_value=data["current_value"],
            message=data["message"],
            component=data["component"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        
        alert.status = AlertStatus(data["status"])
        
        if data["acknowledged_at"]:
            alert.acknowledged_at = datetime.fromisoformat(data["acknowledged_at"])
        
        if data["resolved_at"]:
            alert.resolved_at = datetime.fromisoformat(data["resolved_at"])
        
        alert.acknowledged_by = data["acknowledged_by"]
        alert.resolution_message = data["resolution_message"]
        
        return alert


class AlertNotifier:
    """Class for sending alert notifications via various channels"""
    
    @staticmethod
    def send_email(
        alert: Alert,
        recipient: str,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str
    ) -> bool:
        """
        Send an email notification for an alert.
        
        Args:
            alert: Alert instance to send
            recipient: Email recipient
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender email address
            sender_password: Sender email password
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient
            msg['Subject'] = f"[{alert.severity.value.upper()}] OmniPulse Alert: {alert.metric_name}"
            
            # Create message body
            body = f"""
            <html>
            <body>
                <h2>OmniPulse Alert</h2>
                <p><strong>Metric:</strong> {alert.metric_name}</p>
                <p><strong>Component:</strong> {alert.component}</p>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Current Value:</strong> {alert.current_value}</p>
                <p><strong>Threshold:</strong> {alert.threshold}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Message:</strong> {alert.message}</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Connect to SMTP server and send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {recipient} for {alert.metric_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    @staticmethod
    def send_webhook(
        alert: Alert,
        webhook_url: str,
        custom_headers: Dict[str, str] = None
    ) -> bool:
        """
        Send a webhook notification for an alert.
        
        Args:
            alert: Alert instance to send
            webhook_url: Webhook URL
            custom_headers: Optional custom headers
            
        Returns:
            True if successful, False otherwise
        """
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            if custom_headers:
                headers.update(custom_headers)
            
            payload = {
                "alert": alert.to_dict(),
                "dashboard_url": st.experimental_get_query_params().get("dashboard_url", [""])[0]
            }
            
            response = requests.post(
                webhook_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=5
            )
            
            if response.status_code < 400:
                logger.info(f"Webhook alert sent to {webhook_url} for {alert.metric_name}")
                return True
            else:
                logger.error(f"Webhook alert failed with status {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class AlertManager:
    """Class for managing alerts and alert configurations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the alert manager.
        
        Args:
            config: Configuration dictionary containing alert settings
        """
        self.config = config
        self.active_alerts = []
        self.alert_history = []
        
        # Initialize alert store in session state if it doesn't exist
        if "alerts" not in st.session_state:
            st.session_state.alerts = {
                "active": [],
                "history": []
            }
    
    def check_threshold(
        self,
        metric_name: str,
        current_value: float,
        component: str,
        thresholds: Dict[str, float],
        message_template: str = "{metric_name} has exceeded the {severity} threshold ({current_value:.2f} > {threshold:.2f})"
    ) -> Optional[Alert]:
        """
        Check if a metric exceeds its thresholds and create an alert if needed.
        
        Args:
            metric_name: Name of the metric to check
            current_value: Current value of the metric
            component: Component the metric belongs to (e.g., 'ollama', 'system')
            thresholds: Dictionary with thresholds for different severity levels
            message_template: Template for alert message
            
        Returns:
            Alert instance if threshold was exceeded, None otherwise
        """
        # Check if alerts are enabled for this component
        component_config = self.config.get(component, {})
        if not component_config.get("alerts_enabled", False):
            return None
        
        # Get metric-specific thresholds
        metric_thresholds = component_config.get("thresholds", {}).get(metric_name, None)
        
        # Use provided thresholds if metric-specific ones are not configured
        if not metric_thresholds:
            metric_thresholds = thresholds
        
        # Check critical threshold first
        if "critical" in metric_thresholds and current_value >= metric_thresholds["critical"]:
            severity = AlertSeverity.CRITICAL
            threshold = metric_thresholds["critical"]
        # Then check warning threshold
        elif "warning" in metric_thresholds and current_value >= metric_thresholds["warning"]:
            severity = AlertSeverity.WARNING
            threshold = metric_thresholds["warning"]
        # Then check info threshold
        elif "info" in metric_thresholds and current_value >= metric_thresholds["info"]:
            severity = AlertSeverity.INFO
            threshold = metric_thresholds["info"]
        else:
            # No thresholds exceeded
            return None
        
        # Check if alert for this metric already exists and is active
        for alert in st.session_state.alerts["active"]:
            if (alert["metric_name"] == metric_name and 
                alert["component"] == component and 
                alert["status"] == AlertStatus.ACTIVE.value):
                
                # Check if the severity has changed
                if alert["severity"] != severity.value:
                    # Update the existing alert
                    alert["severity"] = severity.value
                    alert["threshold"] = threshold
                    alert["current_value"] = current_value
                    alert["message"] = message_template.format(
                        metric_name=metric_name,
                        severity=severity.value,
                        current_value=current_value,
                        threshold=threshold
                    )
                    
                    logger.info(f"Updated {severity.value} alert for {metric_name} (value: {current_value})")
                
                # Alert already exists, no need to create a new one
                return None
        
        # Create a new alert
        alert_id = f"{component}_{metric_name}_{int(time.time())}"
        message = message_template.format(
            metric_name=metric_name,
            severity=severity.value,
            current_value=current_value,
            threshold=threshold
        )
        
        new_alert = Alert(
            alert_id=alert_id,
            metric_name=metric_name,
            severity=severity,
            threshold=threshold,
            current_value=current_value,
            message=message,
            component=component
        )
        
        # Store the alert in session state
        st.session_state.alerts["active"].append(new_alert.to_dict())
        
        # Send notifications if configured
        self._send_notifications(new_alert)
        
        logger.info(f"Created new {severity.value} alert for {metric_name} (value: {current_value})")
        
        return new_alert
    
    def _send_notifications(self, alert: Alert) -> None:
        """
        Send notifications for a new alert based on configuration.
        
        Args:
            alert: The alert to send notifications for
        """
        # Get notification configuration
        notification_config = self.config.get("notifications", {})
        
        # Check if notifications are enabled for this severity
        severity_config = notification_config.get(alert.severity.value, {})
        if not severity_config.get("enabled", False):
            return
        
        # Send email notification if configured
        email_config = notification_config.get("email", {})
        if email_config.get("enabled", False):
            recipient = email_config.get("recipient")
            smtp_server = email_config.get("smtp_server")
            smtp_port = email_config.get("smtp_port")
            sender_email = email_config.get("sender_email")
            sender_password = email_config.get("sender_password")
            
            if all([recipient, smtp_server, smtp_port, sender_email, sender_password]):
                AlertNotifier.send_email(
                    alert=alert,
                    recipient=recipient,
                    smtp_server=smtp_server,
                    smtp_port=smtp_port,
                    sender_email=sender_email,
                    sender_password=sender_password
                )
        
        # Send webhook notification if configured
        webhook_config = notification_config.get("webhook", {})
        if webhook_config.get("enabled", False):
            webhook_url = webhook_config.get("url")
            custom_headers = webhook_config.get("headers", {})
            
            if webhook_url:
                AlertNotifier.send_webhook(
                    alert=alert,
                    webhook_url=webhook_url,
                    custom_headers=custom_headers
                )
    
    def acknowledge_alert(self, alert_id: str, user: str = "admin") -> bool:
        """
        Acknowledge an active alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            user: User who is acknowledging the alert
            
        Returns:
            True if successful, False otherwise
        """
        for i, alert_dict in enumerate(st.session_state.alerts["active"]):
            if alert_dict["alert_id"] == alert_id:
                # Convert dict to Alert object
                alert = Alert.from_dict(alert_dict)
                
                # Acknowledge the alert
                alert.acknowledge(user)
                
                # Update the alert in session state
                st.session_state.alerts["active"][i] = alert.to_dict()
                
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
        
        logger.warning(f"Could not find alert {alert_id} to acknowledge")
        return False
    
    def resolve_alert(self, alert_id: str, resolution_message: str = "") -> bool:
        """
        Resolve an active alert and move it to history.
        
        Args:
            alert_id: ID of the alert to resolve
            resolution_message: Optional message describing the resolution
            
        Returns:
            True if successful, False otherwise
        """
        for i, alert_dict in enumerate(st.session_state.alerts["active"]):
            if alert_dict["alert_id"] == alert_id:
                # Convert dict to Alert object
                alert = Alert.from_dict(alert_dict)
                
                # Resolve the alert
                alert.resolve(resolution_message)
                
                # Move the alert from active to history
                resolved_alert = alert.to_dict()
                st.session_state.alerts["active"].pop(i)
                st.session_state.alerts["history"].append(resolved_alert)
                
                logger.info(f"Alert {alert_id} resolved: {resolution_message}")
                return True
        
        logger.warning(f"Could not find alert {alert_id} to resolve")
        return False
    
    def auto_resolve_alerts(self, component: str, metric_name: str, current_value: float) -> None:
        """
        Automatically resolve alerts when the metric returns to normal.
        
        Args:
            component: Component the metric belongs to
            metric_name: Name of the metric
            current_value: Current value of the metric
        """
        # Get component configuration
        component_config = self.config.get(component, {})
        metric_thresholds = component_config.get("thresholds", {}).get(metric_name, {})
        
        # If no thresholds defined, cannot auto-resolve
        if not metric_thresholds:
            return
        
        # Determine the lowest threshold for this metric
        lowest_threshold = min(
            metric_thresholds.get("info", float('inf')),
            metric_thresholds.get("warning", float('inf')),
            metric_thresholds.get("critical", float('inf'))
        )
        
        # If current value is below the lowest threshold, resolve any active alerts
        if current_value < lowest_threshold:
            for i, alert_dict in enumerate(st.session_state.alerts["active"]):
                if (alert_dict["metric_name"] == metric_name and 
                    alert_dict["component"] == component and 
                    alert_dict["status"] in [AlertStatus.ACTIVE.value, AlertStatus.ACKNOWLEDGED.value]):
                    
                    # Resolve the alert
                    self.resolve_alert(
                        alert_dict["alert_id"],
                        f"Automatically resolved: value returned to normal ({current_value:.2f} < {lowest_threshold:.2f})"
                    )
    
    def get_active_alerts(
        self,
        component: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get active alerts, optionally filtered by component and severity.
        
        Args:
            component: Optional component filter
            severity: Optional severity filter
            limit: Maximum number of alerts to return
            
        Returns:
            List of active alerts as dictionaries
        """
        active_alerts = st.session_state.alerts["active"]
        
        # Apply filters
        if component:
            active_alerts = [a for a in active_alerts if a["component"] == component]
        
        if severity:
            active_alerts = [a for a in active_alerts if a["severity"] == severity.value]
        
        # Sort by timestamp (newest first) and apply limit
        active_alerts.sort(key=lambda a: a["timestamp"], reverse=True)
        return active_alerts[:limit]
    
    def get_alert_history(
        self,
        component: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get alert history, optionally filtered by various criteria.
        
        Args:
            component: Optional component filter
            severity: Optional severity filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of alerts to return
            
        Returns:
            List of historical alerts as dictionaries
        """
        alert_history = st.session_state.alerts["history"]
        
        # Apply filters
        if component:
            alert_history = [a for a in alert_history if a["component"] == component]
        
        if severity:
            alert_history = [a for a in alert_history if a["severity"] == severity.value]
        
        if start_time:
            alert_history = [a for a in alert_history if datetime.fromisoformat(a["timestamp"]) >= start_time]
        
        if end_time:
            alert_history = [a for a in alert_history if datetime.fromisoformat(a["timestamp"]) <= end_time]
        
        # Sort by timestamp (newest first) and apply limit
        alert_history.sort(key=lambda a: a["timestamp"], reverse=True)
        return alert_history[:limit]
    
    def render_alert_widget(self, location: str = "sidebar") -> None:
        """
        Render an alert widget in the Streamlit interface.
        
        Args:
            location: Where to render the widget ('sidebar' or 'main')
        """
        active_alerts = self.get_active_alerts()
        
        if location == "sidebar":
            container = st.sidebar
        else:
            container = st
        
        with container:
            if active_alerts:
                st.warning(f"🚨 {len(active_alerts)} Active Alerts")
                
                with st.expander("View Alerts"):
                    for alert in active_alerts[:5]:  # Show only the top 5 in the widget
                        severity_color = {
                            "info": "blue",
                            "warning": "orange",
                            "critical": "red"
                        }.get(alert["severity"], "gray")
                        
                        st.markdown(
                            f"<div style='padding: 10px; border-left: 4px solid {severity_color}; margin-bottom: 10px;'>"
                            f"<strong>{alert['metric_name']}</strong> ({alert['component']})<br>"
                            f"{alert['message']}<br>"
                            f"<small>{alert['timestamp']}</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    
                    if len(active_alerts) > 5:
                        st.text(f"+ {len(active_alerts) - 5} more alerts...")
                    
                    if st.button("View All Alerts"):
                        st.session_state.show_alerts_page = True
            else:
                st.success("✅ No Active Alerts")


def check_system_thresholds(
    alert_manager: AlertManager,
    system_metrics: Dict[str, float]
) -> None:
    """
    Check system metrics against thresholds and create alerts if needed.
    
    Args:
        alert_manager: AlertManager instance
        system_metrics: Dictionary of system metrics
    """
    # CPU usage
    if "cpu_percent" in system_metrics:
        alert_manager.check_threshold(
            metric_name="cpu_usage",
            current_value=system_metrics["cpu_percent"],
            component="system",
            thresholds={
                "info": 70,
                "warning": 85,
                "critical": 95
            },
            message_template="CPU usage is high: {current_value:.1f}% (threshold: {threshold:.1f}%)"
        )
        
        # Also check for auto-resolution
        alert_manager.auto_resolve_alerts(
            component="system",
            metric_name="cpu_usage",
            current_value=system_metrics["cpu_percent"]
        )
    
    # Memory usage
    if "memory_percent" in system_metrics:
        alert_manager.check_threshold(
            metric_name="memory_usage",
            current_value=system_metrics["memory_percent"],
            component="system",
            thresholds={
                "info": 70,
                "warning": 85,
                "critical": 95
            },
            message_template="Memory usage is high: {current_value:.1f}% (threshold: {threshold:.1f}%)"
        )
        
        # Also check for auto-resolution
        alert_manager.auto_resolve_alerts(
            component="system",
            metric_name="memory_usage",
            current_value=system_metrics["memory_percent"]
        )
    
    # Disk usage
    if "disk_percent" in system_metrics:
        alert_manager.check_threshold(
            metric_name="disk_usage",
            current_value=system_metrics["disk_percent"],
            component="system",
            thresholds={
                "info": 70,
                "warning": 85,
                "critical": 95
            },
            message_template="Disk usage is high: {current_value:.1f}% (threshold: {threshold:.1f}%)"
        )
        
        # Also check for auto-resolution
        alert_manager.auto_resolve_alerts(
            component="system",
            metric_name="disk_usage",
            current_value=system_metrics["disk_percent"]
        )


def check_ollama_thresholds(
    alert_manager: AlertManager,
    ollama_metrics: Dict[str, float]
) -> None:
    """
    Check Ollama metrics against thresholds and create alerts if needed.
    
    Args:
        alert_manager: AlertManager instance
        ollama_metrics: Dictionary of Ollama metrics
    """
    # Request latency
    if "avg_request_time" in ollama_metrics:
        alert_manager.check_threshold(
            metric_name="request_latency",
            current_value=ollama_metrics["avg_request_time"],
            component="ollama",
            thresholds={
                "info": 1.0,  # 1 second
                "warning": 3.0,  # 3 seconds
                "critical": 10.0  # 10 seconds
            },
            message_template="Ollama request latency is high: {current_value:.2f}s (threshold: {threshold:.2f}s)"
        )
        
        # Also check for auto-resolution
        alert_manager.auto_resolve_alerts(
            component="ollama",
            metric_name="request_latency",
            current_value=ollama_metrics["avg_request_time"]
        )
    
    # Error rate
    if "error_rate" in ollama_metrics:
        alert_manager.check_threshold(
            metric_name="error_rate",
            current_value=ollama_metrics["error_rate"],
            component="ollama",
            thresholds={
                "info": 1.0,  # 1%
                "warning": 5.0,  # 5%
                "critical": 10.0  # 10%
            },
            message_template="Ollama error rate is high: {current_value:.2f}% (threshold: {threshold:.2f}%)"
        )
        
        # Also check for auto-resolution
        alert_manager.auto_resolve_alerts(
            component="ollama",
            metric_name="error_rate",
            current_value=ollama_metrics["error_rate"]
        )
    
    # Memory usage (if available)
    if "memory_usage" in ollama_metrics:
        alert_manager.check_threshold(
            metric_name="memory_usage",
            current_value=ollama_metrics["memory_usage"],
            component="ollama",
            thresholds={
                "info": 70,  # 70%
                "warning": 85,  # 85%
                "critical": 95  # 95%
            },
            message_template="Ollama memory usage is high: {current_value:.1f}% (threshold: {threshold:.1f}%)"
        )
        
        # Also check for auto-resolution
        alert_manager.auto_resolve_alerts(
            component="ollama",
            metric_name="memory_usage",
            current_value=ollama_metrics["memory_usage"]
        )


def check_python_thresholds(
    alert_manager: AlertManager,
    python_metrics: Dict[str, Dict[str, float]]
) -> None:
    """
    Check Python application metrics against thresholds and create alerts if needed.
    
    Args:
        alert_manager: AlertManager instance
        python_metrics: Dictionary of Python application metrics by process name
    """
    for process_name, metrics in python_metrics.items():
        # Memory usage
        if "memory_percent" in metrics:
            alert_manager.check_threshold(
                metric_name=f"{process_name}_memory_usage",
                current_value=metrics["memory_percent"],
                component="python",
                thresholds={
                    "info": 15,  # 15%
                    "warning": 25,  # 25%
                    "critical": 40  # 40%
                },
                message_template=f"Python process '{process_name}' memory usage is high: {{current_value:.1f}}% (threshold: {{threshold:.1f}}%)"
            )
            
            # Also check for auto-resolution
            alert_manager.auto_resolve_alerts(
                component="python",
                metric_name=f"{process_name}_memory_usage",
                current_value=metrics["memory_percent"]
            )
        
        # CPU usage
        if "cpu_percent" in metrics:
            alert_manager.check_threshold(
                metric_name=f"{process_name}_cpu_usage",
                current_value=metrics["cpu_percent"],
                component="python",
                thresholds={
                    "info": 50,  # 50%
                    "warning": 75,  # 75%
                    "critical": 90  # 90%
                },
                message_template=f"Python process '{process_name}' CPU usage is high: {{current_value:.1f}}% (threshold: {{threshold:.1f}}%)"
            )
            
            # Also check for auto-resolution
            alert_manager.auto_resolve_alerts(
                component="python",
                metric_name=f"{process_name}_cpu_usage",
                current_value=metrics["cpu_percent"]
            )
        
        # Thread count (if available)
        if "thread_count" in metrics:
            alert_manager.check_threshold(
                metric_name=f"{process_name}_thread_count",
                current_value=metrics["thread_count"],
                component="python",
                thresholds={
                    "info": 50,  # 50 threads
                    "warning": 100,  # 100 threads
                    "critical": 200  # 200 threads
                },
                message_template=f"Python process '{process_name}' has a high thread count: {{current_value}} (threshold: {{threshold}})"
            )
            
            # Also check for auto-resolution
            alert_manager.auto_resolve_alerts(
                component="python",
                metric_name=f"{process_name}_thread_count",
                current_value=metrics["thread_count"]
            )


def render_alerts_page() -> None:
    """
    Render a full page for viewing and managing alerts.
    """
    st.title("🚨 Alert Management")
    
    # Add tabs for active alerts and alert history
    tab1, tab2, tab3 = st.tabs(["Active Alerts", "Alert History", "Settings"])
    
    with tab1:
        st.header("Active Alerts")
        
        # Get active alerts from session state
        if "alerts" in st.session_state and "active" in st.session_state.alerts:
            active_alerts = st.session_state.alerts["active"]
            
            if not active_alerts:
                st.success("✅ No active alerts")
            else:
                # Add filters
                col1, col2, col3 = st.columns(3)
                with col1:
                    components = ["All"] + list(set(a["component"] for a in active_alerts))
                    component_filter = st.selectbox("Component", components, key="active_component_filter")
                
                with col2:
                    severities = ["All", "critical", "warning", "info"]
                    severity_filter = st.selectbox("Severity", severities, key="active_severity_filter")
                
                with col3:
                    sort_options = ["Newest first", "Oldest first", "Severity (high to low)", "Severity (low to high)"]
                    sort_option = st.selectbox("Sort by", sort_options, key="active_sort_option")
                
                # Apply filters
                filtered_alerts = active_alerts
                if component_filter != "All":
                    filtered_alerts = [a for a in filtered_alerts if a["component"] == component_filter]
                
                if severity_filter != "All":
                    filtered_alerts = [a for a in filtered_alerts if a["severity"] == severity_filter]
                
                # Apply sorting
                if sort_option == "Newest first":
                    filtered_alerts.sort(key=lambda a: a["timestamp"], reverse=True)
                elif sort_option == "Oldest first":
                    filtered_alerts.sort(key=lambda a: a["timestamp"])
                elif sort_option == "Severity (high to low)":
                    severity_order = {"critical": 0, "warning": 1, "info": 2}
                    filtered_alerts.sort(key=lambda a: severity_order.get(a["severity"], 3))
                elif sort_option == "Severity (low to high)":
                    severity_order = {"info": 0, "warning": 1, "critical": 2}
                    filtered_alerts.sort(key=lambda a: severity_order.get(a["severity"], 3))
                
                # Display alerts
                st.write(f"Showing {len(filtered_alerts)} alerts")
                
                for i, alert in enumerate(filtered_alerts):
                    severity_colors = {
                        "critical": "#F8333C",
                        "warning": "#FCAB10",
                        "info": "#2B9EB3"
                    }
                    severity_color = severity_colors.get(alert["severity"], "#AAAAAA")
                    
                    status_colors = {
                        "active": "#F8333C",
                        "acknowledged": "#FCAB10",
                        "resolved": "#44AF69"
                    }
                    status_color = status_colors.get(alert["status"], "#AAAAAA")
                    
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="padding: 15px; border-left: 5px solid {severity_color}; margin-bottom: 15px; background-color: rgba(0,0,0,0.03); border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <div>
                                        <h3 style="margin: 0; font-size: 18px;">{alert["metric_name"]}</h3>
                                        <p style="margin: 5px 0;"><strong>Component:</strong> {alert["component"]}</p>
                                    </div>
                                    <div>
                                        <span style="background-color: {severity_color}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 12px; text-transform: uppercase;">{alert["severity"]}</span>
                                        <span style="background-color: {status_color}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 12px; text-transform: uppercase; margin-left: 5px;">{alert["status"]}</span>
                                    </div>
                                </div>
                                <p style="margin: 10px 0;">{alert["message"]}</p>
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="color: #666; font-size: 12px;">Triggered: {alert["timestamp"]}</span>
                                    <div>
                                        <button id="ack_btn_{i}" style="background: none; border: 1px solid #666; padding: 3px 8px; border-radius: 3px; margin-right: 5px; cursor: pointer;">Acknowledge</button>
                                        <button id="res_btn_{i}" style="background: none; border: 1px solid #F8333C; color: #F8333C; padding: 3px 8px; border-radius: 3px; cursor: pointer;">Resolve</button>
                                    </div>
                                </div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if alert["status"] == "active":
                                acknowledge_reason = st.text_input(
                                    "Acknowledgement note (optional)", 
                                    key=f"ack_note_{alert['alert_id']}"
                                )
                            
                            if alert["status"] != "resolved":
                                resolution_note = st.text_input(
                                    "Resolution note (optional)", 
                                    key=f"res_note_{alert['alert_id']}"
                                )
                        
                        with col2:
                            if alert["status"] == "active":
                                if st.button("Acknowledge", key=f"ack_btn_{alert['alert_id']}"):
                                    alert_manager = AlertManager({})  # Empty config, just for using the method
                                    alert_manager.acknowledge_alert(alert["alert_id"], user="admin")
                                    st.experimental_rerun()
                            
                            if alert["status"] != "resolved":
                                if st.button("Resolve", key=f"res_btn_{alert['alert_id']}"):
                                    alert_manager = AlertManager({})  # Empty config, just for using the method
                                    alert_manager.resolve_alert(
                                        alert["alert_id"], 
                                        resolution_message=st.session_state.get(f"res_note_{alert['alert_id']}", "")
                                    )
                                    st.experimental_rerun()
        else:
            st.info("Alert system initializing...")
    
    with tab2:
        st.header("Alert History")
        
        # Get alert history from session state
        if "alerts" in st.session_state and "history" in st.session_state.alerts:
            alert_history = st.session_state.alerts["history"]
            
            if not alert_history:
                st.info("No alert history available")
            else:
                # Add filters
                col1, col2, col3 = st.columns(3)
                with col1:
                    components = ["All"] + list(set(a["component"] for a in alert_history))
                    component_filter = st.selectbox("Component", components, key="history_component_filter")
                
                with col2:
                    severities = ["All", "critical", "warning", "info"]
                    severity_filter = st.selectbox("Severity", severities, key="history_severity_filter")
                
                with col3:
                    time_range = st.selectbox(
                        "Time Range",
                        ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"],
                        key="history_time_range"
                    )
                
                # Apply filters
                filtered_history = alert_history
                if component_filter != "All":
                    filtered_history = [a for a in filtered_history if a["component"] == component_filter]
                
                if severity_filter != "All":
                    filtered_history = [a for a in filtered_history if a["severity"] == severity_filter]
                
                # Apply time range filter
                now = datetime.now()
                if time_range == "Last 24 hours":
                    start_time = now - timedelta(days=1)
                    filtered_history = [a for a in filtered_history if datetime.fromisoformat(a["timestamp"]) >= start_time]
                elif time_range == "Last 7 days":
                    start_time = now - timedelta(days=7)
                    filtered_history = [a for a in filtered_history if datetime.fromisoformat(a["timestamp"]) >= start_time]
                elif time_range == "Last 30 days":
                    start_time = now - timedelta(days=30)
                    filtered_history = [a for a in filtered_history if datetime.fromisoformat(a["timestamp"]) >= start_time]
                
                # Sort by timestamp (newest first)
                filtered_history.sort(key=lambda a: a["timestamp"], reverse=True)
                
                # Display history
                st.write(f"Showing {len(filtered_history)} alerts")
                
                # Create DataFrame for more compact display
                if filtered_history:
                    history_data = []
                    for alert in filtered_history:
                        history_data.append({
                            "Alert ID": alert["alert_id"],
                            "Metric": alert["metric_name"],
                            "Component": alert["component"],
                            "Severity": alert["severity"],
                            "Value": f"{alert['current_value']:.2f}" if isinstance(alert['current_value'], (int, float)) else alert['current_value'],
                            "Threshold": f"{alert['threshold']:.2f}" if isinstance(alert['threshold'], (int, float)) else alert['threshold'],
                            "Triggered": alert["timestamp"],
                            "Resolved": alert["resolved_at"] or "N/A",
                            "Resolution": alert["resolution_message"] or "N/A"
                        })
                    
                    history_df = pd.DataFrame(history_data)
                    st.dataframe(history_df, use_container_width=True)
                    
                    # Add export option
                    if st.button("Export History to CSV"):
                        csv = history_df.to_csv(index=False)
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"alert_history_{now_str}.csv",
                            mime="text/csv"
                        )
        else:
            st.info("Alert system initializing...")
    
    with tab3:
        st.header("Alert Settings")
        
        # Get current config
        config = {}
        if "config" in st.session_state:
            config = st.session_state.config
        
        with st.form("alert_settings_form"):
            st.subheader("General Alert Settings")
            
            enable_alerts = st.checkbox("Enable Alert System", value=config.get("alerts_enabled", True))
            
            st.subheader("System Monitoring Alerts")
            enable_system_alerts = st.checkbox("Enable System Alerts", value=config.get("system", {}).get("alerts_enabled", True))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                cpu_warning = st.slider(
                    "CPU Warning Threshold (%)",
                    min_value=50,
                    max_value=95,
                    value=config.get("system", {}).get("thresholds", {}).get("cpu_usage", {}).get("warning", 85),
                    step=5
                )
                cpu_critical = st.slider(
                    "CPU Critical Threshold (%)",
                    min_value=60,
                    max_value=100,
                    value=config.get("system", {}).get("thresholds", {}).get("cpu_usage", {}).get("critical", 95),
                    step=5
                )
            
            with col2:
                mem_warning = st.slider(
                    "Memory Warning Threshold (%)",
                    min_value=50,
                    max_value=95,
                    value=config.get("system", {}).get("thresholds", {}).get("memory_usage", {}).get("warning", 85),
                    step=5
                )
                mem_critical = st.slider(
                    "Memory Critical Threshold (%)",
                    min_value=60,
                    max_value=100,
                    value=config.get("system", {}).get("thresholds", {}).get("memory_usage", {}).get("critical", 95),
                    step=5
                )
            
            with col3:
                disk_warning = st.slider(
                    "Disk Warning Threshold (%)",
                    min_value=70,
                    max_value=95,
                    value=config.get("system", {}).get("thresholds", {}).get("disk_usage", {}).get("warning", 85),
                    step=5
                )
                disk_critical = st.slider(
                    "Disk Critical Threshold (%)",
                    min_value=80,
                    max_value=99,
                    value=config.get("system", {}).get("thresholds", {}).get("disk_usage", {}).get("critical", 95),
                    step=5
                )
            
            st.subheader("Ollama Monitoring Alerts")
            enable_ollama_alerts = st.checkbox("Enable Ollama Alerts", value=config.get("ollama", {}).get("alerts_enabled", True))
            
            col1, col2 = st.columns(2)
            with col1:
                latency_warning = st.slider(
                    "Latency Warning Threshold (sec)",
                    min_value=0.5,
                    max_value=10.0,
                    value=config.get("ollama", {}).get("thresholds", {}).get("request_latency", {}).get("warning", 3.0),
                    step=0.5
                )
                latency_critical = st.slider(
                    "Latency Critical Threshold (sec)",
                    min_value=1.0,
                    max_value=20.0,
                    value=config.get("ollama", {}).get("thresholds", {}).get("request_latency", {}).get("critical", 10.0),
                    step=1.0
                )
            
            with col2:
                error_warning = st.slider(
                    "Error Rate Warning Threshold (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=config.get("ollama", {}).get("thresholds", {}).get("error_rate", {}).get("warning", 5.0),
                    step=1.0
                )
                error_critical = st.slider(
                    "Error Rate Critical Threshold (%)",
                    min_value=5.0,
                    max_value=90.0,
                    value=config.get("ollama", {}).get("thresholds", {}).get("error_rate", {}).get("critical", 10.0),
                    step=5.0
                )
            
            st.subheader("Python Monitoring Alerts")
            enable_python_alerts = st.checkbox("Enable Python App Alerts", value=config.get("python", {}).get("alerts_enabled", True))
            
            col1, col2 = st.columns(2)
            with col1:
                py_mem_warning = st.slider(
                    "Python Mem Warning Threshold (%)",
                    min_value=10,
                    max_value=50,
                    value=config.get("python", {}).get("thresholds", {}).get("memory_usage", {}).get("warning", 25),
                    step=5
                )
                py_mem_critical = st.slider(
                    "Python Mem Critical Threshold (%)",
                    min_value=20,
                    max_value=70,
                    value=config.get("python", {}).get("thresholds", {}).get("memory_usage", {}).get("critical", 40),
                    step=5
                )
            
            with col2:
                py_cpu_warning = st.slider(
                    "Python CPU Warning Threshold (%)",
                    min_value=30,
                    max_value=90,
                    value=config.get("python", {}).get("thresholds", {}).get("cpu_usage", {}).get("warning", 75),
                    step=5
                )
                py_cpu_critical = st.slider(
                    "Python CPU Critical Threshold (%)",
                    min_value=50,
                    max_value=99,
                    value=config.get("python", {}).get("thresholds", {}).get("cpu_usage", {}).get("critical", 90),
                    step=5
                )
            
            st.subheader("Notification Settings")
            enable_email = st.checkbox("Enable Email Notifications", value=config.get("notifications", {}).get("email", {}).get("enabled", False))
            
            if enable_email:
                col1, col2 = st.columns(2)
                with col1:
                    recipient_email = st.text_input(
                        "Recipient Email",
                        value=config.get("notifications", {}).get("email", {}).get("recipient", "")
                    )
                    smtp_server = st.text_input(
                        "SMTP Server",
                        value=config.get("notifications", {}).get("email", {}).get("smtp_server", "smtp.gmail.com")
                    )
                
                with col2:
                    smtp_port = st.number_input(
                        "SMTP Port",
                        min_value=1,
                        max_value=65535,
                        value=config.get("notifications", {}).get("email", {}).get("smtp_port", 587)
                    )
                    sender_email = st.text_input(
                        "Sender Email",
                        value=config.get("notifications", {}).get("email", {}).get("sender_email", "")
                    )
            
            enable_webhook = st.checkbox("Enable Webhook Notifications", value=config.get("notifications", {}).get("webhook", {}).get("enabled", False))
            
            if enable_webhook:
                webhook_url = st.text_input(
                    "Webhook URL",
                    value=config.get("notifications", {}).get("webhook", {}).get("url", "")
                )
            
            # Save settings button
            submit_button = st.form_submit_button("Save Alert Settings")
            
            if submit_button:
                # Build new config
                new_config = {
                    "alerts_enabled": enable_alerts,
                    "system": {
                        "alerts_enabled": enable_system_alerts,
                        "thresholds": {
                            "cpu_usage": {
                                "warning": cpu_warning,
                                "critical": cpu_critical
                            },
                            "memory_usage": {
                                "warning": mem_warning,
                                "critical": mem_critical
                            },
                            "disk_usage": {
                                "warning": disk_warning,
                                "critical": disk_critical
                            }
                        }
                    },
                    "ollama": {
                        "alerts_enabled": enable_ollama_alerts,
                        "thresholds": {
                            "request_latency": {
                                "warning": latency_warning,
                                "critical": latency_critical
                            },
                            "error_rate": {
                                "warning": error_warning,
                                "critical": error_critical
                            }
                        }
                    },
                    "python": {
                        "alerts_enabled": enable_python_alerts,
                        "thresholds": {
                            "memory_usage": {
                                "warning": py_mem_warning,
                                "critical": py_mem_critical
                            },
                            "cpu_usage": {
                                "warning": py_cpu_warning,
                                "critical": py_cpu_critical
                            }
                        }
                    },
                    "notifications": {
                        "email": {
                            "enabled": enable_email,
                            "recipient": recipient_email if enable_email else "",
                            "smtp_server": smtp_server if enable_email else "",
                            "smtp_port": smtp_port if enable_email else 587,
                            "sender_email": sender_email if enable_email else ""
                        },
                        "webhook": {
                            "enabled": enable_webhook,
                            "url": webhook_url if enable_webhook else ""
                        }
                    }
                }
                
                # Update session state
                st.session_state.config = new_config
                
                # Show success message
                st.success("Alert settings saved successfully!")


def clear_alerts():
    """
    Clear all alerts from session state (for testing/development).
    """
    if "alerts" in st.session_state:
        st.session_state.alerts = {
            "active": [],
            "history": []
        }
        logger.info("All alerts cleared from session state")


def generate_test_alerts():
    """
    Generate test alerts for development and testing.
    """
    if "alerts" not in st.session_state:
        st.session_state.alerts = {
            "active": [],
            "history": []
        }
    
    # Generate test active alerts
    test_alerts = [
        Alert(
            alert_id="test_cpu_1",
            metric_name="cpu_usage",
            severity=AlertSeverity.WARNING,
            threshold=85.0,
            current_value=92.5,
            message="CPU usage is high: 92.5% (threshold: 85.0%)",
            component="system"
        ),
        Alert(
            alert_id="test_memory_1",
            metric_name="memory_usage",
            severity=AlertSeverity.CRITICAL,
            threshold=95.0,
            current_value=97.1,
            message="Memory usage is high: 97.1% (threshold: 95.0%)",
            component="system"
        ),
        Alert(
            alert_id="test_latency_1",
            metric_name="request_latency",
            severity=AlertSeverity.INFO,
            threshold=1.0,
            current_value=1.2,
            message="Ollama request latency is high: 1.2s (threshold: 1.0s)",
            component="ollama"
        )
    ]
    
    # Add to session state
    for alert in test_alerts:
        st.session_state.alerts["active"].append(alert.to_dict())
    
    # Generate test historical alerts
    test_history = [
        Alert(
            alert_id="test_hist_1",
            metric_name="disk_usage",
            severity=AlertSeverity.CRITICAL,
            threshold=95.0,
            current_value=98.2,
            message="Disk usage is high: 98.2% (threshold: 95.0%)",
            component="system",
            timestamp=datetime.now() - timedelta(hours=12)
        ),
        Alert(
            alert_id="test_hist_2",
            metric_name="error_rate",
            severity=AlertSeverity.WARNING,
            threshold=5.0,
            current_value=8.3,
            message="Ollama error rate is high: 8.3% (threshold: 5.0%)",
            component="ollama",
            timestamp=datetime.now() - timedelta(days=2)
        )
    ]
    
    # Mark as resolved and add to history
    for alert in test_history:
        alert.resolve("Test resolution message")
        st.session_state.alerts["history"].append(alert.to_dict())
    
    logger.info("Generated test alerts")