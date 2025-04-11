"""
Custom monitoring module for user-defined metrics and integrations.
"""

import logging
import json
import os
import time
import requests
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from collections import deque
import pandas as pd
import asyncio
import importlib.util
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("custom_monitor")

# Constants
MAX_HISTORY_SIZE = 1000  # Maximum number of data points to keep in memory
DEFAULT_INTERVAL = 60  # Default collection interval in seconds
DEFAULT_CONFIG_PATH = "config/custom_monitors.json"


class CustomMonitor:
    """
    Monitor for custom user-defined metrics and integrations.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 max_history: int = MAX_HISTORY_SIZE):
        """
        Initialize the custom monitor.
        
        Args:
            config_path: Path to custom monitor configuration file
            max_history: Maximum number of historical data points to keep
        """
        self.max_history = max_history
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        
        # Initialize data structures to store monitoring data
        self.metrics_history = {}  # Map of metric_name -> deque of historical values
        self.last_collection_time = {}  # Map of metric_name -> last collection timestamp
        self.metric_configs = {}  # Map of metric_name -> configuration
        
        # Collection functions for different types of metrics
        self.collectors = {
            "http": self._collect_http_metric,
            "command": self._collect_command_metric,
            "script": self._collect_script_metric,
            "file": self._collect_file_metric,
            "function": self._collect_function_metric,
        }
        
        # Load configuration from file
        self._load_config()
        
        logger.info(f"Initialized custom monitor")
        logger.info(f"Configured metrics: {list(self.metric_configs.keys())}")
    
    def _load_config(self):
        """Load custom monitor configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Process configuration
                for metric_config in config.get('metrics', []):
                    metric_name = metric_config.get('name')
                    if not metric_name:
                        logger.warning(f"Skipping metric config without name: {metric_config}")
                        continue
                    
                    self.metric_configs[metric_name] = metric_config
                    self.metrics_history[metric_name] = deque(maxlen=self.max_history)
                    
                    logger.info(f"Loaded configuration for metric: {metric_name}")
            else:
                # Create default configuration file if it doesn't exist
                self._create_default_config()
        
        except Exception as e:
            logger.error(f"Error loading custom monitor configuration: {str(e)}")
    
    def _create_default_config(self):
        """Create default configuration file if it doesn't exist."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Create default configuration
            default_config = {
                "metrics": [
                    {
                        "name": "example_http_metric",
                        "type": "http",
                        "url": "http://localhost:8000/metrics",
                        "method": "GET",
                        "headers": {},
                        "interval": 60,
                        "timeout": 10,
                        "enabled": False
                    },
                    {
                        "name": "example_command_metric",
                        "type": "command",
                        "command": "echo '{\"value\": 123}'",
                        "parser": "json",
                        "interval": 60,
                        "enabled": False
                    }
                ]
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            logger.info(f"Created default configuration at {self.config_path}")
        
        except Exception as e:
            logger.error(f"Error creating default configuration: {str(e)}")
    
    def add_metric(self, metric_config: Dict[str, Any]) -> bool:
        """
        Add a new custom metric configuration.
        
        Args:
            metric_config: Configuration for the metric
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metric_name = metric_config.get('name')
            metric_type = metric_config.get('type')
            
            if not metric_name:
                logger.error("Metric configuration must include a name")
                return False
            
            if not metric_type or metric_type not in self.collectors:
                logger.error(f"Invalid metric type: {metric_type}. Must be one of {list(self.collectors.keys())}")
                return False
            
            # Add the metric configuration
            self.metric_configs[metric_name] = metric_config
            
            # Initialize history for this metric
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = deque(maxlen=self.max_history)
            
            # Save the updated configuration
            self._save_config()
            
            logger.info(f"Added custom metric: {metric_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding custom metric: {str(e)}")
            return False
    
    def remove_metric(self, metric_name: str) -> bool:
        """
        Remove a custom metric configuration.
        
        Args:
            metric_name: Name of the metric to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if metric_name not in self.metric_configs:
                logger.warning(f"Metric not found: {metric_name}")
                return False
            
            # Remove the metric configuration
            del self.metric_configs[metric_name]
            
            # Remove history for this metric
            if metric_name in self.metrics_history:
                del self.metrics_history[metric_name]
            
            # Remove last collection time
            if metric_name in self.last_collection_time:
                del self.last_collection_time[metric_name]
            
            # Save the updated configuration
            self._save_config()
            
            logger.info(f"Removed custom metric: {metric_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error removing custom metric: {str(e)}")
            return False
    
    def update_metric(self, metric_name: str, metric_config: Dict[str, Any]) -> bool:
        """
        Update a custom metric configuration.
        
        Args:
            metric_name: Name of the metric to update
            metric_config: New configuration for the metric
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if metric_name not in self.metric_configs:
                logger.warning(f"Metric not found: {metric_name}")
                return False
            
            # Ensure the name in the config matches
            if 'name' in metric_config and metric_config['name'] != metric_name:
                logger.error(f"Metric name in config ({metric_config['name']}) does not match target metric ({metric_name})")
                return False
            
            # Update the metric configuration
            self.metric_configs[metric_name].update(metric_config)
            
            # Save the updated configuration
            self._save_config()
            
            logger.info(f"Updated custom metric: {metric_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error updating custom metric: {str(e)}")
            return False
    
    def _save_config(self):
        """Save the current configuration to the config file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Create config object
            config = {
                "metrics": list(self.metric_configs.values())
            }
            
            # Write to file
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved custom monitor configuration to {self.config_path}")
        
        except Exception as e:
            logger.error(f"Error saving custom monitor configuration: {str(e)}")
    
    async def _collect_http_metric(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect a metric via HTTP request.
        
        Args:
            config: Metric configuration
            
        Returns:
            Dictionary with collected metric data
        """
        url = config.get('url')
        method = config.get('method', 'GET')
        headers = config.get('headers', {})
        timeout = config.get('timeout', 10)
        
        start_time = time.time()
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            # Parse the response based on content type
            content_type = response.headers.get('Content-Type', '')
            
            if 'application/json' in content_type:
                data = response.json()
            else:
                data = response.text
            
            return {
                "timestamp": datetime.now().isoformat(),
                "status_code": response.status_code,
                "duration_seconds": duration,
                "data": data,
                "success": response.status_code < 400
            }
        
        except Exception as e:
            logger.error(f"Error collecting HTTP metric {config.get('name')}: {str(e)}")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "success": False
            }    
    
    async def collect_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Collect all enabled metrics.
        
        Returns:
            Dictionary mapping metric names to collected data
        """
        results = {}
        
        for metric_name, config in self.metric_configs.items():
            # Skip disabled metrics
            if not config.get('enabled', True):
                continue
            
            # Check if it's time to collect this metric
            if metric_name in self.last_collection_time:
                last_time = self.last_collection_time[metric_name]
                interval = config.get('interval', DEFAULT_INTERVAL)
                
                # Skip if not enough time has passed
                time_diff = (datetime.now() - last_time).total_seconds()
                if time_diff < interval:
                    logger.debug(f"Skipping metric {metric_name}: not ready yet (interval: {interval}s, elapsed: {time_diff}s)")
                    continue
            
            # Collect the metric
            result = await self.collect_metric(metric_name)
            results[metric_name] = result
        
        return results
    
    def get_metric_history(self, metric_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical data for a specific metric.
        
        Args:
            metric_name: Name of the metric
            limit: Maximum number of data points to return
            
        Returns:
            List of historical metric data points
        """
        if metric_name not in self.metrics_history:
            return []
        
        history = list(self.metrics_history[metric_name])
        
        if limit is not None and limit > 0:
            history = history[-limit:]
        
        return history
    
    def get_all_metrics_status(self):
        """
        Get the status of all custom metrics.
        
        Returns:
            List of dictionaries with metric status information
        """
        result = []
        
        for metric_name, config in self.metrics_config.items():
            try:
                # Get the latest data point if available
                latest_data = None
                if metric_name in self.metrics_data and self.metrics_data[metric_name]:
                    latest_data = self.metrics_data[metric_name][-1]
                
                status = {
                    "name": metric_name,
                    "type": config.get("type", "unknown"),
                    "enabled": config.get("enabled", False),
                    "last_collected": latest_data["timestamp"] if latest_data else None,
                    "last_value": latest_data["value"] if latest_data else None,
                    "status": "active" if latest_data and time.time() - latest_data["timestamp"] < 300 else "inactive",
                    "config": config
                }
                
                result.append(status)
            except Exception as e:
                # Include failed metrics with error info
                result.append({
                    "name": metric_name,
                    "type": config.get("type", "unknown"),
                    "enabled": config.get("enabled", False),
                    "status": "error",
                    "error": str(e)
                })
        
        return result
    
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert monitoring data to pandas DataFrames for analysis.
        
        Returns:
            Dictionary of DataFrames with monitoring data
        """
        dfs = {}
        
        for metric_name, history in self.metrics_history.items():
            if not history:
                dfs[metric_name] = pd.DataFrame()
                continue
            
            # Extract data points
            data_points = []
            
            for point in history:
                # Extract timestamp and success
                base_point = {
                    "timestamp": point.get("timestamp"),
                    "success": point.get("success", False),
                    "duration_seconds": point.get("duration_seconds", None),
                }
                
                # Extract data values
                if "data" in point and isinstance(point["data"], dict):
                    for key, value in point["data"].items():
                        if isinstance(value, (int, float, str, bool)) or value is None:
                            base_point[f"data_{key}"] = value
                
                # Add error if present
                if "error" in point:
                    base_point["error"] = point["error"]
                
                data_points.append(base_point)
            
            # Create DataFrame
            df = pd.DataFrame(data_points)
            
            # Convert timestamp to datetime if possible
            if "timestamp" in df.columns:
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                except Exception:
                    pass
            
            dfs[metric_name] = df
        
        return dfs
    
    async def background_monitoring_task(self, interval: int = DEFAULT_INTERVAL):
        """
        Background task to continuously collect metrics.
        
        Args:
            interval: Interval in seconds between checking for metrics to collect
        """
        while True:
            try:
                await self.collect_all_metrics()
                logger.info(f"Collected custom metrics successfully")
            except Exception as e:
                logger.error(f"Error in background monitoring: {str(e)}")
            
            await asyncio.sleep(interval)
"""
Custom monitoring module for user-defined metrics and integrations.
"""

import logging
import json
import os
import time
import requests
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from collections import deque
import pandas as pd
import asyncio
import importlib.util
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("custom_monitor")

# Constants
MAX_HISTORY_SIZE = 1000  # Maximum number of data points to keep in memory
DEFAULT_INTERVAL = 60  # Default collection interval in seconds
DEFAULT_CONFIG_PATH = "config/custom_monitors.json"


class CustomMonitor:
    """
    Monitor for custom user-defined metrics and integrations.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 max_history: int = MAX_HISTORY_SIZE):
        """
        Initialize the custom monitor.
        
        Args:
            config_path: Path to custom monitor configuration file
            max_history: Maximum number of historical data points to keep
        """
        self.max_history = max_history
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        
        # Initialize data structures to store monitoring data
        self.metrics_history = {}  # Map of metric_name -> deque of historical values
        self.last_collection_time = {}  # Map of metric_name -> last collection timestamp
        self.metric_configs = {}  # Map of metric_name -> configuration
        
        # Collection functions for different types of metrics
        self.collectors = {
            "http": self._collect_http_metric,
            "command": self._collect_command_metric,
            "script": self._collect_script_metric,
            "file": self._collect_file_metric,
            "function": self._collect_function_metric,
        }
        
        # Load configuration from file
        self._load_config()
        
        logger.info(f"Initialized custom monitor")
        logger.info(f"Configured metrics: {list(self.metric_configs.keys())}")
    
    def _load_config(self):
        """Load custom monitor configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Process configuration
                for metric_config in config.get('metrics', []):
                    metric_name = metric_config.get('name')
                    if not metric_name:
                        logger.warning(f"Skipping metric config without name: {metric_config}")
                        continue
                    
                    self.metric_configs[metric_name] = metric_config
                    self.metrics_history[metric_name] = deque(maxlen=self.max_history)
                    
                    logger.info(f"Loaded configuration for metric: {metric_name}")
            else:
                # Create default configuration file if it doesn't exist
                self._create_default_config()
        
        except Exception as e:
            logger.error(f"Error collecting command metric {config.get('name')}: {str(e)}")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "success": False
            }
    
    def _create_default_config(self):
        """Create default configuration file if it doesn't exist."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Create default configuration
            default_config = {
                "metrics": [
                    {
                        "name": "example_http_metric",
                        "type": "http",
                        "url": "http://localhost:8000/metrics",
                        "method": "GET",
                        "headers": {},
                        "interval": 60,
                        "timeout": 10,
                        "enabled": False
                    },
                    {
                        "name": "example_command_metric",
                        "type": "command",
                        "command": "echo '{\"value\": 123}'",
                        "parser": "json",
                        "interval": 60,
                        "enabled": False
                    }
                ]
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            logger.info(f"Created default configuration at {self.config_path}")
        
        except Exception as e:
            logger.error(f"Error creating default configuration: {str(e)}")
    
    def add_metric(self, metric_config: Dict[str, Any]) -> bool:
        """
        Add a new custom metric configuration.
        
        Args:
            metric_config: Configuration for the metric
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metric_name = metric_config.get('name')
            metric_type = metric_config.get('type')
            
            if not metric_name:
                logger.error("Metric configuration must include a name")
                return False
            
            if not metric_type or metric_type not in self.collectors:
                logger.error(f"Invalid metric type: {metric_type}. Must be one of {list(self.collectors.keys())}")
                return False
            
            # Add the metric configuration
            self.metric_configs[metric_name] = metric_config
            
            # Initialize history for this metric
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = deque(maxlen=self.max_history)
            
            # Save the updated configuration
            self._save_config()
            
            logger.info(f"Added custom metric: {metric_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding custom metric: {str(e)}")
            return False
    
    def remove_metric(self, metric_name: str) -> bool:
        """
        Remove a custom metric configuration.
        
        Args:
            metric_name: Name of the metric to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if metric_name not in self.metric_configs:
                logger.warning(f"Metric not found: {metric_name}")
                return False
            
            # Remove the metric configuration
            del self.metric_configs[metric_name]
            
            # Remove history for this metric
            if metric_name in self.metrics_history:
                del self.metrics_history[metric_name]
            
            # Remove last collection time
            if metric_name in self.last_collection_time:
                del self.last_collection_time[metric_name]
            
            # Save the updated configuration
            self._save_config()
            
            logger.info(f"Removed custom metric: {metric_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error removing custom metric: {str(e)}")
            return False
    
    def update_metric(self, metric_name: str, metric_config: Dict[str, Any]) -> bool:
        """
        Update a custom metric configuration.
        
        Args:
            metric_name: Name of the metric to update
            metric_config: New configuration for the metric
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if metric_name not in self.metric_configs:
                logger.warning(f"Metric not found: {metric_name}")
                return False
            
            # Ensure the name in the config matches
            if 'name' in metric_config and metric_config['name'] != metric_name:
                logger.error(f"Metric name in config ({metric_config['name']}) does not match target metric ({metric_name})")
                return False
            
            # Update the metric configuration
            self.metric_configs[metric_name].update(metric_config)
            
            # Save the updated configuration
            self._save_config()
            
            logger.info(f"Updated custom metric: {metric_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error updating custom metric: {str(e)}")
            return False
    
    def _save_config(self):
        """Save the current configuration to the config file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Create config object
            config = {
                "metrics": list(self.metric_configs.values())
            }
            
            # Write to file
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved custom monitor configuration to {self.config_path}")
        
        except Exception as e:
            logger.error(f"Error saving custom monitor configuration: {str(e)}")
    
    async def _collect_http_metric(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect a metric via HTTP request.
        
        Args:
            config: Metric configuration
            
        Returns:
            Dictionary with collected metric data
        """
        url = config.get('url')
        method = config.get('method', 'GET')
        headers = config.get('headers', {})
        timeout = config.get('timeout', 10)
        
        start_time = time.time()
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            # Parse the response based on content type
            content_type = response.headers.get('Content-Type', '')
            
            if 'application/json' in content_type:
                data = response.json()
            else:
                data = response.text
            
            return {
                "timestamp": datetime.now().isoformat(),
                "status_code": response.status_code,
                "duration_seconds": duration,
                "data": data,
                "success": response.status_code < 400
            }
        
        except Exception as e:
            logger.error(f"Error collecting HTTP metric {config.get('name')}: {str(e)}")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "success": False
            }
    
    async def _collect_command_metric(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect a metric by running a command.
        
        Args:
            config: Metric configuration
            
        Returns:
            Dictionary with collected metric data
        """
        command = config.get('command')
        parser = config.get('parser', 'text')
        timeout = config.get('timeout', 30)
        
        start_time = time.time()
        
        try:
            if sys.platform == 'win32':
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=True
                )
            else:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=True,
                    executable='/bin/bash'
                )
            
            duration = time.time() - start_time
            
            # Parse the output based on the parser type
            if parser == 'json':
                try:
                    data = json.loads(process.stdout)
                except json.JSONDecodeError:
                    data = {"raw": process.stdout}
            else:
                data = {"raw": process.stdout}
            
            return {
                "timestamp": datetime.now().isoformat(),
                "exit_code": process.returncode,
                "duration_seconds": duration,
                "data": data,
                "stderr": process.stderr,
                "success": process.returncode == 0
            }
        
        except subprocess.TimeoutExpired:
            logger.error(f"Command timeout for metric {config.get('name')}")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "error": "Command timed out",
                "duration_seconds": time.time() - start_time,
                "success": False
            }
        
        except Exception as e:
            logger.error(f"Error collecting HTTP metric {config.get('name')}: {str(e)}")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "success": False
            }
        
    
    async def _collect_script_metric(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect a metric by executing a script file.

        Args:
            config: Metric configuration

        Returns:
            Dictionary with collected metric data
        """
        script_path = config.get('script_path')
        parser = config.get('parser', 'text')
        timeout = config.get('timeout', 30)

        start_time = time.time()

        try:
            process = subprocess.run(
                [script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                executable='/bin/bash' if sys.platform != 'win32' else None
            )

            duration = time.time() - start_time

            if parser == 'json':
                try:
                    data = json.loads(process.stdout)
                except json.JSONDecodeError:
                    data = {"raw": process.stdout}
            else:
                data = {"raw": process.stdout}

            return {
                "timestamp": datetime.now().isoformat(),
                "exit_code": process.returncode,
                "duration_seconds": duration,
                "data": data,
                "stderr": process.stderr,
                "success": process.returncode == 0
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Script timeout for metric {config.get('name')}")

            return {
                "timestamp": datetime.now().isoformat(),
                "error": "Script timed out",
                "duration_seconds": time.time() - start_time,
                "success": False
            }

        except Exception as e:
            logger.error(f"Error collecting script metric {config.get('name')}: {str(e)}")

            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "success": False
            }

    async def _collect_file_metric(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect a metric by reading a file.

        Args:
            config: Metric configuration

        Returns:
            Dictionary with collected metric data
        """
        file_path = config.get('file_path')
        parser = config.get('parser', 'text')

        start_time = time.time()

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            if parser == 'json':
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    data = {"raw": content}
            else:
                data = {"raw": content}

            return {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": time.time() - start_time,
                "data": data,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error reading file metric {config.get('name')}: {str(e)}")

            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "success": False
            }

    async def _collect_function_metric(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect a metric by invoking a function.

        Args:
            config: Metric configuration

        Returns:
            Dictionary with collected metric data
        """
        func = config.get('function')
        args = config.get('args', [])
        kwargs = config.get('kwargs', {})

        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            return {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": time.time() - start_time,
                "data": {"result": result},
                "success": True
            }

        except Exception as e:
            logger.error(f"Error executing function metric {config.get('name')}: {str(e)}")

            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "success": False
            }

    def get_all_metrics_status(self):
        """
        Get the status of all custom metrics.
        
        Returns:
            List of dictionaries with metric status information
        """
        result = []
        
        # Check if metrics_config exists
        if not hasattr(self, 'metrics_config'):
            # Return empty result if no metrics config is available
            return result
        
        for metric_name, config in self.metrics_config.items():
            try:
                # Get the latest data point if available
                latest_data = None
                if hasattr(self, 'metrics_data') and metric_name in self.metrics_data and self.metrics_data[metric_name]:
                    latest_data = self.metrics_data[metric_name][-1]
                
                status = {
                    "name": metric_name,
                    "type": config.get("type", "unknown"),
                    "enabled": config.get("enabled", False),
                    "last_collected": latest_data["timestamp"] if latest_data else None,
                    "last_value": latest_data["value"] if latest_data else None,
                    "status": "active" if latest_data and time.time() - latest_data["timestamp"] < 300 else "inactive",
                    "config": config
                }
                
                result.append(status)
            except Exception as e:
                # Include failed metrics with error info
                result.append({
                    "name": metric_name,
                    "type": config.get("type", "unknown"),
                    "enabled": config.get("enabled", False),
                    "status": "error",
                    "error": str(e)
                })
        
        return result