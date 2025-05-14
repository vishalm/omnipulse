"""
System monitoring module for collecting system metrics like CPU, memory, disk, and network.
"""

import psutil
import platform
import time
import logging
import os
import socket
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from collections import deque
import asyncio
import GPUtil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_monitor")

# Constants
MAX_HISTORY_SIZE = 1000  # Maximum number of data points to keep in memory
DEFAULT_INTERVAL = 5  # Default collection interval in seconds
DEFAULT_DISK_PATHS = ["/", "/home", "/tmp"]  # Default disk paths to monitor
DEFAULT_NETWORK_INTERFACES = ["eth0", "wlan0"]  # Default network interfaces to monitor


class SystemMonitor:
    """
    Monitor for system resources (CPU, memory, disk, network).
    """
    
    def __init__(self, 
                 max_history: int = MAX_HISTORY_SIZE,
                 disk_paths: Optional[List[str]] = None,
                 network_interfaces: Optional[List[str]] = None):
        """
        Initialize the system monitor.
        
        Args:
            max_history: Maximum number of historical data points to keep
            disk_paths: List of disk paths to monitor
            network_interfaces: List of network interfaces to monitor
        """
        self.max_history = max_history
        
        # Set disk paths to monitor
        if disk_paths is None:
            # On Windows, use drive letters
            if platform.system() == "Windows":
                self.disk_paths = [f"{d}:\\" for d in "CDEF" if os.path.exists(f"{d}:\\")]
            else:
                # On Unix-like systems, use default paths
                self.disk_paths = [p for p in DEFAULT_DISK_PATHS if os.path.exists(p)]
        else:
            self.disk_paths = disk_paths
        
        # Set network interfaces to monitor
        if network_interfaces is None:
            # Automatically detect active interfaces
            self.network_interfaces = self._detect_network_interfaces()
        else:
            self.network_interfaces = network_interfaces
        
        # Initialize data structures to store monitoring data
        self.cpu_history = deque(maxlen=max_history)
        self.memory_history = deque(maxlen=max_history)
        self.disk_history = deque(maxlen=max_history)
        self.network_history = deque(maxlen=max_history)
        self.gpu_history = deque(maxlen=max_history) if self._has_gpu() else None
        self.system_info = self._collect_system_info()
        
        # For calculating rates (e.g., network throughput)
        self.last_network_io = None
        self.last_network_time = None
        
        logger.info(f"Initialized system monitor")
        logger.info(f"Monitoring disk paths: {self.disk_paths}")
        logger.info(f"Monitoring network interfaces: {self.network_interfaces}")
        
    def _detect_network_interfaces(self) -> List[str]:
        """
        Detect active network interfaces.
        
        Returns:
            List of active network interface names
        """
        interfaces = []
        
        try:
            stats = psutil.net_if_stats()
            io_counters = psutil.net_io_counters(pernic=True)
            
            for interface, stats_data in stats.items():
                # Skip loopback and interfaces that are down
                if interface == "lo" or not stats_data.isup:
                    continue
                
                # Check if the interface has any traffic
                if interface in io_counters:
                    if io_counters[interface].bytes_sent > 0 or io_counters[interface].bytes_recv > 0:
                        interfaces.append(interface)
            
            # Fallback to default interfaces if none detected
            if not interfaces:
                return [iface for iface in DEFAULT_NETWORK_INTERFACES 
                        if iface in stats and stats[iface].isup]
            
            return interfaces
        
        except Exception as e:
            logger.error(f"Error detecting network interfaces: {str(e)}")
            return DEFAULT_NETWORK_INTERFACES
    
    def _has_gpu(self) -> bool:
        """
        Check if the system has NVIDIA GPUs.
        
        Returns:
            True if NVIDIA GPUs are detected, False otherwise
        """
        try:
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except Exception:
            return False
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """
        Collect general system information.
        
        Returns:
            Dictionary with system information
        """
        try:
            # Get CPU information
            cpu_info = {
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_model": self._get_cpu_model(),
                "cpu_frequency": self._get_cpu_frequency(),
            }
            
            # Get memory information
            memory = psutil.virtual_memory()
            memory_info = {
                "total_memory_gb": round(memory.total / (1024**3), 2),
                "available_memory_gb": round(memory.available / (1024**3), 2),
            }
            
            # Get disk information
            disk_info = {}
            for disk_path in self.disk_paths:
                try:
                    disk_usage = psutil.disk_usage(disk_path)
                    disk_info[disk_path] = {
                        "total_gb": round(disk_usage.total / (1024**3), 2),
                        "used_gb": round(disk_usage.used / (1024**3), 2),
                        "free_gb": round(disk_usage.free / (1024**3), 2),
                        "percent_used": disk_usage.percent,
                    }
                except Exception as e:
                    logger.error(f"Error getting disk info for {disk_path}: {str(e)}")
            
            # Get network information
            network_info = {}
            for interface in self.network_interfaces:
                try:
                    if_addrs = psutil.net_if_addrs().get(interface, [])
                    ip_addresses = []
                    
                    for addr in if_addrs:
                        if addr.family == socket.AF_INET:
                            ip_addresses.append(addr.address)
                    
                    network_info[interface] = {
                        "ip_addresses": ip_addresses
                    }
                except Exception as e:
                    logger.error(f"Error getting network info for {interface}: {str(e)}")
            
            # Get GPU information if available
            gpu_info = {}
            if self._has_gpu():
                try:
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        gpu_info[f"gpu_{i}"] = {
                            "name": gpu.name,
                            "memory_total_gb": round(gpu.memoryTotal / 1024, 2),
                            "driver": gpu.driver,
                            "id": gpu.id,
                        }
                except Exception as e:
                    logger.error(f"Error getting GPU info: {str(e)}")
            
            # Get OS information
            os_info = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "hostname": socket.gethostname(),
            }
            
            # Get Python information
            python_info = {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu": cpu_info,
                "memory": memory_info,
                "disk": disk_info,
                "network": network_info,
                "gpu": gpu_info,
                "os": os_info,
                "python": python_info,
            }
        
        except Exception as e:
            logger.error(f"Error collecting system info: {str(e)}")
            return {"error": str(e)}
    
    def _get_cpu_model(self) -> str:
        """
        Get CPU model name.
        
        Returns:
            CPU model name as string
        """
        try:
            if platform.system() == "Windows":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                     r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                processor_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                winreg.CloseKey(key)
                return processor_name
            
            elif platform.system() == "Darwin":  # macOS
                import subprocess
                output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                return output
            
            else:  # Linux and other Unix-like systems
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
                return "Unknown CPU"
        
        except Exception:
            return platform.processor() or "Unknown CPU"
    
    def _get_cpu_frequency(self) -> Dict[str, float]:
        """
        Get CPU frequency information.
        
        Returns:
            Dictionary with CPU frequency information
        """
        try:
            freq = psutil.cpu_freq()
            if freq:
                return {
                    "current_mhz": round(freq.current, 2),
                    "min_mhz": round(freq.min, 2) if freq.min else None,
                    "max_mhz": round(freq.max, 2) if freq.max else None,
                }
            return {"current_mhz": None, "min_mhz": None, "max_mhz": None}
        except Exception:
            return {"current_mhz": None, "min_mhz": None, "max_mhz": None}
    
    def collect_cpu_metrics(self) -> Dict[str, Any]:
        """
        Collect CPU metrics.
        
        Returns:
            Dictionary with CPU metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_times = psutil.cpu_times_percent(interval=0.1)
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "overall_percent": round(sum(cpu_percent) / len(cpu_percent), 2),
                "per_cpu_percent": cpu_percent,
                "user_percent": cpu_times.user,
                "system_percent": cpu_times.system,
                "idle_percent": cpu_times.idle,
            }
            
            # Add additional fields if available
            if hasattr(cpu_times, "nice"):
                metrics["nice_percent"] = cpu_times.nice
            if hasattr(cpu_times, "iowait"):
                metrics["iowait_percent"] = cpu_times.iowait
            if hasattr(cpu_times, "irq"):
                metrics["irq_percent"] = cpu_times.irq
            if hasattr(cpu_times, "softirq"):
                metrics["softirq_percent"] = cpu_times.softirq
            
            # Add load averages on Unix-like systems
            if platform.system() != "Windows":
                metrics["load_avg_1min"], metrics["load_avg_5min"], metrics["load_avg_15min"] = os.getloadavg()
            
            self.cpu_history.append(metrics)
            return metrics
        
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {str(e)}")
            return {"error": str(e)}
    
    def collect_memory_metrics(self) -> Dict[str, Any]:
        """
        Collect memory metrics.
        
        Returns:
            Dictionary with memory metrics
        """
        try:
            virtual_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "virtual": {
                    "total_gb": round(virtual_memory.total / (1024**3), 2),
                    "available_gb": round(virtual_memory.available / (1024**3), 2),
                    "used_gb": round(virtual_memory.used / (1024**3), 2),
                    "free_gb": round(virtual_memory.free / (1024**3), 2),
                    "percent": virtual_memory.percent,
                },
                "swap": {
                    "total_gb": round(swap_memory.total / (1024**3), 2),
                    "used_gb": round(swap_memory.used / (1024**3), 2),
                    "free_gb": round(swap_memory.free / (1024**3), 2),
                    "percent": swap_memory.percent,
                }
            }
            
            # Add additional fields if available
            if hasattr(virtual_memory, "active"):
                metrics["virtual"]["active_gb"] = round(virtual_memory.active / (1024**3), 2)
            if hasattr(virtual_memory, "inactive"):
                metrics["virtual"]["inactive_gb"] = round(virtual_memory.inactive / (1024**3), 2)
            if hasattr(virtual_memory, "buffers"):
                metrics["virtual"]["buffers_gb"] = round(virtual_memory.buffers / (1024**3), 2)
            if hasattr(virtual_memory, "cached"):
                metrics["virtual"]["cached_gb"] = round(virtual_memory.cached / (1024**3), 2)
            
            self.memory_history.append(metrics)
            return metrics
        
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {str(e)}")
            return {"error": str(e)}
    
    def collect_disk_metrics(self) -> Dict[str, Any]:
        """
        Collect disk metrics.
        
        Returns:
            Dictionary with disk metrics
        """
        try:
            disk_metrics = {
                "timestamp": datetime.now().isoformat(),
                "partitions": {},
                "io": {},
            }
            
            # Get disk usage for monitored paths
            for disk_path in self.disk_paths:
                try:
                    disk_usage = psutil.disk_usage(disk_path)
                    disk_metrics["partitions"][disk_path] = {
                        "total_gb": round(disk_usage.total / (1024**3), 2),
                        "used_gb": round(disk_usage.used / (1024**3), 2),
                        "free_gb": round(disk_usage.free / (1024**3), 2),
                        "percent": disk_usage.percent,
                    }
                except Exception as e:
                    logger.error(f"Error getting disk usage for {disk_path}: {str(e)}")
            
            # Get disk I/O counters
            disk_io = psutil.disk_io_counters(perdisk=True)
            for disk_name, counters in disk_io.items():
                disk_metrics["io"][disk_name] = {
                    "read_count": counters.read_count,
                    "write_count": counters.write_count,
                    "read_bytes": counters.read_bytes,
                    "write_bytes": counters.write_bytes,
                    "read_time_ms": counters.read_time,
                    "write_time_ms": counters.write_time,
                }
            
            self.disk_history.append(disk_metrics)
            return disk_metrics
        
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {str(e)}")
            return {"error": str(e)}
    
    def collect_network_metrics(self) -> Dict[str, Any]:
        """
        Collect network metrics.
        
        Returns:
            Dictionary with network metrics
        """
        try:
            current_time = time.time()
            
            try:
                net_io_counters = psutil.net_io_counters(pernic=True)
            except Exception as e:
                logger.warning(f"Unable to get network IO counters: {str(e)}")
                net_io_counters = {}
            
            try:
                connections_count = len(psutil.net_connections())
            except Exception as e:
                logger.warning(f"Unable to get network connections: {str(e)}")
                connections_count = 0
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "interfaces": {},
                "connections_count": connections_count,
            }
            
            # Process each monitored interface
            for interface in self.network_interfaces:
                # Default metrics in case we can't get actual data
                interface_metrics = {
                    "bytes_sent": 0,
                    "bytes_recv": 0,
                    "packets_sent": 0,
                    "packets_recv": 0,
                    "errin": 0,
                    "errout": 0,
                    "dropin": 0,
                    "dropout": 0,
                    "send_rate_bytes_per_sec": 0,
                    "recv_rate_bytes_per_sec": 0,
                }
                
                # Try to get actual network data if available
                if interface in net_io_counters:
                    try:
                        counters = net_io_counters[interface]
                        
                        interface_metrics.update({
                            "bytes_sent": counters.bytes_sent,
                            "bytes_recv": counters.bytes_recv,
                            "packets_sent": counters.packets_sent,
                            "packets_recv": counters.packets_recv,
                            "errin": counters.errin,
                            "errout": counters.errout,
                            "dropin": counters.dropin,
                            "dropout": counters.dropout,
                        })
                        
                        # Calculate throughput if we have previous measurements
                        if (self.last_network_io is not None and 
                            self.last_network_time is not None and 
                            interface in self.last_network_io):
                            
                            time_diff = current_time - self.last_network_time
                            last_counters = self.last_network_io[interface]
                            
                            # Only calculate if time difference is significant to avoid division by zero
                            if time_diff > 0.1:
                                bytes_sent_diff = counters.bytes_sent - last_counters.bytes_sent
                                bytes_recv_diff = counters.bytes_recv - last_counters.bytes_recv
                                
                                # Handle potential negative values (counter resets)
                                if bytes_sent_diff >= 0:
                                    interface_metrics["send_rate_bytes_per_sec"] = round(bytes_sent_diff / time_diff, 2)
                                if bytes_recv_diff >= 0:
                                    interface_metrics["recv_rate_bytes_per_sec"] = round(bytes_recv_diff / time_diff, 2)
                    except Exception as e:
                        logger.warning(f"Error processing interface {interface} metrics: {str(e)}")
                
                metrics["interfaces"][interface] = interface_metrics
            
            # If we have no interfaces with data, try to detect active interfaces again
            if not metrics["interfaces"] and platform.system() != "Darwin":  # Skip auto-detection on macOS
                try:
                    updated_interfaces = self._detect_network_interfaces()
                    if updated_interfaces:
                        self.network_interfaces = updated_interfaces
                        logger.info(f"Updated network interfaces to: {self.network_interfaces}")
                except Exception as e:
                    logger.warning(f"Failed to re-detect network interfaces: {str(e)}")
            
            # Store current values for next calculation if available
            if net_io_counters:
                self.last_network_io = net_io_counters
                self.last_network_time = current_time
            
            self.network_history.append(metrics)
            return metrics
        
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}", exc_info=True)
            # Return a minimal valid data structure even on error
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "interfaces": {},
                "connections_count": 0,
                "error": str(e)
            }
            self.network_history.append(metrics)
            return metrics
    
    def collect_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Collect GPU metrics if available.
        
        Returns:
            Dictionary with GPU metrics or None if not available
        """
        if not self._has_gpu():
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "gpus": {},
            }
            
            for i, gpu in enumerate(gpus):
                metrics["gpus"][f"gpu_{i}"] = {
                    "name": gpu.name,
                    "load_percent": gpu.load * 100,
                    "memory_total_gb": round(gpu.memoryTotal / 1024, 2),
                    "memory_used_gb": round(gpu.memoryUsed / 1024, 2),
                    "memory_free_gb": round((gpu.memoryTotal - gpu.memoryUsed) / 1024, 2),
                    "memory_percent": round(gpu.memoryUtil * 100, 2),
                    "temperature_c": gpu.temperature,
                }
            
            if self.gpu_history is not None:
                self.gpu_history.append(metrics)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {str(e)}")
            return {"error": str(e)}
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collect all system metrics.
        
        Returns:
            Dictionary with all system metrics
        """
        cpu_metrics = self.collect_cpu_metrics()
        memory_metrics = self.collect_memory_metrics()
        disk_metrics = self.collect_disk_metrics()
        network_metrics = self.collect_network_metrics()
        gpu_metrics = self.collect_gpu_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": cpu_metrics,
            "memory": memory_metrics,
            "disk": disk_metrics,
            "network": network_metrics,
            "gpu": gpu_metrics,
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get a summary of system metrics.
        
        Returns:
            Dictionary with system metrics summary
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info,
        }
        
        # Add latest metrics if available
        if self.cpu_history:
            summary["cpu"] = self.cpu_history[-1]
        
        if self.memory_history:
            summary["memory"] = self.memory_history[-1]
        
        if self.disk_history:
            summary["disk"] = self.disk_history[-1]
        
        if self.network_history:
            summary["network"] = self.network_history[-1]
        
        if self.gpu_history and self.gpu_history:
            summary["gpu"] = self.gpu_history[-1]
        
        return summary
    
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert monitoring data to pandas DataFrames for analysis.
        
        Returns:
            Dictionary of DataFrames with monitoring data
        """
        dfs = {}
        
        if self.cpu_history:
            dfs["cpu"] = pd.DataFrame(list(self.cpu_history))
        else:
            dfs["cpu"] = pd.DataFrame()
            
        if self.memory_history:
            dfs["memory"] = pd.DataFrame(list(self.memory_history))
        else:
            dfs["memory"] = pd.DataFrame()
            
        if self.disk_history:
            dfs["disk"] = pd.DataFrame(list(self.disk_history))
        else:
            dfs["disk"] = pd.DataFrame()
            
        if self.network_history:
            dfs["network"] = pd.DataFrame(list(self.network_history))
        else:
            dfs["network"] = pd.DataFrame()
            
        if self.gpu_history:
            dfs["gpu"] = pd.DataFrame(list(self.gpu_history))
        else:
            dfs["gpu"] = pd.DataFrame()
        
        return dfs
    
    async def background_monitoring_task(self, interval: int = DEFAULT_INTERVAL):
        """
        Background task to continuously collect metrics.
        
        Args:
            interval: Interval in seconds between metric collections
        """
        while True:
            try:
                self.collect_all_metrics()
                logger.info(f"Collected system metrics successfully")
            except Exception as e:
                logger.error(f"Error in background monitoring: {str(e)}")
            
            await asyncio.sleep(interval)