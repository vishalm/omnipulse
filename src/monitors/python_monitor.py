"""
Python applications monitoring module for tracking Python processes and their performance.
"""

import psutil
import time
import logging
import os
import json
import subprocess
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import deque
import pandas as pd
import asyncio
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("python_monitor")

# Constants
MAX_HISTORY_SIZE = 1000  # Maximum number of data points to keep in memory
DEFAULT_INTERVAL = 10  # Default collection interval in seconds


class PythonMonitor:
    """
    Monitor for Python applications and processes.
    """
    
    def __init__(self, 
                 max_history: int = MAX_HISTORY_SIZE,
                 process_name_filter: Optional[List[str]] = None,
                 include_streamlit: bool = True,
                 include_jupyter: bool = True):
        """
        Initialize the Python applications monitor.
        
        Args:
            max_history: Maximum number of historical data points to keep
            process_name_filter: Optional list of process names to monitor
            include_streamlit: Whether to include Streamlit processes
            include_jupyter: Whether to include Jupyter processes
        """
        self.max_history = max_history
        self.process_name_filter = process_name_filter or []
        
        # Add common Python frameworks to the filter if specified
        if include_streamlit:
            self.process_name_filter.extend(["streamlit", "streamlit.cli"])
        if include_jupyter:
            self.process_name_filter.extend(["jupyter", "jupyter-notebook", "jupyter-lab"])
        
        # Initialize data structures to store monitoring data
        self.processes_history = deque(maxlen=max_history)
        self.active_processes = {}  # Map of pid -> process info
        
        # Map of package imports for each monitored process 
        # {pid: {'package_name': 'version'}}
        self.process_packages = {}
        
        # Store installed packages
        self.installed_packages = self._get_installed_packages()
        
        logger.info(f"Initialized Python monitor")
        logger.info(f"Monitoring processes: {self.process_name_filter}")
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """
        Get installed Python packages.
        
        Returns:
            Dictionary of installed packages and their versions
        """
        try:
            # Execute pip list and parse output
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            packages = json.loads(result.stdout)
            return {pkg["name"].lower(): pkg["version"] for pkg in packages}
        
        except Exception as e:
            logger.error(f"Error getting installed packages: {str(e)}")
            return {}
    
    def _is_python_process(self, proc: psutil.Process) -> bool:
        """
        Check if a process is a Python process.
        
        Args:
            proc: Process to check
            
        Returns:
            True if it's a Python process, False otherwise
        """
        try:
            # Check by executable name
            if proc.name().lower() in ["python", "python3", "python.exe", "pythonw.exe"]:
                return True
            
            # Check command line if accessible
            cmdline = proc.cmdline()
            if cmdline and any("python" in cmd.lower() for cmd in cmdline):
                return True
            
            # Check by examining open files (might be resource-intensive)
            # for file in proc.open_files():
            #     if file.path.endswith('.py'):
            #         return True
                
            return False
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return False
        except Exception as e:
            logger.debug(f"Error checking if process is Python: {str(e)}")
            return False
    
    def _is_monitored_process(self, proc: psutil.Process, cmdline: List[str]) -> bool:
        """
        Check if a process should be monitored based on process name filter.
        
        Args:
            proc: Process to check
            cmdline: Command line of the process
            
        Returns:
            True if it should be monitored, False otherwise
        """
        try:
            # If no filter is specified, monitor all Python processes
            if not self.process_name_filter:
                return True
            
            # Check if process name matches any filter
            if any(filter_name.lower() in proc.name().lower() for filter_name in self.process_name_filter):
                return True
            
            # Check if any command line argument matches any filter
            if cmdline and any(any(filter_name.lower() in cmd.lower() for filter_name in self.process_name_filter) for cmd in cmdline):
                return True
            
            return False
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return False
        except Exception as e:
            logger.debug(f"Error checking if process should be monitored: {str(e)}")
            return False
    
    def _get_process_packages(self, pid: int) -> Dict[str, str]:
        """
        Get packages imported by a Python process.
        This is an approximation and may not be accurate for all processes.
        
        Args:
            pid: Process ID
            
        Returns:
            Dictionary of imported packages and their versions
        """
        # Check if we already have package info for this process
        if pid in self.process_packages:
            return self.process_packages[pid]
        
        # This is a best-effort attempt and may not work for all processes
        try:
            proc = psutil.Process(pid)
            
            # Try to infer packages from command line
            cmdline = proc.cmdline()
            packages = {}
            
            for cmd in cmdline:
                if cmd.endswith('.py'):
                    try:
                        # Read the Python file and look for imports
                        if os.path.exists(cmd):
                            with open(cmd, 'r') as f:
                                content = f.read()
                                
                                # Very basic import detection - not foolproof
                                for line in content.split('\n'):
                                    line = line.strip()
                                    if line.startswith('import ') or line.startswith('from '):
                                        parts = line.split(' ')
                                        if len(parts) >= 2:
                                            package = parts[1].split('.')[0]
                                            if package in self.installed_packages:
                                                packages[package] = self.installed_packages[package]
                    except Exception:
                        pass
            
            # Look for known frameworks
            if any('streamlit' in cmd for cmd in cmdline):
                packages['streamlit'] = self.installed_packages.get('streamlit', 'unknown')
            if any('jupyter' in cmd for cmd in cmdline):
                packages['jupyter'] = self.installed_packages.get('jupyter', 'unknown')
            if any('flask' in cmd for cmd in cmdline):
                packages['flask'] = self.installed_packages.get('flask', 'unknown')
            if any('django' in cmd for cmd in cmdline):
                packages['django'] = self.installed_packages.get('django', 'unknown')
            
            # Cache the result
            self.process_packages[pid] = packages
            return packages
        
        except Exception as e:
            logger.debug(f"Error getting process packages for PID {pid}: {str(e)}")
            return {}
    
    def collect_python_processes(self) -> Dict[str, Any]:
        """
        Collect information about running Python processes.
        
        Returns:
            Dictionary with Python process information
        """
        try:
            current_time = datetime.now()
            processes_info = {
                "timestamp": current_time.isoformat(),
                "processes": [],
                "summary": {
                    "total_count": 0,
                    "total_memory_mb": 0,
                    "total_cpu_percent": 0,
                    "frameworks": {}
                }
            }
            
            # Track current PIDs to detect terminated processes
            current_pids = set()
            
            # Iterate over all processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username']):
                try:
                    # Skip non-Python processes quickly
                    if not self._is_python_process(proc):
                        continue
                    
                    # Get detailed information
                    pid = proc.pid
                    cmdline = proc.cmdline()
                    
                    # Check if this is a process we want to monitor
                    if not self._is_monitored_process(proc, cmdline):
                        continue
                    
                    current_pids.add(pid)
                    
                    # Get process details
                    try:
                        proc_info = proc.as_dict(attrs=[
                            'pid', 'name', 'username', 'create_time', 
                            'cpu_percent', 'memory_percent', 'memory_info',
                            'status', 'num_threads', 'connections'
                        ])
                        
                        # Additional process information
                        proc_info['cmdline'] = ' '.join(cmdline) if cmdline else ''
                        proc_info['memory_mb'] = proc_info['memory_info'].rss / (1024 * 1024) if proc_info['memory_info'] else 0
                        
                        # Get process working directory
                        try:
                            proc_info['cwd'] = proc.cwd()
                        except:
                            proc_info['cwd'] = 'Unknown'
                        
                        # Get open files
                        try:
                            proc_info['open_files'] = [f.path for f in proc.open_files()]
                        except:
                            proc_info['open_files'] = []
                        
                        # Get environment variables (limited for security/privacy)
                        try:
                            environ = proc.environ()
                            proc_info['python_path'] = environ.get('PYTHONPATH', '')
                            proc_info['virtual_env'] = environ.get('VIRTUAL_ENV', '')
                        except:
                            proc_info['python_path'] = ''
                            proc_info['virtual_env'] = ''
                        
                        # Get imported packages
                        proc_info['packages'] = self._get_process_packages(pid)
                        
                        # Determine framework being used
                        framework = 'unknown'
                        if 'streamlit' in proc_info['cmdline'].lower() or 'streamlit' in proc_info['packages']:
                            framework = 'streamlit'
                        elif 'jupyter' in proc_info['cmdline'].lower() or 'jupyter' in proc_info['packages']:
                            framework = 'jupyter'
                        elif 'flask' in proc_info['cmdline'].lower() or 'flask' in proc_info['packages']:
                            framework = 'flask'
                        elif 'django' in proc_info['cmdline'].lower() or 'django' in proc_info['packages']:
                            framework = 'django'
                        elif 'fastapi' in proc_info['cmdline'].lower() or 'fastapi' in proc_info['packages']:
                            framework = 'fastapi'
                        
                        proc_info['framework'] = framework
                        
                        # Add to summary counts
                        processes_info['summary']['total_count'] += 1
                        processes_info['summary']['total_memory_mb'] += proc_info['memory_mb']
                        processes_info['summary']['total_cpu_percent'] += proc_info['cpu_percent']
                        
                        # Update framework counts
                        if framework not in processes_info['summary']['frameworks']:
                            processes_info['summary']['frameworks'][framework] = 0
                        processes_info['summary']['frameworks'][framework] += 1
                        
                        # Add to processes list
                        processes_info['processes'].append(proc_info)
                        
                        # Update active processes cache
                        self.active_processes[pid] = proc_info
                    
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                except Exception as e:
                    logger.debug(f"Error processing Python process: {str(e)}")
                    continue
            
            # Remove terminated processes from active processes cache
            terminated_pids = set(self.active_processes.keys()) - current_pids
            for pid in terminated_pids:
                if pid in self.active_processes:
                    del self.active_processes[pid]
                if pid in self.process_packages:
                    del self.process_packages[pid]
            
            # Round summary values
            processes_info['summary']['total_memory_mb'] = round(processes_info['summary']['total_memory_mb'], 2)
            processes_info['summary']['total_cpu_percent'] = round(processes_info['summary']['total_cpu_percent'], 2)
            
            # Add to history
            self.processes_history.append(processes_info)
            
            return processes_info
        
        except Exception as e:
            logger.error(f"Error collecting Python processes: {str(e)}")
            return {"error": str(e)}
    
    def get_process_details(self, pid: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific Python process.
        
        Args:
            pid: Process ID
            
        Returns:
            Dictionary with detailed process information
        """
        try:
            proc = psutil.Process(pid)
            
            # Check if it's a Python process
            if not self._is_python_process(proc):
                return {"error": "Not a Python process"}
            
            # Get detailed information
            proc_info = proc.as_dict(attrs=[
                'pid', 'name', 'username', 'create_time', 
                'cpu_percent', 'memory_percent', 'memory_info',
                'status', 'num_threads', 'connections', 'cpu_times',
                'io_counters', 'num_ctx_switches', 'nice', 'num_fds'
            ])
            
            cmdline = proc.cmdline()
            proc_info['cmdline'] = ' '.join(cmdline) if cmdline else ''
            proc_info['memory_mb'] = proc_info['memory_info'].rss / (1024 * 1024) if proc_info['memory_info'] else 0
            
            # Get process working directory
            try:
                proc_info['cwd'] = proc.cwd()
            except:
                proc_info['cwd'] = 'Unknown'
            
            # Get open files
            try:
                proc_info['open_files'] = [f.path for f in proc.open_files()]
            except:
                proc_info['open_files'] = []
            
            # Get environment variables (limited for security/privacy)
            try:
                environ = proc.environ()
                proc_info['python_path'] = environ.get('PYTHONPATH', '')
                proc_info['virtual_env'] = environ.get('VIRTUAL_ENV', '')
            except:
                proc_info['python_path'] = ''
                proc_info['virtual_env'] = ''
            
            # Get imported packages
            proc_info['packages'] = self._get_process_packages(pid)
            
            # Get children processes
            try:
                proc_info['children'] = [child.pid for child in proc.children()]
            except:
                proc_info['children'] = []
            
            # Get parent process
            try:
                parent = proc.parent()
                proc_info['parent'] = parent.pid if parent else None
            except:
                proc_info['parent'] = None
            
            return proc_info
        
        except psutil.NoSuchProcess:
            return {"error": "Process does not exist"}
        except psutil.AccessDenied:
            return {"error": "Access denied to process information"}
        except Exception as e:
            logger.error(f"Error getting process details for PID {pid}: {str(e)}")
            return {"error": str(e)}
    
    def get_process_stats(self, pid: int) -> Dict[str, Any]:
        """
        Get statistics about a specific Python process over time.
        
        Args:
            pid: Process ID
            
        Returns:
            Dictionary with process statistics
        """
        # Extract statistics for this process from history
        process_stats = []
        
        for entry in self.processes_history:
            for proc in entry['processes']:
                if proc['pid'] == pid:
                    process_stats.append({
                        'timestamp': entry['timestamp'],
                        'cpu_percent': proc['cpu_percent'],
                        'memory_mb': proc['memory_mb'],
                        'num_threads': proc['num_threads'],
                        'status': proc['status']
                    })
        
        return {
            'pid': pid,
            'stats': process_stats,
            'count': len(process_stats)
        }
    
    def get_processes_summary(self, framework: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of Python processes.
        
        Args:
            framework: Optional framework filter
            
        Returns:
            Dictionary with process summary
        """
        if not self.processes_history:
            return {
                "total_count": 0,
                "frameworks": {},
                "total_memory_mb": 0,
                "total_cpu_percent": 0,
            }
        
        # Get the latest entry
        latest = self.processes_history[-1]
        
        # Filter by framework if specified
        if framework:
            filtered_processes = [p for p in latest['processes'] if p.get('framework') == framework]
            
            return {
                "total_count": len(filtered_processes),
                "total_memory_mb": round(sum(p['memory_mb'] for p in filtered_processes), 2),
                "total_cpu_percent": round(sum(p['cpu_percent'] for p in filtered_processes), 2),
                "framework": framework
            }
        
        return latest['summary']
    
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert monitoring data to pandas DataFrames for analysis.
        
        Returns:
            Dictionary of DataFrames with monitoring data
        """
        if not self.processes_history:
            return {"processes": pd.DataFrame()}
        
        # Flatten the process data into a format suitable for a DataFrame
        rows = []
        for entry in self.processes_history:
            timestamp = entry["timestamp"]
            
            for proc in entry["processes"]:
                row = {
                    "timestamp": timestamp,
                    "pid": proc["pid"],
                    "name": proc["name"],
                    "cpu_percent": proc["cpu_percent"],
                    "memory_mb": proc["memory_mb"],
                    "status": proc["status"],
                    "num_threads": proc["num_threads"],
                    "framework": proc.get("framework", "unknown"),
                }
                rows.append(row)
        
        return {"processes": pd.DataFrame(rows)}
    
    async def background_monitoring_task(self, interval: int = DEFAULT_INTERVAL):
        """
        Background task to continuously collect metrics.
        
        Args:
            interval: Interval in seconds between metric collections
        """
        while True:
            try:
                self.collect_python_processes()
                logger.info(f"Collected Python process metrics successfully")
            except Exception as e:
                logger.error(f"Error in background monitoring: {str(e)}")
            
            await asyncio.sleep(interval)