"""
Tests for monitor modules in OmniPulse.
"""

import unittest
import asyncio
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import httpx

# Import monitor modules
from src.monitors.ollama_monitor import OllamaMonitor
from src.monitors.system_monitor import SystemMonitor
from src.monitors.python_monitor import PythonMonitor
from src.monitors.custom_monitor import CustomMonitor


class TestOllamaMonitor(unittest.TestCase):
    """Test the Ollama monitor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock API client
        self.mock_client = MagicMock()
        self.mock_client.get = AsyncMock()
        self.mock_client.post = AsyncMock()
        self.mock_client.aclose = AsyncMock()
        
        # Create test monitor with mock client
        self.monitor = OllamaMonitor(api_url="http://test-api:11434")
        self.monitor.client = self.mock_client
    
    async def test_perform_health_check_success(self):
        """Test health check success."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        self.mock_client.get.return_value = mock_response
        
        # Call health check
        result = await self.monitor.perform_health_check()
        
        # Verify request was made correctly
        self.mock_client.get.assert_called_once_with("http://test-api:11434/api/tags")
        
        # Verify result
        self.assertTrue(result["healthy"])
        self.assertEqual(result["status_code"], 200)
    
    async def test_perform_health_check_failure(self):
        """Test health check failure."""
        # Set up mock to raise exception
        self.mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        
        # Call health check
        result = await self.monitor.perform_health_check()
        
        # Verify request was attempted
        self.mock_client.get.assert_called_once()
        
        # Verify result
        self.assertFalse(result["healthy"])
        self.assertIn("error", result)
    
    async def test_get_models(self):
        """Test getting models list."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "model1", "size": 5368709120, "modified_at": "2025-01-01", "format": "gguf"},
                {"name": "model2", "size": 3221225472, "modified_at": "2025-02-01", "format": "gguf"}
            ]
        }
        self.mock_client.get.return_value = mock_response
        
        # Call get models
        result = await self.monitor.get_models()
        
        # Verify request was made correctly
        self.mock_client.get.assert_called_once_with("http://test-api:11434/api/tags")
        
        # Verify result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "model1")
        self.assertEqual(result[1]["name"], "model2")
        
        # Verify models were cached
        self.assertEqual(len(self.monitor.models_info), 2)
        self.assertIn("model1", self.monitor.models_info)
        self.assertIn("model2", self.monitor.models_info)
    
    async def test_get_model_info(self):
        """Test getting specific model info."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "model1",
            "size": 5368709120,
            "parameter_size": "7B",
            "format": "gguf",
            "metadata": {
                "family": "llama",
                "template": "chat",
                "quantization": "Q4_K_M"
            }
        }
        self.mock_client.post.return_value = mock_response
        
        # Call get model info
        result = await self.monitor.get_model_info("model1")
        
        # Verify request was made correctly
        self.mock_client.post.assert_called_once_with(
            "http://test-api:11434/api/show",
            json={"name": "model1"}
        )
        
        # Verify result
        self.assertEqual(result["name"], "model1")
        self.assertEqual(result["parameter_size"], "7B")
        
        # Verify model info was cached
        self.assertIn("model1", self.monitor.models_info)
        self.assertEqual(self.monitor.models_info["model1"], result)
    
    async def test_simulate_request(self):
        """Test simulating a request."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "model1",
            "response": "This is a test response.",
        }
        self.mock_client.post.return_value = mock_response
        
        # Call simulate request
        start_time = datetime.now().timestamp()
        result = await self.monitor.simulate_request("model1", "Test prompt")
        
        # Verify request was made correctly
        self.mock_client.post.assert_called_once_with(
            "http://test-api:11434/api/generate",
            json={
                "model": "model1",
                "prompt": "Test prompt",
                "stream": False,
            },
            timeout=30
        )
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["model"], "model1")
        self.assertIn("request_id", result)
        self.assertIn("duration_seconds", result)
        self.assertIn("prompt_tokens", result)
        self.assertIn("completion_tokens", result)
        self.assertIn("total_tokens", result)
        
        # Verify metrics were recorded
        self.assertEqual(len(self.monitor.request_history), 1)
        self.assertEqual(len(self.monitor.token_throughput_history), 1)
        self.assertEqual(len(self.monitor.latency_history), 1)
    
    async def test_simulate_request_error(self):
        """Test simulating a request with an error."""
        # Set up mock to raise exception
        self.mock_client.post.side_effect = httpx.RequestError("Connection error")
        
        # Call simulate request
        result = await self.monitor.simulate_request("model1", "Test prompt")
        
        # Verify request was attempted
        self.mock_client.post.assert_called_once()
        
        # Verify result
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("duration_seconds", result)
        
        # Verify error was recorded
        self.assertEqual(len(self.monitor.error_history), 1)
    
    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        # Add some test data
        self.monitor.request_history.append({
            "request_id": "req1",
            "model": "model1",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": 1.5,
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "success": True
        })
        
        self.monitor.request_history.append({
            "request_id": "req2",
            "model": "model2",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": 2.0,
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
            "success": False
        })
        
        # Get stats
        stats = self.monitor.get_performance_stats()
        
        # Verify stats
        self.assertEqual(stats["total_requests"], 2)
        self.assertAlmostEqual(stats["average_latency"], 1.75, places=2)
        self.assertEqual(stats["success_rate"], 50.0)
        self.assertEqual(stats["error_rate"], 50.0)
        self.assertEqual(len(stats["requests_per_model"]), 2)
    
    def test_get_token_usage_stats(self):
        """Test getting token usage statistics."""
        # Add some test data
        self.monitor.request_history.append({
            "request_id": "req1",
            "model": "model1",
            "timestamp": datetime.now().isoformat(),
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "success": True
        })
        
        self.monitor.request_history.append({
            "request_id": "req2",
            "model": "model2",
            "timestamp": datetime.now().isoformat(),
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
            "success": True
        })
        
        # Get stats
        stats = self.monitor.get_token_usage_stats()
        
        # Verify stats
        self.assertEqual(stats["total_tokens"], 70)
        self.assertEqual(stats["prompt_tokens"], 25)
        self.assertEqual(stats["completion_tokens"], 45)
        self.assertEqual(len(stats["tokens_per_model"]), 2)
        self.assertEqual(stats["tokens_per_model"]["model1"], 30)
        self.assertEqual(stats["tokens_per_model"]["model2"], 40)
    
    def test_to_dataframe(self):
        """Test converting monitor data to DataFrames."""
        # Add some test data
        self.monitor.request_history.append({
            "request_id": "req1",
            "model": "model1",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": 1.5,
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "success": True
        })
        
        self.monitor.latency_history.append({
            "timestamp": datetime.now().isoformat(),
            "model": "model1",
            "latency_seconds": 1.5
        })
        
        self.monitor.token_throughput_history.append({
            "timestamp": datetime.now().isoformat(),
            "model": "model1",
            "tokens_per_second": 20.0
        })
        
        # Get DataFrames
        dfs = self.monitor.to_dataframe()
        
        # Verify DataFrames
        self.assertIn("requests", dfs)
        self.assertIn("latency", dfs)
        self.assertIn("token_throughput", dfs)
        self.assertEqual(len(dfs["requests"]), 1)
        self.assertEqual(len(dfs["latency"]), 1)
        self.assertEqual(len(dfs["token_throughput"]), 1)
    
    def tearDown(self):
        """Clean up after tests."""
        # Make sure to run the event loop to close the client
        asyncio.run(self.monitor.close())


class TestSystemMonitor(unittest.TestCase):
    """Test the System monitor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test monitor
        self.monitor = SystemMonitor(max_history=10)
    
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_times_percent")
    def test_collect_cpu_metrics(self, mock_cpu_times, mock_cpu_percent):
        """Test collecting CPU metrics."""
        # Set up mocks
        mock_cpu_percent.return_value = [10.0, 20.0, 30.0, 40.0]
        mock_cpu_times_return = MagicMock()
        mock_cpu_times_return.user = 30.0
        mock_cpu_times_return.system = 20.0
        mock_cpu_times_return.idle = 50.0
        mock_cpu_times.return_value = mock_cpu_times_return
        
        # Collect metrics
        metrics = self.monitor.collect_cpu_metrics()
        
        # Verify metrics
        self.assertIn("timestamp", metrics)
        self.assertEqual(metrics["overall_percent"], 25.0)  # Average of the values
        self.assertEqual(metrics["user_percent"], 30.0)
        self.assertEqual(metrics["system_percent"], 20.0)
        self.assertEqual(metrics["idle_percent"], 50.0)
        
        # Verify metrics were stored
        self.assertEqual(len(self.monitor.cpu_history), 1)
    
    @patch("psutil.virtual_memory")
    @patch("psutil.swap_memory")
    def test_collect_memory_metrics(self, mock_swap, mock_virtual):
        """Test collecting memory metrics."""
        # Set up mocks
        mock_virtual_return = MagicMock()
        mock_virtual_return.total = 1024 * 1024 * 1024 * 16  # 16 GB
        mock_virtual_return.available = 1024 * 1024 * 1024 * 8  # 8 GB
        mock_virtual_return.used = 1024 * 1024 * 1024 * 8  # 8 GB
        mock_virtual_return.free = 1024 * 1024 * 1024 * 8  # 8 GB
        mock_virtual_return.percent = 50.0
        mock_virtual.return_value = mock_virtual_return
        
        mock_swap_return = MagicMock()
        mock_swap_return.total = 1024 * 1024 * 1024 * 8  # 8 GB
        mock_swap_return.used = 1024 * 1024 * 1024 * 2  # 2 GB
        mock_swap_return.free = 1024 * 1024 * 1024 * 6  # 6 GB
        mock_swap_return.percent = 25.0
        mock_swap.return_value = mock_swap_return
        
        # Collect metrics
        metrics = self.monitor.collect_memory_metrics()
        
        # Verify metrics
        self.assertIn("timestamp", metrics)
        self.assertEqual(metrics["virtual"]["total_gb"], 16.0)
        self.assertEqual(metrics["virtual"]["available_gb"], 8.0)
        self.assertEqual(metrics["virtual"]["used_gb"], 8.0)
        self.assertEqual(metrics["virtual"]["free_gb"], 8.0)
        self.assertEqual(metrics["virtual"]["percent"], 50.0)
        
        self.assertEqual(metrics["swap"]["total_gb"], 8.0)
        self.assertEqual(metrics["swap"]["used_gb"], 2.0)
        self.assertEqual(metrics["swap"]["free_gb"], 6.0)
        self.assertEqual(metrics["swap"]["percent"], 25.0)
        
        # Verify metrics were stored
        self.assertEqual(len(self.monitor.memory_history), 1)
    
    @patch("psutil.disk_usage")
    @patch("psutil.disk_io_counters")
    def test_collect_disk_metrics(self, mock_io, mock_usage):
        """Test collecting disk metrics."""
        # Set up mocks
        mock_usage_return = MagicMock()
        mock_usage_return.total = 1024 * 1024 * 1024 * 500  # 500 GB
        mock_usage_return.used = 1024 * 1024 * 1024 * 250  # 250 GB
        mock_usage_return.free = 1024 * 1024 * 1024 * 250  # 250 GB
        mock_usage_return.percent = 50.0
        mock_usage.return_value = mock_usage_return
        
        mock_io.return_value = {
            "sda": MagicMock(
                read_count=1000,
                write_count=500,
                read_bytes=1024 * 1024 * 100,
                write_bytes=1024 * 1024 * 50,
                read_time=1000,
                write_time=500
            )
        }
        
        # Set monitor to track root path
        self.monitor.disk_paths = ["/"]
        
        # Collect metrics
        metrics = self.monitor.collect_disk_metrics()
        
        # Verify metrics
        self.assertIn("timestamp", metrics)
        self.assertIn("partitions", metrics)
        self.assertIn("/", metrics["partitions"])
        self.assertEqual(metrics["partitions"]["/"]["total_gb"], 500.0)
        self.assertEqual(metrics["partitions"]["/"]["used_gb"], 250.0)
        self.assertEqual(metrics["partitions"]["/"]["free_gb"], 250.0)
        self.assertEqual(metrics["partitions"]["/"]["percent"], 50.0)
        
        self.assertIn("io", metrics)
        self.assertIn("sda", metrics["io"])
        self.assertEqual(metrics["io"]["sda"]["read_count"], 1000)
        self.assertEqual(metrics["io"]["sda"]["write_count"], 500)
        
        # Verify metrics were stored
        self.assertEqual(len(self.monitor.disk_history), 1)
    
    @patch("psutil.net_io_counters")
    @patch("psutil.net_connections")
    def test_collect_network_metrics(self, mock_connections, mock_io):
        """Test collecting network metrics."""
        # Set up mocks
        mock_io.return_value = {
            "eth0": MagicMock(
                bytes_sent=1024 * 1024 * 100,
                bytes_recv=1024 * 1024 * 200,
                packets_sent=1000,
                packets_recv=2000,
                errin=0,
                errout=0,
                dropin=0,
                dropout=0
            )
        }
        
        mock_connections.return_value = [MagicMock(), MagicMock()]
        
        # Set monitor to track eth0
        self.monitor.network_interfaces = ["eth0"]
        
        # Collect metrics
        metrics = self.monitor.collect_network_metrics()
        
        # Verify metrics
        self.assertIn("timestamp", metrics)
        self.assertIn("interfaces", metrics)
        self.assertIn("eth0", metrics["interfaces"])
        self.assertEqual(metrics["interfaces"]["eth0"]["bytes_sent"], 1024 * 1024 * 100)
        self.assertEqual(metrics["interfaces"]["eth0"]["bytes_recv"], 1024 * 1024 * 200)
        self.assertEqual(metrics["connections_count"], 2)
        
        # Verify metrics were stored
        self.assertEqual(len(self.monitor.network_history), 1)
    
    def test_collect_all_metrics(self):
        """Test collecting all system metrics."""
        # Patch individual collection methods
        with patch.object(self.monitor, 'collect_cpu_metrics', return_value={"cpu": "metrics"}), \
             patch.object(self.monitor, 'collect_memory_metrics', return_value={"memory": "metrics"}), \
             patch.object(self.monitor, 'collect_disk_metrics', return_value={"disk": "metrics"}), \
             patch.object(self.monitor, 'collect_network_metrics', return_value={"network": "metrics"}), \
             patch.object(self.monitor, 'collect_gpu_metrics', return_value={"gpu": "metrics"}):
            
            # Collect all metrics
            metrics = self.monitor.collect_all_metrics()
            
            # Verify metrics
            self.assertIn("timestamp", metrics)
            self.assertEqual(metrics["cpu"], {"cpu": "metrics"})
            self.assertEqual(metrics["memory"], {"memory": "metrics"})
            self.assertEqual(metrics["disk"], {"disk": "metrics"})
            self.assertEqual(metrics["network"], {"network": "metrics"})
            self.assertEqual(metrics["gpu"], {"gpu": "metrics"})
    
    def test_to_dataframe(self):
        """Test converting monitor data to DataFrames."""
        # Add some test data
        self.monitor.cpu_history.append({
            "timestamp": datetime.now().isoformat(),
            "overall_percent": 25.0,
            "user_percent": 15.0,
            "system_percent": 10.0,
            "idle_percent": 75.0
        })
        
        self.monitor.memory_history.append({
            "timestamp": datetime.now().isoformat(),
            "virtual": {
                "total_gb": 16.0,
                "available_gb": 8.0,
                "used_gb": 8.0,
                "free_gb": 8.0,
                "percent": 50.0
            },
            "swap": {
                "total_gb": 8.0,
                "used_gb": 2.0,
                "free_gb": 6.0,
                "percent": 25.0
            }
        })
        
        # Get DataFrames
        dfs = self.monitor.to_dataframe()
        
        # Verify DataFrames
        self.assertIn("cpu", dfs)
        self.assertIn("memory", dfs)
        self.assertEqual(len(dfs["cpu"]), 1)
        self.assertEqual(len(dfs["memory"]), 1)


class TestPythonMonitor(unittest.TestCase):
    """Test the Python monitor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test monitor
        self.monitor = PythonMonitor(max_history=10, process_name_filter=["python", "streamlit"])
    
    @patch("psutil.process_iter")
    def test_collect_python_processes(self, mock_process_iter):
        """Test collecting Python processes."""
        # Set up mock processes
        mock_proc1 = MagicMock()
        mock_proc1.pid = 1001
        mock_proc1.name.return_value = "python"
        mock_proc1.username.return_value = "user"
        mock_proc1.create_time.return_value = datetime.now().timestamp() - 3600  # 1 hour ago
        mock_proc1.cpu_percent.return_value = 10.5
        mock_proc1.memory_percent.return_value = 2.0
        mock_proc1.memory_info.return_value = MagicMock(rss=1024 * 1024 * 150)  # 150 MB
        mock_proc1.status.return_value = "running"
        mock_proc1.num_threads.return_value = 5
        mock_proc1.connections.return_value = []
        mock_proc1.cmdline.return_value = ["python", "-m", "streamlit", "run", "app.py"]
        mock_proc1.as_dict.return_value = {
            "pid": 1001,
            "name": "python",
            "username": "user",
            "create_time": datetime.now().timestamp() - 3600,
            "cpu_percent": 10.5,
            "memory_percent": 2.0,
            "memory_info": MagicMock(rss=1024 * 1024 * 150),
            "status": "running",
            "num_threads": 5,
            "connections": []
        }
        
        mock_proc2 = MagicMock()
        mock_proc2.pid = 1002
        mock_proc2.name.return_value = "non-python"
        mock_proc2.cmdline.return_value = ["non-python"]
        
        mock_process_iter.return_value = [mock_proc1, mock_proc2]
        
        # Mock is_python_process and is_monitored_process methods
        with patch.object(self.monitor, '_is_python_process', side_effect=lambda proc: proc.name() == "python"), \
             patch.object(self.monitor, '_is_monitored_process', return_value=True), \
             patch.object(self.monitor, '_get_process_packages', return_value={"streamlit": "1.20.0"}):
            
            # Collect Python processes
            result = self.monitor.collect_python_processes()
            
            # Verify result
            self.assertIn("timestamp", result)
            self.assertIn("processes", result)
            self.assertEqual(len(result["processes"]), 1)
            self.assertEqual(result["processes"][0]["pid"], 1001)
            self.assertEqual(result["processes"][0]["name"], "python")
            self.assertEqual(result["processes"][0]["cpu_percent"], 10.5)
            self.assertAlmostEqual(result["processes"][0]["memory_mb"], 150.0, places=1)
            
            self.assertIn("summary", result)
            self.assertEqual(result["summary"]["total_count"], 1)
            
            # Verify data was stored
            self.assertEqual(len(self.monitor.processes_history), 1)
    
    def test_get_process_details(self):
        """Test getting detailed process information."""
        # Mock Process class
        with patch("psutil.Process") as mock_process_class:
            # Set up mock process
            mock_proc = MagicMock()
            mock_proc.pid = 1001
            mock_proc.name.return_value = "python"
            mock_proc.username.return_value = "user"
            mock_proc.create_time.return_value = datetime.now().timestamp() - 3600  # 1 hour ago
            mock_proc.cpu_percent.return_value = 10.5
            mock_proc.memory_percent.return_value = 2.0
            mock_proc.memory_info.return_value = MagicMock(rss=1024 * 1024 * 150)  # 150 MB
            mock_proc.status.return_value = "running"
            mock_proc.num_threads.return_value = 5
            mock_proc.connections.return_value = []
            mock_proc.cmdline.return_value = ["python", "-m", "streamlit", "run", "app.py"]
            mock_proc.cwd.return_value = "/home/user/app"
            mock_proc.open_files.return_value = [MagicMock(path="/home/user/app/app.py")]
            mock_proc.environ.return_value = {"PYTHONPATH": "/usr/lib/python", "VIRTUAL_ENV": "/home/user/venv"}
            mock_proc.as_dict.return_value = {
                "pid": 1001,
                "name": "python",
                "username": "user",
                "create_time": datetime.now().timestamp() - 3600,
                "cpu_percent": 10.5,
                "memory_percent": 2.0,
                "memory_info": MagicMock(rss=1024 * 1024 * 150),
                "status": "running",
                "num_threads": 5,
                "connections": []
            }
            
            # Return mock process from Process constructor
            mock_process_class.return_value = mock_proc
            
            # Mock is_python_process
            with patch.object(self.monitor, '_is_python_process', return_value=True), \
                 patch.object(self.monitor, '_get_process_packages', return_value={"streamlit": "1.20.0"}):
                
                # Get process details
                details = self.monitor.get_process_details(1001)
                
                # Verify result
                self.assertEqual(details["pid"], 1001)
                self.assertEqual(details["name"], "python")
                self.assertEqual(details["cpu_percent"], 10.5)
                self.assertEqual(details["memory_percent"], 2.0)
                self.assertAlmostEqual(details["memory_mb"], 150.0, places=1)
                self.assertEqual(details["status"], "running")
                self.assertEqual(details["num_threads"], 5)
                self.assertEqual(details["cwd"], "/home/user/app")
                self.assertEqual(details["open_files"], ["/home/user/app/app.py"])
                self.assertEqual(details["python_path"], "/usr/lib/python")
                self.assertEqual(details["virtual_env"], "/home/user/venv")
                self.assertEqual(details["packages"], {"streamlit": "1.20.0"})
    
    def test_get_processes_summary(self):
        """Test getting process summary."""
        # Add some test data
        self.monitor.processes_history.append({
            "timestamp": datetime.now().isoformat(),
            "processes": [
                {
                    "pid": 1001,
                    "name": "python",
                    "cpu_percent": 10.5,
                    "memory_mb": 150.2,
                    "num_threads": 5,
                    "status": "running",
                    "framework": "streamlit"
                },
                {
                    "pid": 1002,
                    "name": "python",
                    "cpu_percent": 5.2,
                    "memory_mb": 200.5,
                    "num_threads": 8,
                    "status": "running",
                    "framework": "jupyter"
                }
            ],
            "summary": {
                "total_count": 2,
                "total_memory_mb": 350.7,
                "total_cpu_percent": 15.7,
                "frameworks": {
                    "streamlit": 1,
                    "jupyter": 1
                }
            }
        })
        
        # Get summary
        summary = self.monitor.get_processes_summary()
        
        # Verify summary
        self.assertEqual(summary["total_count"], 2)
        self.assertAlmostEqual(summary["total_memory_mb"], 350.7, places=1)
        self.assertAlmostEqual(summary["total_cpu_percent"], 15.7, places=1)
        self.assertEqual(summary["frameworks"]["streamlit"], 1)
        self.assertEqual(summary["frameworks"]["jupyter"], 1)
        
        # Get summary filtered by framework
        streamlit_summary = self.monitor.get_processes_summary(framework="streamlit")
        
        # Verify filtered summary
        self.assertEqual(streamlit_summary["total_count"], 1)
        self.assertEqual(streamlit_summary["framework"], "streamlit")


class TestCustomMonitor(unittest.TestCase):
    """Test the Custom monitor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary config file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "custom_monitors.json")
        
        # Create test monitor
        self.monitor = CustomMonitor(config_path=self.config_path)
    
    def test_add_metric(self):
        """Test adding a custom metric."""
        # Define metric config
        metric_config = {
            "name": "system_load",
            "type": "command",
            "command": "cat /proc/loadavg",
            "parser": "text",
            "interval": 60,
            "enabled": True
        }
        
        # Add metric
        result = self.monitor.add_metric(metric_config)
        
        # Verify result
        self.assertTrue(result)
        self.assertIn("system_load", self.monitor.metric_configs)
        self.assertEqual(self.monitor.metric_configs["system_load"], metric_config)
    
    def test_update_metric(self):
        """Test updating a custom metric."""
        # Add a metric first
        self.monitor.metric_configs["system_load"] = {
            "name": "system_load",
            "type": "command",
            "command": "cat /proc/loadavg",
            "parser": "text",
            "interval": 60,
            "enabled": True
        }
        
        # Update metric
        update_config = {
            "command": "uptime",
            "interval": 120
        }
        
        result = self.monitor.update_metric("system_load", update_config)
        
        # Verify result
        self.assertTrue(result)
        self.assertEqual(self.monitor.metric_configs["system_load"]["command"], "uptime")
        self.assertEqual(self.monitor.metric_configs["system_load"]["interval"], 120)
        self.assertEqual(self.monitor.metric_configs["system_load"]["type"], "command")  # Unchanged
    
    def test_remove_metric(self):
        """Test removing a custom metric."""
        # Add a metric first
        self.monitor.metric_configs["system_load"] = {
            "name": "system_load",
            "type": "command",
            "command": "cat /proc/loadavg",
            "parser": "text",
            "interval": 60,
            "enabled": True
        }
        
        # Add history data
        self.monitor.metrics_history["system_load"] = deque(maxlen=10)
        self.monitor.metrics_history["system_load"].append({"data": "test"})
        self.monitor.last_collection_time["system_load"] = datetime.now()
        
        # Remove metric
        result = self.monitor.remove_metric("system_load")
        
        # Verify result
        self.assertTrue(result)
        self.assertNotIn("system_load", self.monitor.metric_configs)
        self.assertNotIn("system_load", self.monitor.metrics_history)
        self.assertNotIn("system_load", self.monitor.last_collection_time)
    
        
"""
Tests for monitor modules in OmniPulse.
"""

import unittest
import asyncio
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import httpx

# Import monitor modules
from src.monitors.ollama_monitor import OllamaMonitor
from src.monitors.system_monitor import SystemMonitor
from src.monitors.python_monitor import PythonMonitor
from src.monitors.custom_monitor import CustomMonitor


class TestOllamaMonitor(unittest.TestCase):
    """Test the Ollama monitor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock API client
        self.mock_client = MagicMock()
        self.mock_client.get = AsyncMock()
        self.mock_client.post = AsyncMock()
        self.mock_client.aclose = AsyncMock()
        
        # Create test monitor with mock client
        self.monitor = OllamaMonitor(api_url="http://test-api:11434")
        self.monitor.client = self.mock_client
    
    async def test_perform_health_check_success(self):
        """Test health check success."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        self.mock_client.get.return_value = mock_response
        
        # Call health check
        result = await self.monitor.perform_health_check()
        
        # Verify request was made correctly
        self.mock_client.get.assert_called_once_with("http://test-api:11434/api/tags")
        
        # Verify result
        self.assertTrue(result["healthy"])
        self.assertEqual(result["status_code"], 200)
    
    async def test_perform_health_check_failure(self):
        """Test health check failure."""
        # Set up mock to raise exception
        self.mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        
        # Call health check
        result = await self.monitor.perform_health_check()
        
        # Verify request was attempted
        self.mock_client.get.assert_called_once()
        
        # Verify result
        self.assertFalse(result["healthy"])
        self.assertIn("error", result)
    
    async def test_get_models(self):
        """Test getting models list."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "model1", "size": 5368709120, "modified_at": "2025-01-01", "format": "gguf"},
                {"name": "model2", "size": 3221225472, "modified_at": "2025-02-01", "format": "gguf"}
            ]
        }
        self.mock_client.get.return_value = mock_response
        
        # Call get models
        result = await self.monitor.get_models()
        
        # Verify request was made correctly
        self.mock_client.get.assert_called_once_with("http://test-api:11434/api/tags")
        
        # Verify result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "model1")
        self.assertEqual(result[1]["name"], "model2")
        
        # Verify models were cached
        self.assertEqual(len(self.monitor.models_info), 2)
        self.assertIn("model1", self.monitor.models_info)
        self.assertIn("model2", self.monitor.models_info)

if __name__ == "__main__":
    unittest.main()