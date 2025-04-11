"""
Tests for dashboard modules in OmniPulse.
"""

import unittest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

# Import dashboard modules
from src.dashboards import ollama_dashboard, system_dashboard, python_dashboard, custom_dashboard


class TestOllamaDashboard(unittest.TestCase):
    """Test the Ollama dashboard functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock monitor
        self.mock_monitor = MagicMock()
        self.mock_monitor.perform_health_check = AsyncMock(return_value={"healthy": True})
        self.mock_monitor.collect_all_metrics = AsyncMock(return_value={"models": []})
        self.mock_monitor.get_performance_stats = MagicMock(return_value={
            "total_requests": 100,
            "average_latency": 1.5,
            "average_throughput": 50.0,
            "success_rate": 98.5,
            "error_rate": 1.5
        })
        self.mock_monitor.get_token_usage_stats = MagicMock(return_value={
            "total_tokens": 10000,
            "prompt_tokens": 3000,
            "completion_tokens": 7000
        })
        
        # Create mock DataFrames
        self.mock_dfs = {
            "requests": pd.DataFrame({
                "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(10)],
                "model": ["model1", "model2"] * 5,
                "duration_seconds": np.random.uniform(0.5, 3.0, 10),
                "success": [True] * 9 + [False],
                "total_tokens": np.random.randint(100, 1000, 10),
                "prompt_tokens": np.random.randint(30, 300, 10),
                "completion_tokens": np.random.randint(70, 700, 10),
                "request_id": [f"req_{i}" for i in range(10)]
            }),
            "latency": pd.DataFrame({
                "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(10)],
                "model": ["model1", "model2"] * 5,
                "latency_seconds": np.random.uniform(0.5, 3.0, 10)
            }),
            "token_throughput": pd.DataFrame({
                "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(10)],
                "model": ["model1", "model2"] * 5,
                "tokens_per_second": np.random.uniform(30, 100, 10)
            })
        }
        self.mock_monitor.to_dataframe = MagicMock(return_value=self.mock_dfs)
        
        # Mock get_models method
        self.mock_monitor.get_models = AsyncMock(return_value=[
            {"name": "model1", "size": 1024 * 1024 * 1024 * 5, "modified_at": "2025-01-01", "format": "gguf"},
            {"name": "model2", "size": 1024 * 1024 * 1024 * 3, "modified_at": "2025-02-01", "format": "gguf"}
        ])
        
        # Mock get_model_info method
        self.mock_monitor.get_model_info = AsyncMock(return_value={
            "name": "model1",
            "size": 1024 * 1024 * 1024 * 5,
            "parameter_size": "7B",
            "format": "gguf",
            "metadata": {
                "family": "llama",
                "template": "chat",
                "quantization": "Q4_K_M",
                "context_length": 4096
            },
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        })
        
        # Mock simulate_request method
        self.mock_monitor.simulate_request = AsyncMock(return_value={
            "success": True,
            "response": "This is a test response",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "duration_seconds": 1.5
        })
        
        # Mock active_requests attribute
        self.mock_monitor.active_requests = {
            "req_1": {
                "model": "model1",
                "start_time": datetime.now().timestamp() - 10,
                "status": "in_progress"
            }
        }
    
    @patch("streamlit.subheader")
    @patch("streamlit.success")
    @patch("streamlit.tabs")
    async def test_render_dashboard_healthy(self, mock_tabs, mock_success, mock_subheader):
        """Test dashboard rendering when Ollama is healthy."""
        # Mock tabs object
        mock_tabs_obj = MagicMock()
        mock_tabs_obj.__iter__.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_tabs.return_value = mock_tabs_obj
        
        # Run the dashboard render function
        await ollama_dashboard.render_dashboard(self.mock_monitor, "Last hour")
        
        # Verify health check was called
        self.mock_monitor.perform_health_check.assert_called_once()
        
        # Verify success message was shown
        mock_success.assert_called_once()
        
        # Verify tabs were created
        mock_tabs.assert_called_once_with(["Overview", "Models", "Performance", "Requests", "Settings"])
    
    @patch("streamlit.subheader")
    @patch("streamlit.error")
    @patch("streamlit.expander")
    async def test_render_dashboard_unhealthy(self, mock_expander, mock_error, mock_subheader):
        """Test dashboard rendering when Ollama is unhealthy."""
        # Set health check to return unhealthy
        self.mock_monitor.perform_health_check = AsyncMock(return_value={"healthy": False, "error": "Connection refused"})
        
        # Run the dashboard render function
        await ollama_dashboard.render_dashboard(self.mock_monitor, "Last hour")
        
        # Verify health check was called
        self.mock_monitor.perform_health_check.assert_called_once()
        
        # Verify error message was shown
        mock_error.assert_called_once()
    
    @patch("streamlit.columns")
    @patch("streamlit.subheader")
    @patch("streamlit.spinner")
    async def test_render_overview_tab(self, mock_spinner, mock_subheader, mock_columns):
        """Test overview tab rendering."""
        # Mock columns
        mock_col = MagicMock()
        mock_columns.return_value = [mock_col, mock_col, mock_col, mock_col]
        
        # Run the overview tab render function
        await ollama_dashboard.render_overview_tab(self.mock_monitor, "Last hour")
        
        # Verify metrics collection was called
        self.mock_monitor.collect_all_metrics.assert_called_once()
        self.mock_monitor.get_performance_stats.assert_called_once()
        self.mock_monitor.get_token_usage_stats.assert_called_once()
    
    @patch("streamlit.selectbox")
    @patch("streamlit.spinner")
    @patch("streamlit.subheader")
    async def test_render_models_tab(self, mock_subheader, mock_spinner, mock_selectbox):
        """Test models tab rendering."""
        # Mock selectbox to return "model1"
        mock_selectbox.return_value = "model1"
        
        # Run the models tab render function
        await ollama_dashboard.render_models_tab(self.mock_monitor)
        
        # Verify models were fetched
        self.mock_monitor.get_models.assert_called_once()
        
        # Verify model info was fetched for the selected model
        self.mock_monitor.get_model_info.assert_called_once_with("model1")
    
    @patch("streamlit.columns")
    @patch("streamlit.subheader")
    @patch("streamlit.spinner")
    async def test_render_performance_tab(self, mock_spinner, mock_subheader, mock_columns):
        """Test performance tab rendering."""
        # Mock columns
        mock_col = MagicMock()
        mock_columns.return_value = [mock_col, mock_col, mock_col, mock_col]
        
        # Run the performance tab render function
        await ollama_dashboard.render_performance_tab(self.mock_monitor, "Last hour")
        
        # Verify performance stats were fetched
        self.mock_monitor.get_performance_stats.assert_called_once()
        self.mock_monitor.get_token_usage_stats.assert_called_once()
    
    def test_time_range_to_filter(self):
        """Test the time range conversion function."""
        self.assertEqual(ollama_dashboard.time_range_to_filter("Last 15 minutes"), "15m")
        self.assertEqual(ollama_dashboard.time_range_to_filter("Last hour"), "1h")
        self.assertEqual(ollama_dashboard.time_range_to_filter("Last 3 hours"), "3h")
        self.assertEqual(ollama_dashboard.time_range_to_filter("Last day"), "24h")
        self.assertEqual(ollama_dashboard.time_range_to_filter("Last week"), "7d")
        self.assertIsNone(ollama_dashboard.time_range_to_filter("Custom"))
    
    def test_filter_dataframe_by_time(self):
        """Test the DataFrame time filtering function."""
        # Create test DataFrame
        df = pd.DataFrame({
            "timestamp": [
                datetime.now() - timedelta(minutes=10),  # within 15 minutes
                datetime.now() - timedelta(minutes=30),  # within hour
                datetime.now() - timedelta(hours=2),     # within 3 hours
                datetime.now() - timedelta(hours=12),    # within day
                datetime.now() - timedelta(days=3)       # within week
            ],
            "value": [1, 2, 3, 4, 5]
        })
        
        # Test filtering
        self.assertEqual(len(ollama_dashboard.filter_dataframe_by_time(df, "Last 15 minutes")), 1)
        self.assertEqual(len(ollama_dashboard.filter_dataframe_by_time(df, "Last hour")), 2)
        self.assertEqual(len(ollama_dashboard.filter_dataframe_by_time(df, "Last 3 hours")), 3)
        self.assertEqual(len(ollama_dashboard.filter_dataframe_by_time(df, "Last day")), 4)
        self.assertEqual(len(ollama_dashboard.filter_dataframe_by_time(df, "Last week")), 5)


class TestSystemDashboard(unittest.TestCase):
    """Test the System dashboard functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock monitor
        self.mock_monitor = MagicMock()
        self.mock_monitor.system_info = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "cpu_count_physical": 8,
                "cpu_count_logical": 16,
                "cpu_model": "Intel Core i9",
                "cpu_frequency": {
                    "current_mhz": 2800,
                    "min_mhz": 2200,
                    "max_mhz": 3500
                }
            },
            "memory": {
                "total_memory_gb": 32,
                "available_memory_gb": 16
            },
            "disk": {
                "/": {
                    "total_gb": 500,
                    "used_gb": 250,
                    "free_gb": 250,
                    "percent_used": 50
                }
            },
            "network": {
                "eth0": {
                    "ip_addresses": ["192.168.1.100"]
                }
            }
        }
        
        # Create mock DataFrames
        self.mock_dfs = {
            "cpu": pd.DataFrame({
                "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(10)],
                "overall_percent": np.random.uniform(10, 90, 10),
                "user_percent": np.random.uniform(5, 60, 10),
                "system_percent": np.random.uniform(5, 30, 10),
                "idle_percent": np.random.uniform(10, 80, 10),
                "load_avg_1min": np.random.uniform(0, 5, 10),
                "load_avg_5min": np.random.uniform(0, 4, 10),
                "load_avg_15min": np.random.uniform(0, 3, 10)
            }),
            "memory": pd.DataFrame({
                "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(10)],
                "virtual": [
                    {
                        "total_gb": 32,
                        "available_gb": 16 - i,
                        "used_gb": 16 + i,
                        "free_gb": 16 - i,
                        "percent": 50 + i * 2
                    } for i in range(10)
                ],
                "swap": [
                    {
                        "total_gb": 8,
                        "used_gb": i * 0.5,
                        "free_gb": 8 - i * 0.5,
                        "percent": i * 6.25
                    } for i in range(10)
                ]
            }),
            "disk": pd.DataFrame({
                "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(10)],
                "partitions": [
                    {
                        "/": {
                            "total_gb": 500,
                            "used_gb": 250 + i,
                            "free_gb": 250 - i,
                            "percent": 50 + i * 0.2
                        }
                    } for i in range(10)
                ],
                "io": [
                    {
                        "sda": {
                            "read_bytes": 1024 * 1024 * 100 + i * 1024 * 1024,
                            "write_bytes": 1024 * 1024 * 50 + i * 1024 * 1024
                        }
                    } for i in range(10)
                ]
            }),
            "network": pd.DataFrame({
                "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(10)],
                "interfaces": [
                    {
                        "eth0": {
                            "bytes_sent": 1024 * 1024 * 100 + i * 1024 * 1024,
                            "bytes_recv": 1024 * 1024 * 200 + i * 1024 * 1024,
                            "send_rate_bytes_per_sec": 1024 * 100 + i * 1024,
                            "recv_rate_bytes_per_sec": 1024 * 200 + i * 1024
                        }
                    } for i in range(10)
                ],
                "connections_count": list(range(50, 60))
            })
        }
        self.mock_monitor.to_dataframe = MagicMock(return_value=self.mock_dfs)
        
        # Mock collect_all_metrics method
        self.mock_monitor.collect_all_metrics = MagicMock(return_value={
            "timestamp": datetime.now().isoformat(),
            "cpu": {"overall_percent": 45},
            "memory": {"virtual": {"percent": 60}},
            "disk": {"partitions": {"/": {"percent": 50}}},
            "network": {"interfaces": {"eth0": {"send_rate_bytes_per_sec": 1024 * 100, "recv_rate_bytes_per_sec": 1024 * 200}}}
        })
        
        # Mock disk_paths and network_interfaces attributes
        self.mock_monitor.disk_paths = ["/", "/home"]
        self.mock_monitor.network_interfaces = ["eth0", "wlan0"]
    
    @patch("streamlit.subheader")
    @patch("streamlit.tabs")
    def test_render_dashboard(self, mock_tabs, mock_subheader):
        """Test dashboard rendering."""
        # Mock tabs object
        mock_tabs_obj = MagicMock()
        mock_tabs_obj.__iter__.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_tabs.return_value = mock_tabs_obj
        
        # Run the dashboard render function
        system_dashboard.render_dashboard(self.mock_monitor, "Last hour")
        
        # Verify system info and metrics were collected
        self.mock_monitor.collect_all_metrics.assert_called_once()
        
        # Verify tabs were created
        mock_tabs.assert_called_once_with(["Overview", "CPU", "Memory", "Disk", "Network", "GPU", "Settings"])


class TestPythonDashboard(unittest.TestCase):
    """Test the Python dashboard functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock monitor
        self.mock_monitor = MagicMock()
        
        # Mock collect_python_processes method
        self.mock_monitor.collect_python_processes = MagicMock(return_value={
            "timestamp": datetime.now().isoformat(),
            "processes": [
                {
                    "pid": 1001,
                    "name": "python",
                    "cpu_percent": 10.5,
                    "memory_mb": 150.2,
                    "num_threads": 5,
                    "status": "running",
                    "framework": "streamlit",
                    "cmdline": "streamlit run app.py",
                    "username": "user"
                },
                {
                    "pid": 1002,
                    "name": "python",
                    "cpu_percent": 5.2,
                    "memory_mb": 200.5,
                    "num_threads": 8,
                    "status": "running",
                    "framework": "jupyter",
                    "cmdline": "jupyter notebook",
                    "username": "user"
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
        
        # Create mock DataFrames
        self.mock_dfs = {
            "processes": pd.DataFrame({
                "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(10)],
                "pid": [1001, 1002] * 5,
                "name": ["python"] * 10,
                "cpu_percent": np.random.uniform(5, 20, 10),
                "memory_mb": np.random.uniform(100, 300, 10),
                "status": ["running"] * 10,
                "num_threads": [5, 8] * 5,
                "framework": ["streamlit", "jupyter"] * 5
            })
        }
        self.mock_monitor.to_dataframe = MagicMock(return_value=self.mock_dfs)
        
        # Mock get_process_details method
        self.mock_monitor.get_process_details = MagicMock(return_value={
            "pid": 1001,
            "name": "python",
            "cpu_percent": 10.5,
            "memory_mb": 150.2,
            "memory_percent": 0.5,
            "num_threads": 5,
            "status": "running",
            "username": "user",
            "cmdline": "streamlit run app.py",
            "cwd": "/home/user/app",
            "create_time": (datetime.now() - timedelta(hours=2)).timestamp(),
            "num_fds": 8,
            "open_files": ["/home/user/app/app.py", "/home/user/app/data.csv"],
            "connections": [],
            "children": [],
            "packages": {"streamlit": "1.20.0", "pandas": "1.5.0", "numpy": "1.23.0"}
        })
        
        # Mock get_process_stats method
        self.mock_monitor.get_process_stats = MagicMock(return_value={
            "pid": 1001,
            "stats": [
                {
                    "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                    "cpu_percent": 10.5 - i * 0.5,
                    "memory_mb": 150.2 + i * 2,
                    "num_threads": 5,
                    "status": "running"
                } for i in range(10)
            ],
            "count": 10
        })
        
        # Mock get_processes_summary method
        self.mock_monitor.get_processes_summary = MagicMock(return_value={
            "total_count": 2,
            "total_memory_mb": 350.7,
            "total_cpu_percent": 15.7,
            "frameworks": {
                "streamlit": 1,
                "jupyter": 1
            }
        })
    
    @patch("streamlit.subheader")
    @patch("streamlit.tabs")
    def test_render_dashboard(self, mock_tabs, mock_subheader):
        """Test dashboard rendering."""
        # Mock tabs object
        mock_tabs_obj = MagicMock()
        mock_tabs_obj.__iter__.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_tabs.return_value = mock_tabs_obj
        
        # Run the dashboard render function
        python_dashboard.render_dashboard(self.mock_monitor, "Last hour")
        
        # Verify processes were collected
        self.mock_monitor.collect_python_processes.assert_called_once()
        
        # Verify tabs were created
        mock_tabs.assert_called_once_with(["Overview", "Processes", "Performance", "Settings"])


class TestCustomDashboard(unittest.TestCase):
    """Test the Custom dashboard functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock monitor
        self.mock_monitor = MagicMock()
        
        # Mock get_all_metrics_status method
        self.mock_monitor.get_all_metrics_status = MagicMock(return_value={
            "system_load": {
                "name": "system_load",
                "type": "command",
                "enabled": True,
                "interval": 60,
                "last_collection": datetime.now().isoformat(),
                "history_size": 10,
                "latest_status": "success"
            },
            "memory_usage": {
                "name": "memory_usage",
                "type": "command",
                "enabled": True,
                "interval": 60,
                "last_collection": datetime.now().isoformat(),
                "history_size": 10,
                "latest_status": "success"
            }
        })
        
        # Mock collect_all_metrics method
        self.mock_monitor.collect_all_metrics = MagicMock(return_value={
            "system_load": {
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "duration_seconds": 0.1,
                "data": {
                    "load_1m": 1.2,
                    "load_5m": 1.0,
                    "load_15m": 0.8
                }
            },
            "memory_usage": {
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "duration_seconds": 0.1,
                "data": {
                    "total": 32768,
                    "used": 16384,
                    "free": 16384,
                    "percent_used": 50.0
                }
            }
        })
        
        # Mock get_metric_history method
        self.mock_monitor.get_metric_history = MagicMock(return_value=[
            {
                "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                "success": True,
                "duration_seconds": 0.1,
                "data": {
                    "load_1m": 1.2 - i * 0.05,
                    "load_5m": 1.0 - i * 0.03,
                    "load_15m": 0.8 - i * 0.01
                }
            } for i in range(10)
        ])
        
        # Mock collect_metric method
        self.mock_monitor.collect_metric = AsyncMock(return_value={
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "duration_seconds": 0.1,
            "data": {
                "load_1m": 1.2,
                "load_5m": 1.0,
                "load_15m": 0.8
            }
        })
        
        # Mock add_metric, update_metric, and remove_metric methods
        self.mock_monitor.add_metric = MagicMock(return_value=True)
        self.mock_monitor.update_metric = MagicMock(return_value=True)
        self.mock_monitor.remove_metric = MagicMock(return_value=True)
    
    @patch("streamlit.subheader")
    @patch("streamlit.tabs")
    @patch("streamlit.session_state", {})
    def test_render_dashboard_with_metrics(self, mock_tabs, mock_subheader):
        """Test dashboard rendering with metrics."""
        # Mock tabs object
        mock_tabs_obj = MagicMock()
        mock_tabs_obj.__iter__.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_tabs.return_value = mock_tabs_obj
        
        # Run the dashboard render function
        custom_dashboard.render_dashboard(self.mock_monitor, "Last hour")
        
        # Verify metrics status was fetched
        self.mock_monitor.get_all_metrics_status.assert_called_once()
        
        # Verify tabs were created
        mock_tabs.assert_called_once_with(["Dashboard", "Metrics Explorer", "Configure"])
    
    @patch("streamlit.subheader")
    @patch("streamlit.info")
    @patch("streamlit.markdown")
    @patch("streamlit.expander")
    def test_render_empty_dashboard(self, mock_expander, mock_markdown, mock_info, mock_subheader):
        """Test empty dashboard rendering."""
        # Set get_all_metrics_status to return empty dict
        self.mock_monitor.get_all_metrics_status = MagicMock(return_value={})
        
        # Run the empty dashboard render function
        custom_dashboard.render_empty_dashboard(self.mock_monitor)
        
        # Verify info message was shown
        mock_info.assert_called_once()
        
        # Verify quick setup section was shown
        mock_markdown.assert_called()


if __name__ == "__main__":
    unittest.main()