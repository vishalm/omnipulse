"""
Ollama monitoring module for collecting metrics from Ollama LLM service.
"""

import httpx
import time
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from collections import deque
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ollama_monitor")

# Constants
DEFAULT_OLLAMA_API_URL = "http://localhost:11434"
MAX_HISTORY_SIZE = 1000  # Maximum number of data points to keep in memory


class OllamaMonitor:
    """
    Monitor for Ollama LLM service that collects various metrics.
    """
    
    def __init__(self, api_url: Optional[str] = None, max_history: int = MAX_HISTORY_SIZE):
        """
        Initialize the Ollama monitor.
        
        Args:
            api_url: URL of the Ollama API
            max_history: Maximum number of historical data points to keep
        """
        self.api_url = api_url or os.environ.get("OLLAMA_API_URL", DEFAULT_OLLAMA_API_URL)
        self.max_history = max_history
        
        # Initialize data structures to store monitoring data
        self.models_info = {}
        self.request_history = deque(maxlen=max_history)
        self.token_throughput_history = deque(maxlen=max_history)
        self.latency_history = deque(maxlen=max_history)
        self.error_history = deque(maxlen=max_history)
        self.resource_usage_history = deque(maxlen=max_history)
        
        # For active requests tracking
        self.active_requests = {}
        
        # Client for API calls
        self.client = httpx.AsyncClient(timeout=30.0)
        
        logger.info(f"Initialized Ollama monitor with API at {self.api_url}")
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = await self.client.get(f"{self.api_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json().get("models", [])
                
                # Update models info cache
                for model in models_data:
                    self.models_info[model["name"]] = model
                
                return models_data
            else:
                logger.error(f"Failed to get models: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error fetching models: {str(e)}")
            return []
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model to get info for
            
        Returns:
            Dictionary with model information
        """
        try:
            response = await self.client.post(
                f"{self.api_url}/api/show",
                json={"name": model_name}
            )
            if response.status_code == 200:
                model_info = response.json()
                # Cache the model info
                self.models_info[model_name] = model_info
                return model_info
            else:
                logger.error(f"Failed to get model info: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching model info: {str(e)}")
            return {}
    
    async def simulate_request(self, model_name: str, prompt: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Make a completion request to measure performance.
        
        Args:
            model_name: Name of the model to use
            prompt: Prompt to send
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with performance metrics
        """
        start_time = time.time()
        request_id = f"{int(start_time)}-{model_name}"
        
        try:
            # Record the start of the request
            self.active_requests[request_id] = {
                "model": model_name,
                "start_time": start_time,
                "status": "in_progress",
            }
            
            # Make the request
            response = await self.client.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Calculate metrics
                metrics = {
                    "request_id": request_id,
                    "model": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "duration_seconds": duration,
                    "prompt_tokens": len(prompt.split()),  # Rough estimate
                    "completion_tokens": len(result.get("response", "").split()),  # Rough estimate
                    "total_tokens": len(prompt.split()) + len(result.get("response", "").split()),
                    "success": True
                }
                
                # Update histories
                self.request_history.append(metrics)
                self.token_throughput_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "tokens_per_second": metrics["total_tokens"] / max(duration, 0.001)
                })
                self.latency_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "latency_seconds": duration
                })
                
                # Update active request status
                self.active_requests[request_id]["status"] = "completed"
                self.active_requests[request_id]["end_time"] = end_time
                self.active_requests[request_id]["metrics"] = metrics
                
                return metrics
            else:
                # Record error
                error_data = {
                    "request_id": request_id,
                    "model": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "error_code": response.status_code,
                    "error_message": response.text
                }
                self.error_history.append(error_data)
                
                # Update active request status
                self.active_requests[request_id]["status"] = "failed"
                self.active_requests[request_id]["error"] = error_data
                
                logger.error(f"Request failed: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text, "duration_seconds": duration}
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            # Record error
            error_data = {
                "request_id": request_id,
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "error_code": "Exception",
                "error_message": str(e)
            }
            self.error_history.append(error_data)
            
            # Update active request status
            self.active_requests[request_id]["status"] = "failed"
            self.active_requests[request_id]["error"] = error_data
            
            logger.error(f"Exception during request: {str(e)}")
            return {"success": False, "error": str(e), "duration_seconds": duration}
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect system metrics from Ollama service.
        
        Returns:
            Dictionary with system metrics
        """
        # Note: Ollama doesn't have a built-in metrics endpoint
        # This is a placeholder - in a real implementation, you might
        # want to query system metrics from the host running Ollama
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": None,  # Placeholder
            "memory_usage": None,  # Placeholder
            "gpu_usage": None,  # Placeholder
        }
        
        self.resource_usage_history.append(metrics)
        return metrics
    
    async def perform_health_check(self) -> Dict[str, bool]:
        """
        Check if Ollama service is healthy.
        
        Returns:
            Dictionary with health status
        """
        # try:
        #     # Ollama doesn't have a dedicated health endpoint, so we use /api/tags
        #     response = await self.client.get(f"{self.api_url}/api/tags")
        #     return {
        #         "timestamp": datetime.now().isoformat(),
        #         "healthy": response.status_code == 200,
        #         "status_code": response.status_code
        #     }
        # except Exception as e:

        try:
        # Use a new event loop for each health check
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.api_url}/api/tags")
                if response.status_code == 200:
                    return {"healthy": True}
                else:
                    return {"healthy": False, "error": f"HTTP error {response.status_code}"}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "healthy": False,
                "error": str(e)
            }
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collect all metrics in one go.
        
        Returns:
            Dictionary with all metrics
        """
        health = await self.perform_health_check()
        
        if not health.get("healthy", False):
            logger.warning("Ollama service is not healthy, skipping metrics collection")
            return {
                "health": health,
                "models": [],
                "system_metrics": None
            }
        
        models = await self.get_models()
        system_metrics = await self.collect_system_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "health": health,
            "models": models,
            "system_metrics": system_metrics
        }
    
    def get_performance_stats(self, time_range: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate performance statistics based on collected metrics.
        
        Args:
            time_range: Optional time range filter (e.g., "1h", "24h")
            
        Returns:
            Dictionary with performance statistics
        """
        # Convert histories to dataframes for easier analysis
        if not self.request_history:
            return {
                "total_requests": 0,
                "average_latency": 0,
                "average_throughput": 0,
                "success_rate": 0,
                "requests_per_model": {},
                "error_rate": 0
            }
        
        df_requests = pd.DataFrame(list(self.request_history))
        df_errors = pd.DataFrame(list(self.error_history)) if self.error_history else pd.DataFrame()
        
        # Apply time range filter if specified
        if time_range and df_requests.shape[0] > 0:
            # Convert time_range to timedelta
            if time_range == "1h":
                cutoff = pd.Timestamp.now() - pd.Timedelta(hours=1)
            elif time_range == "24h":
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=1)
            else:
                # Default to all data
                cutoff = pd.Timestamp.min
            
            if "timestamp" in df_requests.columns:
                df_requests["timestamp"] = pd.to_datetime(df_requests["timestamp"])
                df_requests = df_requests[df_requests["timestamp"] >= cutoff]
            
            if not df_errors.empty and "timestamp" in df_errors.columns:
                df_errors["timestamp"] = pd.to_datetime(df_errors["timestamp"]) 
                df_errors = df_errors[df_errors["timestamp"] >= cutoff]
        
        # Calculate statistics
        total_requests = len(df_requests)
        
        if total_requests > 0:
            avg_latency = df_requests["duration_seconds"].mean() if "duration_seconds" in df_requests.columns else 0
            
            if "total_tokens" in df_requests.columns and "duration_seconds" in df_requests.columns:
                df_requests["tokens_per_second"] = df_requests["total_tokens"] / df_requests["duration_seconds"].clip(lower=0.001)
                avg_throughput = df_requests["tokens_per_second"].mean()
            else:
                avg_throughput = 0
            
            if "success" in df_requests.columns:
                success_rate = df_requests["success"].mean() * 100
            else:
                success_rate = 0
            
            # Requests per model
            if "model" in df_requests.columns:
                requests_per_model = df_requests["model"].value_counts().to_dict()
            else:
                requests_per_model = {}
            
            # Error rate
            error_rate = (len(df_errors) / total_requests * 100) if total_requests > 0 else 0
        else:
            avg_latency = 0
            avg_throughput = 0
            success_rate = 0
            requests_per_model = {}
            error_rate = 0
        
        return {
            "total_requests": total_requests,
            "average_latency": avg_latency,
            "average_throughput": avg_throughput,
            "success_rate": success_rate,
            "requests_per_model": requests_per_model,
            "error_rate": error_rate
        }
    
    def get_token_usage_stats(self, time_range: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate token usage statistics.
        
        Args:
            time_range: Optional time range filter (e.g., "1h", "24h")
            
        Returns:
            Dictionary with token usage statistics
        """
        if not self.request_history:
            return {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tokens_per_model": {}
            }
        
        df = pd.DataFrame(list(self.request_history))
        
        # Apply time range filter if specified
        if time_range and df.shape[0] > 0:
            # Convert time_range to timedelta
            if time_range == "1h":
                cutoff = pd.Timestamp.now() - pd.Timedelta(hours=1)
            elif time_range == "24h":
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=1)
            else:
                # Default to all data
                cutoff = pd.Timestamp.min
            
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df[df["timestamp"] >= cutoff]
        
        # Calculate statistics
        if df.shape[0] > 0:
            total_tokens = df["total_tokens"].sum() if "total_tokens" in df.columns else 0
            prompt_tokens = df["prompt_tokens"].sum() if "prompt_tokens" in df.columns else 0
            completion_tokens = df["completion_tokens"].sum() if "completion_tokens" in df.columns else 0
            
            # Tokens per model
            if "model" in df.columns and "total_tokens" in df.columns:
                tokens_per_model = df.groupby("model")["total_tokens"].sum().to_dict()
            else:
                tokens_per_model = {}
        else:
            total_tokens = 0
            prompt_tokens = 0
            completion_tokens = 0
            tokens_per_model = {}
        
        return {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_model": tokens_per_model
        }
    
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert monitoring data to pandas DataFrames for analysis.
        
        Returns:
            Dictionary of DataFrames with monitoring data
        """
        dfs = {}
        
        if self.request_history:
            dfs["requests"] = pd.DataFrame(list(self.request_history))
        else:
            dfs["requests"] = pd.DataFrame()
            
        if self.token_throughput_history:
            dfs["token_throughput"] = pd.DataFrame(list(self.token_throughput_history))
        else:
            dfs["token_throughput"] = pd.DataFrame()
            
        if self.latency_history:
            dfs["latency"] = pd.DataFrame(list(self.latency_history))
        else:
            dfs["latency"] = pd.DataFrame()
            
        if self.error_history:
            dfs["errors"] = pd.DataFrame(list(self.error_history))
        else:
            dfs["errors"] = pd.DataFrame()
            
        if self.resource_usage_history:
            dfs["resource_usage"] = pd.DataFrame(list(self.resource_usage_history))
        else:
            dfs["resource_usage"] = pd.DataFrame()
        
        return dfs
    
    def get_models_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of all available models.
        
        Returns:
            List of dictionaries with model information
        """
        return list(self.models_info.values())
    
    async def background_monitoring_task(self, interval: int = 60):
        """
        Background task to continuously collect metrics.
        
        Args:
            interval: Interval in seconds between metric collections
        """
        while True:
            try:
                await self.collect_all_metrics()
                logger.info(f"Collected Ollama metrics successfully")
            except Exception as e:
                logger.error(f"Error in background monitoring: {str(e)}")
            
            await asyncio.sleep(interval)