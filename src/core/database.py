"""
Database module for OmniPulse dashboard.
"""

import os
import json
import sqlite3
import threading
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
from pathlib import Path

from src.config.settings import get_setting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("database")


class Database:
    """Database manager for OmniPulse dashboard."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to SQLite database file (None for in-memory)
        """
        
        self.db_path = db_path or ":memory:"
        self.connection_lock = threading.RLock()
        self.initialized = False
        
        # Initialize database
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection.
        
        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    
    def _init_db(self):
        """Initialize database tables if they don't exist."""
        if self.initialized:
            return
        
        with self.connection_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Create metrics table for storing time-series data
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL,
                    value_text TEXT,
                    tags TEXT
                )
                ''')
                
                # Create events table for storing significant events
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    severity TEXT,
                    data TEXT
                )
                ''')
                
                # Create dashboard_config table for storing dashboard configurations
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS dashboard_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    config TEXT NOT NULL,
                    user TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                ''')
                
                # Create settings table for storing application settings
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                ''')
                
                # Create indexes for faster queries
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_category ON metrics (category)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics (name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp)')
                
                conn.commit()
                self.initialized = True
                logger.info(f"Database initialized at {self.db_path}")
            
            except Exception as e:
                logger.error(f"Error initializing database: {str(e)}")
                raise
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def store_metric(self, category: str, name: str, value: Union[float, str], 
                    timestamp: Optional[datetime] = None, tags: Optional[Dict[str, str]] = None):
        """
        Store a metric value in the database.
        
        Args:
            category: Metric category (e.g., 'system', 'ollama')
            name: Metric name (e.g., 'cpu_percent', 'memory_usage')
            value: Metric value (numeric or string)
            timestamp: Timestamp for the metric (default: current time)
            tags: Optional tags as key-value pairs
        """
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = datetime.now()
        
        # Convert timestamp to ISO format
        timestamp_str = timestamp.isoformat()
        
        # Convert tags to JSON if provided
        tags_json = json.dumps(tags) if tags else None
        
        # Determine value type
        value_numeric = None
        value_text = None
        
        if isinstance(value, (int, float)):
            value_numeric = value
        else:
            value_text = str(value)
        
        with self.connection_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO metrics (timestamp, category, name, value, value_text, tags)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp_str, category, name, value_numeric, value_text, tags_json))
                
                conn.commit()
            
            except Exception as e:
                logger.error(f"Error storing metric: {str(e)}")
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def store_event(self, category: str, name: str, description: Optional[str] = None,
                   severity: str = "info", data: Optional[Dict[str, Any]] = None,
                   timestamp: Optional[datetime] = None):
        """
        Store an event in the database.
        
        Args:
            category: Event category (e.g., 'system', 'ollama')
            name: Event name (e.g., 'startup', 'error')
            description: Optional event description
            severity: Event severity (info, warning, error, critical)
            data: Optional additional data as dictionary
            timestamp: Timestamp for the event (default: current time)
        """
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = datetime.now()
        
        # Convert timestamp to ISO format
        timestamp_str = timestamp.isoformat()
        
        # Convert data to JSON if provided
        data_json = json.dumps(data) if data else None
        
        with self.connection_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO events (timestamp, category, name, description, severity, data)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp_str, category, name, description, severity, data_json))
                
                conn.commit()
            
            except Exception as e:
                logger.error(f"Error storing event: {str(e)}")
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def query_metrics(self, category: Optional[str] = None, name: Optional[str] = None, 
                     start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                     limit: Optional[int] = None) -> pd.DataFrame:
        """
        Query metrics from the database.
        
        Args:
            category: Optional category filter
            name: Optional name filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Optional limit on the number of results
            
        Returns:
            DataFrame with query results
        """
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        # Add filters
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if name:
            query += " AND name = ?"
            params.append(name)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        # Add order and limit
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self.connection_lock:
            try:
                conn = self._get_connection()
                
                # Execute query and convert to DataFrame
                df = pd.read_sql_query(query, conn, params=params)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Parse tags JSON
                if 'tags' in df.columns:
                    df['tags'] = df['tags'].apply(lambda x: json.loads(x) if x else None)
                
                return df
            
            except Exception as e:
                logger.error(f"Error querying metrics: {str(e)}")
                return pd.DataFrame()
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def query_events(self, category: Optional[str] = None, name: Optional[str] = None, 
                    severity: Optional[str] = None, start_time: Optional[datetime] = None, 
                    end_time: Optional[datetime] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Query events from the database.
        
        Args:
            category: Optional category filter
            name: Optional name filter
            severity: Optional severity filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Optional limit on the number of results
            
        Returns:
            DataFrame with query results
        """
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        # Add filters
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if name:
            query += " AND name = ?"
            params.append(name)
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        # Add order and limit
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self.connection_lock:
            try:
                conn = self._get_connection()
                
                # Execute query and convert to DataFrame
                df = pd.read_sql_query(query, conn, params=params)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Parse data JSON
                if 'data' in df.columns:
                    df['data'] = df['data'].apply(lambda x: json.loads(x) if x else None)
                
                return df
            
            except Exception as e:
                logger.error(f"Error querying events: {str(e)}")
                return pd.DataFrame()
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def save_dashboard_config(self, name: str, config: Dict[str, Any], user: Optional[str] = None):
        """
        Save a dashboard configuration.
        
        Args:
            name: Dashboard name
            config: Dashboard configuration dictionary
            user: Optional username of the creator
        """
        current_time = datetime.now().isoformat()
        config_json = json.dumps(config)
        
        with self.connection_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Check if config already exists
                cursor.execute("SELECT id FROM dashboard_config WHERE name = ?", (name,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing config
                    cursor.execute('''
                    UPDATE dashboard_config 
                    SET config = ?, updated_at = ?
                    WHERE name = ?
                    ''', (config_json, current_time, name))
                else:
                    # Insert new config
                    cursor.execute('''
                    INSERT INTO dashboard_config (name, config, user, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (name, config_json, user, current_time, current_time))
                
                conn.commit()
            
            except Exception as e:
                logger.error(f"Error saving dashboard config: {str(e)}")
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def load_dashboard_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load a dashboard configuration.
        
        Args:
            name: Dashboard name
            
        Returns:
            Dashboard configuration dictionary or None if not found
        """
        with self.connection_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("SELECT config FROM dashboard_config WHERE name = ?", (name,))
                result = cursor.fetchone()
                
                if result:
                    return json.loads(result['config'])
                return None
            
            except Exception as e:
                logger.error(f"Error loading dashboard config: {str(e)}")
                return None
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def list_dashboard_configs(self) -> List[Dict[str, Any]]:
        """
        List all saved dashboard configurations.
        
        Returns:
            List of dashboard config metadata
        """
        with self.connection_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, name, user, created_at, updated_at 
                    FROM dashboard_config 
                    ORDER BY updated_at DESC
                """)
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "id": row["id"],
                        "name": row["name"],
                        "user": row["user"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"]
                    })
                
                return results
            
            except Exception as e:
                logger.error(f"Error listing dashboard configs: {str(e)}")
                return []
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def delete_dashboard_config(self, name: str) -> bool:
        """
        Delete a dashboard configuration.
        
        Args:
            name: Dashboard name
            
        Returns:
            True if successful, False otherwise
        """
        with self.connection_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM dashboard_config WHERE name = ?", (name,))
                conn.commit()
                
                return cursor.rowcount > 0
            
            except Exception as e:
                logger.error(f"Error deleting dashboard config: {str(e)}")
                return False
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def set_setting(self, key: str, value: Any):
        """
        Store a setting in the database.
        
        Args:
            key: Setting key
            value: Setting value (will be JSON-encoded)
        """
        current_time = datetime.now().isoformat()
        value_json = json.dumps(value)
        
        with self.connection_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """, (key, value_json, current_time))
                
                conn.commit()
            
            except Exception as e:
                logger.error(f"Error setting value: {str(e)}")
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a setting from the database.
        
        Args:
            key: Setting key
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        with self.connection_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
                result = cursor.fetchone()
                
                if result:
                    return json.loads(result['value'])
                return default
            
            except Exception as e:
                logger.error(f"Error getting setting: {str(e)}")
                return default
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def aggregate_metrics(self, category: str, name: str, 
                         interval: str = 'hour',
                         start_time: Optional[datetime] = None, 
                         end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Aggregate metrics by time interval.
        
        Args:
            category: Metric category
            name: Metric name
            interval: Aggregation interval ('minute', 'hour', 'day', 'week', 'month')
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            DataFrame with aggregated metrics
        """
        # Define time format based on interval
        time_formats = {
            'minute': '%Y-%m-%d %H:%M',
            'hour': '%Y-%m-%d %H',
            'day': '%Y-%m-%d',
            'week': '%Y-%W',
            'month': '%Y-%m'
        }
        
        if interval not in time_formats:
            interval = 'hour'
        
        time_format = time_formats[interval]
        
        # Build query with time aggregation
        query = f"""
            SELECT 
                strftime('{time_format}', timestamp) as time_bucket,
                category,
                name,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                COUNT(*) as count
            FROM metrics
            WHERE category = ? AND name = ?
        """
        
        params = [category, name]
        
        # Add time filters
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        # Group by time bucket and sort
        query += " GROUP BY time_bucket, category, name ORDER BY time_bucket"
        
        with self.connection_lock:
            try:
                conn = self._get_connection()
                
                # Execute query and convert to DataFrame
                df = pd.read_sql_query(query, conn, params=params)
                
                # Convert time_bucket to datetime (for plotting)
                if 'time_bucket' in df.columns:
                    # Format depends on the interval
                    if interval == 'minute':
                        df['timestamp'] = pd.to_datetime(df['time_bucket'], format='%Y-%m-%d %H:%M')
                    elif interval == 'hour':
                        df['timestamp'] = pd.to_datetime(df['time_bucket'], format='%Y-%m-%d %H')
                    elif interval == 'day':
                        df['timestamp'] = pd.to_datetime(df['time_bucket'], format='%Y-%m-%d')
                    elif interval == 'week':
                        # Week format is special - convert year-week to a date
                        def year_week_to_date(year_week):
                            year, week = year_week.split('-')
                            return pd.to_datetime(f"{year} {week} 1", format='%Y %W %w')
                        
                        df['timestamp'] = df['time_bucket'].apply(year_week_to_date)
                    elif interval == 'month':
                        df['timestamp'] = pd.to_datetime(df['time_bucket'], format='%Y-%m')
                
                return df
            
            except Exception as e:
                logger.error(f"Error aggregating metrics: {str(e)}")
                return pd.DataFrame()
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def get_metric_statistics(self, category: str, name: str, 
                            start_time: Optional[datetime] = None, 
                            end_time: Optional[datetime] = None) -> Dict[str, float]:
        """
        Get statistics for a specific metric.
        
        Args:
            category: Metric category
            name: Metric name
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary with metric statistics
        """
        query = """
            SELECT 
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                COUNT(*) as count
            FROM metrics
            WHERE category = ? AND name = ?
        """
        
        params = [category, name]
        
        # Add time filters
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        with self.connection_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute(query, params)
                result = cursor.fetchone()
                
                if result:
                    return {
                        "avg": result["avg_value"],
                        "min": result["min_value"],
                        "max": result["max_value"],
                        "count": result["count"]
                    }
                
                return {
                    "avg": 0,
                    "min": 0,
                    "max": 0,
                    "count": 0
                }
            
            except Exception as e:
                logger.error(f"Error getting metric statistics: {str(e)}")
                return {
                    "avg": 0,
                    "min": 0,
                    "max": 0,
                    "count": 0
                }
            
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def export_metrics_to_csv(self, filename: str, category: Optional[str] = None, 
                            name: Optional[str] = None, start_time: Optional[datetime] = None, 
                            end_time: Optional[datetime] = None) -> bool:
        """
        Export metrics to a CSV file.
        
        Args:
            filename: Output filename
            category: Optional category filter
            name: Optional name filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Query metrics
            df = self.query_metrics(category, name, start_time, end_time)
            
            if df.empty:
                logger.warning("No metrics to export")
                return False
            
            # Export to CSV
            df.to_csv(filename, index=False)
            logger.info(f"Exported {len(df)} metrics to {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting metrics to CSV: {str(e)}")
            return False
    
    def import_metrics_from_csv(self, filename: str) -> int:
        """
        Import metrics from a CSV file.
        
        Args:
            filename: Input CSV filename
            
        Returns:
            Number of metrics imported
        """
        try:
            # Read CSV file
            df = pd.read_csv(filename)
            
            if df.empty:
                logger.warning("No metrics to import from CSV")
                return 0
            
            # Check required columns
            required_columns = ['timestamp', 'category', 'name']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"CSV file must contain columns: {required_columns}")
                return 0
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Import each row as a metric
            imported_count = 0
            for _, row in df.iterrows():
                value = row.get('value')
                value_text = row.get('value_text')
                
                # Parse tags if present
                tags = None
                if 'tags' in row and pd.notna(row['tags']):
                    try:
                        tags = json.loads(row['tags'])
                    except:
                        pass
                
                # Store the metric
                self.store_metric(
                    category=row['category'],
                    name=row['name'],
                    value=value if pd.notna(value) else value_text,
                    timestamp=row['timestamp'],
                    tags=tags
                )
                imported_count += 1
            
            logger.info(f"Imported {imported_count} metrics from {filename}")
            return imported_count
        
        except Exception as e:
            logger.error(f"Error importing metrics from CSV: {str(e)}")
            return 0
    
    def purge_old_data(self, retention_days: int = 30) -> int:
        """
        Purge old data from the database.
        
        Args:
            retention_days: Number of days to keep data
            
        Returns:
            Number of records deleted
        """
        cutoff_date = (datetime.now() - pd.Timedelta(days=retention_days)).isoformat()
        
        with self.connection_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Delete old metrics
                cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_date,))
                metrics_deleted = cursor.rowcount
                
                # Delete old events
                cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_date,))
                events_deleted = cursor.rowcount
                
                conn.commit()
                
                total_deleted = metrics_deleted + events_deleted
                logger.info(f"Purged {metrics_deleted} metrics and {events_deleted} events older than {retention_days} days")
                return total_deleted
            
            except Exception as e:
                logger.error(f"Error purging old data: {str(e)}")
                return 0
            
            finally:
                if 'conn' in locals():
                    conn.close()


# Initialize the database
db_path = get_setting("database_path", "data/omnipulse.db")
db = Database(db_path)