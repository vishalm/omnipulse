"""
Tests for utility modules in OmniPulse.
"""

import unittest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Import utility modules
from src.utils import helpers
from src.utils import visualization
from src.utils import alerts
from src.config import settings
from src.config import defaults
from src.config import themes


class TestHelpers(unittest.TestCase):
    """Test the helper utility functions."""
    
    def test_format_number(self):
        """Test number formatting with appropriate suffixes."""
        # Test various ranges
        self.assertEqual(helpers.format_number(123), "123")
        self.assertEqual(helpers.format_number(1234), "1.2K")
        self.assertEqual(helpers.format_number(1234567), "1.2M")
        self.assertEqual(helpers.format_number(1234567890), "1.2B")
        self.assertEqual(helpers.format_number(1234567890000), "1.2T")
        
        # Test negative numbers
        self.assertEqual(helpers.format_number(-1234), "-1.2K")
        
        # Test zero
        self.assertEqual(helpers.format_number(0), "0")
        
        # Test None
        self.assertEqual(helpers.format_number(None), "N/A")
    
    def test_format_bytes(self):
        """Test byte formatting with appropriate units."""
        # Test various ranges
        self.assertEqual(helpers.format_bytes(123), "123 B")
        self.assertEqual(helpers.format_bytes(1234), "1.21 KB")
        self.assertEqual(helpers.format_bytes(1234567), "1.18 MB")
        self.assertEqual(helpers.format_bytes(1234567890), "1.15 GB")
        self.assertEqual(helpers.format_bytes(1234567890000), "1.12 TB")
        
        # Test negative numbers
        self.assertEqual(helpers.format_bytes(-1), "0 B")
        
        # Test zero
        self.assertEqual(helpers.format_bytes(0), "0 B")
        
        # Test None
        self.assertEqual(helpers.format_bytes(None), "N/A")
    
    def test_format_time(self):
        """Test time formatting in readable form."""
        # Test various ranges
        self.assertEqual(helpers.format_time(30), "30.0s")
        self.assertEqual(helpers.format_time(90), "1.5m")
        self.assertEqual(helpers.format_time(3600), "1.0h")
        self.assertEqual(helpers.format_time(86400), "1.0d")
        
        # Test negative numbers
        self.assertEqual(helpers.format_time(-1), "0s")
        
        # Test zero
        self.assertEqual(helpers.format_time(0), "0.0s")
        
        # Test None
        self.assertEqual(helpers.format_time(None), "N/A")
    
    def test_format_date(self):
        """Test date formatting."""
        # Test with datetime object
        now = datetime.now()
        self.assertEqual(helpers.format_date(now, "%Y-%m-%d"), now.strftime("%Y-%m-%d"))
        
        # Test with ISO string
        iso_str = "2025-04-01T12:30:45.123456"
        expected = "2025-04-01 12:30:45"
        self.assertEqual(helpers.format_date(iso_str), expected)
        
        # Test with None
        self.assertEqual(helpers.format_date(None), "N/A")
    
    def test_format_percent(self):
        """Test percentage formatting."""
        # Test with values in range 0-1
        self.assertEqual(helpers.format_percent(0.5), "50.0%")
        self.assertEqual(helpers.format_percent(0.123, 2), "12.30%")
        
        # Test with values in range 0-100
        self.assertEqual(helpers.format_percent(50), "50.0%")
        self.assertEqual(helpers.format_percent(12.34, 2), "12.34%")
        
        # Test with None
        self.assertEqual(helpers.format_percent(None), "N/A")
    
    def test_get_color_for_value(self):
        """Test color selection based on thresholds."""
        # Default thresholds and colors
        self.assertEqual(helpers.get_color_for_value(10, [25, 75]), "green")
        self.assertEqual(helpers.get_color_for_value(50, [25, 75]), "yellow")
        self.assertEqual(helpers.get_color_for_value(80, [25, 75]), "red")
        
        # Custom colors
        custom_colors = ["blue", "purple", "orange"]
        self.assertEqual(helpers.get_color_for_value(10, [25, 75], custom_colors), "blue")
        self.assertEqual(helpers.get_color_for_value(50, [25, 75], custom_colors), "purple")
        self.assertEqual(helpers.get_color_for_value(80, [25, 75], custom_colors), "orange")
    
    def test_time_ago(self):
        """Test human-readable time difference."""
        now = datetime.now()
        
        # Test various time ranges
        self.assertTrue("seconds ago" in helpers.time_ago(now - timedelta(seconds=30)))
        self.assertTrue("minute" in helpers.time_ago(now - timedelta(minutes=1)))
        self.assertTrue("minutes" in helpers.time_ago(now - timedelta(minutes=5)))
        self.assertTrue("hour" in helpers.time_ago(now - timedelta(hours=1)))
        self.assertTrue("hours" in helpers.time_ago(now - timedelta(hours=5)))
        self.assertTrue("day" in helpers.time_ago(now - timedelta(days=1)))
        self.assertTrue("days" in helpers.time_ago(now - timedelta(days=5)))
        self.assertTrue("week" in helpers.time_ago(now - timedelta(weeks=1)))
        self.assertTrue("weeks" in helpers.time_ago(now - timedelta(weeks=2)))
        self.assertTrue("month" in helpers.time_ago(now - timedelta(days=35)))
        self.assertTrue("months" in helpers.time_ago(now - timedelta(days=70)))
        self.assertTrue("year" in helpers.time_ago(now - timedelta(days=370)))
        self.assertTrue("years" in helpers.time_ago(now - timedelta(days=800)))
        
        # Test with None
        self.assertEqual(helpers.time_ago(None), "N/A")
    
    def test_get_trend_icon(self):
        """Test trend icon selection."""
        # Normal trend (up is good)
        self.assertEqual(helpers.get_trend_icon(10, 5), "🔺")  # Increase
        self.assertEqual(helpers.get_trend_icon(5, 10), "🔻")  # Decrease
        self.assertEqual(helpers.get_trend_icon(10, 10), "➖")  # No change
        
        # Inverted trend (down is good)
        self.assertEqual(helpers.get_trend_icon(10, 5, inverted=True), "🔻")  # Increase is bad
        self.assertEqual(helpers.get_trend_icon(5, 10, inverted=True), "🔺")  # Decrease is good
        
        # Test with None
        self.assertEqual(helpers.get_trend_icon(None, 5), "➖")
        self.assertEqual(helpers.get_trend_icon(5, None), "➖")
    
    def test_truncate_string(self):
        """Test string truncation."""
        # No truncation needed
        self.assertEqual(helpers.truncate_string("Short text", 20), "Short text")
        
        # Truncation needed
        self.assertEqual(helpers.truncate_string("This is a long text that should be truncated", 20), "This is a long tex...")
        
        # Test with None
        self.assertEqual(helpers.truncate_string(None), "")
    
    def test_clamp(self):
        """Test value clamping."""
        self.assertEqual(helpers.clamp(5, 0, 10), 5)  # Within range
        self.assertEqual(helpers.clamp(-5, 0, 10), 0)  # Below min
        self.assertEqual(helpers.clamp(15, 0, 10), 10)  # Above max
    
    def test_moving_average(self):
        """Test moving average calculation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Test with window size 3
        result = helpers.moving_average(data, 3)
        self.assertEqual(len(result), len(data))
        self.assertEqual(result[0], 1)  # Only one value for first point
        self.assertEqual(result[1], 1.5)  # Average of first two points
        self.assertEqual(result[2], 2)  # Average of first three points
        self.assertEqual(result[-1], 9)  # Average of last three points
        
        # Test with empty list
        self.assertEqual(helpers.moving_average([]), [])
        
        # Test with window larger than data
        self.assertEqual(helpers.moving_average([1, 2], 5), [1, 2])
    
    def test_calculate_change(self):
        """Test change calculation."""
        # Percent change
        self.assertEqual(helpers.calculate_change(110, 100), 10.0)  # 10% increase
        self.assertEqual(helpers.calculate_change(90, 100), -10.0)  # 10% decrease
        
        # Absolute change
        self.assertEqual(helpers.calculate_change(110, 100, as_percent=False), 10)
        self.assertEqual(helpers.calculate_change(90, 100, as_percent=False), -10)
        
        # Edge cases
        self.assertEqual(helpers.calculate_change(100, 0), 0)  # Avoid division by zero
        self.assertEqual(helpers.calculate_change(None, 100), 0)  # None values
        self.assertEqual(helpers.calculate_change(100, None), 0)
    
    def test_safe_divide(self):
        """Test safe division."""
        self.assertEqual(helpers.safe_divide(10, 2), 5)  # Normal division
        self.assertEqual(helpers.safe_divide(10, 0), 0)  # Division by zero returns default
        self.assertEqual(helpers.safe_divide(10, 0, default=None), None)  # Custom default


class TestSettings(unittest.TestCase):
    """Test the settings module."""
    
    @patch("os.environ", {})
    @patch("streamlit.session_state", {})
    @patch("json.load")
    @patch("builtins.open", MagicMock())
    def test_load_settings_default(self, mock_json_load):
        """Test loading default settings."""
        # Set up mock
        mock_json_load.return_value = {}
        
        # Load settings
        result = settings.load_settings()
        
        # Verify defaults were used
        self.assertEqual(result["ollama_api_url"], settings.DEFAULT_SETTINGS["ollama_api_url"])
    
    @patch("os.environ", {"OMNIPULSE_OLLAMA_API_URL": "http://custom:11434"})
    @patch("streamlit.session_state", {})
    @patch("json.load")
    @patch("builtins.open", MagicMock())
    def test_load_settings_env(self, mock_json_load):
        """Test loading settings from environment variables."""
        # Set up mock
        mock_json_load.return_value = {}
        
        # Load settings
        result = settings.load_settings()
        
        # Verify environment variable was used
        self.assertEqual(result["ollama_api_url"], "http://custom:11434")
    
    @patch("os.environ", {})
    @patch("streamlit.session_state", {"ollama_api_url": "http://session:11434"})
    @patch("json.load")
    @patch("builtins.open", MagicMock())
    def test_load_settings_session(self, mock_json_load):
        """Test loading settings from session state."""
        # Set up mock
        mock_json_load.return_value = {}
        
        # Load settings
        result = settings.load_settings()
        
        # Verify session state was used
        self.assertEqual(result["ollama_api_url"], "http://session:11434")
    
    @patch("os.environ", {})
    @patch("streamlit.session_state", {})
    def test_get_setting(self):
        """Test getting a specific setting."""
        # Mock load_settings to return a fixed dictionary
        with patch.object(settings, 'load_settings', return_value={"test_key": "test_value"}):
            # Get setting
            result = settings.get_setting("test_key", "default")
            
            # Verify result
            self.assertEqual(result, "test_value")
            
            # Get non-existent setting
            result = settings.get_setting("non_existent", "default")
            
            # Verify default was returned
            self.assertEqual(result, "default")


class TestThemes(unittest.TestCase):
    """Test the themes module."""
    
    @patch("streamlit.markdown")
    @patch("streamlit.session_state", {})
    def test_apply_theme(self, mock_markdown):
        """Test applying a theme."""
        # Apply a predefined theme
        themes.apply_theme("dark")
        
        # Verify theme was stored in session state
        self.assertEqual(st.session_state.current_theme_name, "dark")
        self.assertEqual(st.session_state.current_theme, themes.THEMES["dark"])
        
        # Verify markdown was called with CSS
        mock_markdown.assert_called_once()
        css_arg = mock_markdown.call_args[0][0]
        self.assertIn("<style>", css_arg)
        
        # Apply a custom theme
        custom_theme = {"background_color": "#custom"}
        themes.apply_theme("light", custom_theme)
        
        # Verify theme was updated with custom values
        self.assertEqual(st.session_state.current_theme["background_color"], "#custom")
    
    def test_get_theme_options(self):
        """Test getting theme options."""
        options = themes.get_theme_options()
        
        # Verify options
        self.assertIn("light", options)
        self.assertIn("dark", options)
        self.assertEqual(options["light"], "Light Mode")
        self.assertEqual(options["dark"], "Dark Mode")
    
    @patch("streamlit.session_state", {"current_theme": {"background_color": "#test"}})
    def test_get_current_theme(self):
        """Test getting current theme."""
        theme = themes.get_current_theme()
        
        # Verify theme
        self.assertEqual(theme["background_color"], "#test")
    
    @patch("streamlit.session_state", {})
    def test_get_current_theme_default(self):
        """Test getting current theme with no theme set."""
        theme = themes.get_current_theme()
        
        # Verify default theme was returned
        self.assertEqual(theme, themes.THEMES["light"])
    
    def test_add_custom_colors(self):
        """Test adding RGB variants of colors to a theme."""
        theme = {"accent_color": "#FF5500"}
        
        # Add RGB variants
        result = themes.add_custom_colors(theme)
        
        # Verify RGB variants were added
        self.assertIn("accent_color_rgb", result)
        self.assertEqual(result["accent_color_rgb"], "255, 85, 0")
        
        # Verify hover color was added
        self.assertIn("accent_hover_color", result)


class TestAlerts(unittest.TestCase):
    """Test the alerts utility functions."""
    
    @patch("logging.Logger.warning")
    def test_check_threshold(self, mock_warning):
        """Test threshold checking."""
        # Define a test alert config
        alert_config = {
            "name": "CPU Usage Alert",
            "metric": "cpu_percent",
            "threshold": 90,
            "comparison": "greater_than",
            "severity": "warning",
            "enabled": True
        }
        
        # Test value below threshold
        result = alerts.check_threshold(80, alert_config)
        self.assertFalse(result)  # No alert
        mock_warning.assert_not_called()
        
        # Test value above threshold
        result = alerts.check_threshold(95, alert_config)
        self.assertTrue(result)  # Alert triggered
        mock_warning.assert_called_once()
        
        # Reset mock
        mock_warning.reset_mock()
        
        # Test with different comparison
        alert_config["comparison"] = "less_than"
        alert_config["threshold"] = 10
        
        # Test value above threshold
        result = alerts.check_threshold(20, alert_config)
        self.assertFalse(result)  # No alert
        mock_warning.assert_not_called()
        
        # Test value below threshold
        result = alerts.check_threshold(5, alert_config)
        self.assertTrue(result)  # Alert triggered
        mock_warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()