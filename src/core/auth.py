"""
Authentication module for OmniPulse dashboard.
"""

import streamlit as st
import hashlib
import hmac
import base64
import time
import os
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta

from src.config.settings import get_setting


class Auth:
    """Authentication manager for OmniPulse dashboard."""
    
    def __init__(self, cookie_name: str = "omnipulse_auth", cookie_key: Optional[str] = None):
        """
        Initialize the Auth manager.
        
        Args:
            cookie_name: Name of the authentication cookie
            cookie_key: Secret key for cookie encryption (generated if not provided)
        """
        self.cookie_name = cookie_name
        self.cookie_key = cookie_key or os.environ.get("OMNIPULSE_AUTH_KEY", str(uuid.uuid4()))
        self.cookie_expiry_days = get_setting("auth_cookie_expiry_days", 30)
    
    def generate_password_hash(self, password: str) -> str:
        """
        Generate a secure hash for a password.
        
        Args:
            password: Plain text password
            
        Returns:
            Password hash
        """
        # Use SHA-256 for hashing
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password to verify
            password_hash: Password hash to check against
            
        Returns:
            True if password is correct, False otherwise
        """
        # Generate hash of the provided password and compare
        generated_hash = self.generate_password_hash(password)
        return hmac.compare_digest(generated_hash, password_hash)
    
    def _generate_cookie_signature(self, payload: str) -> str:
        """
        Generate signature for cookie payload.
        
        Args:
            payload: Cookie payload to sign
            
        Returns:
            Signature string
        """
        return hmac.new(
            self.cookie_key.encode(), 
            msg=payload.encode(),
            digestmod=hashlib.sha256
        ).hexdigest()
    
    def _encode_cookie(self, username: str, expiry_timestamp: float) -> str:
        """
        Encode cookie with authentication information.
        
        Args:
            username: Username
            expiry_timestamp: Cookie expiry timestamp
            
        Returns:
            Encoded cookie value
        """
        # Create payload
        payload = f"{username}:{expiry_timestamp}"
        
        # Sign payload
        signature = self._generate_cookie_signature(payload)
        
        # Combine payload and signature
        cookie_value = f"{payload}:{signature}"
        
        # Encode as base64
        return base64.urlsafe_b64encode(cookie_value.encode()).decode()
    
    def _decode_cookie(self, cookie_value: str) -> Optional[Tuple[str, float]]:
        """
        Decode and verify a cookie.
        
        Args:
            cookie_value: Encoded cookie value
            
        Returns:
            Tuple of (username, expiry_timestamp) if valid, None otherwise
        """
        try:
            # Decode from base64
            decoded = base64.urlsafe_b64decode(cookie_value.encode()).decode()
            
            # Split into parts
            payload, signature = decoded.rsplit(":", 1)
            username, expiry_timestamp = payload.split(":")
            expiry_timestamp = float(expiry_timestamp)
            
            # Verify signature
            expected_signature = self._generate_cookie_signature(payload)
            if not hmac.compare_digest(signature, expected_signature):
                return None
            
            # Check expiration
            if time.time() > expiry_timestamp:
                return None
            
            return (username, expiry_timestamp)
        
        except Exception:
            return None
    
    def login(self, username: str, password: str) -> bool:
        """
        Authenticate a user and set login cookie if successful.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            True if login successful, False otherwise
        """
        # Check if authentication is enabled
        if not get_setting("enable_authentication", False):
            # Auto-login if auth is disabled
            return self._set_auth_cookie(username)
        
        # Get stored password hash
        stored_username = get_setting("auth_username", "admin")
        stored_hash = get_setting("auth_password_hash", None)
        
        # If no password hash is set, use default (admin/admin)
        if not stored_hash and username == "admin":
            stored_hash = self.generate_password_hash("admin")
        
        # Verify username and password
        if username == stored_username and stored_hash and self.verify_password(password, stored_hash):
            return self._set_auth_cookie(username)
        
        # Check preauthorized emails if applicable
        preauthorized_emails = get_setting("preauthorized_emails", [])
        if username in preauthorized_emails:
            return self._set_auth_cookie(username)
        
        return False
    
    def _set_auth_cookie(self, username: str) -> bool:
        """
        Set authentication cookie for a user.
        
        Args:
            username: Username to set cookie for
            
        Returns:
            True if successful
        """
        # Calculate expiry time
        expiry_timestamp = time.time() + (self.cookie_expiry_days * 24 * 60 * 60)
        
        # Generate cookie
        cookie_value = self._encode_cookie(username, expiry_timestamp)
        
        # Store in Streamlit session
        st.session_state.auth_username = username
        
        # Set cookie
        st.session_state[self.cookie_name] = cookie_value
        
        return True
    
    def logout(self):
        """Log out the current user by clearing authentication."""
        # Clear session state
        if "auth_username" in st.session_state:
            del st.session_state.auth_username
        
        # Clear cookie
        if self.cookie_name in st.session_state:
            del st.session_state[self.cookie_name]
    
    def is_authenticated(self) -> bool:
        """
        Check if the current user is authenticated.
        
        Returns:
            True if authenticated, False otherwise
        """
        # Check if authentication is disabled
        if not get_setting("enable_authentication", False):
            return True
        
        # Check for username in session state (direct login)
        if "auth_username" in st.session_state:
            return True
        
        # Check for cookie in session state
        if self.cookie_name in st.session_state:
            cookie_value = st.session_state[self.cookie_name]
            result = self._decode_cookie(cookie_value)
            
            if result:
                username, _ = result
                st.session_state.auth_username = username
                return True
        
        return False
    
    def get_username(self) -> Optional[str]:
        """
        Get the username of the authenticated user.
        
        Returns:
            Username if authenticated, None otherwise
        """
        if self.is_authenticated():
            return st.session_state.get("auth_username")
        return None
    
    def require_auth(self, on_failure: Optional[Callable] = None):
        """
        Require authentication to access the current page.
        
        Args:
            on_failure: Optional callback to execute on authentication failure
        """
        if not self.is_authenticated():
            if on_failure:
                on_failure()
            else:
                self.show_login_form()
            st.stop()
    
    def show_login_form(self):
        """Display a login form."""
        st.title("OmniPulse Login")
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if self.login(username, password):
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
        
        # Display default credentials hint if using default
        if not get_setting("auth_password_hash"):
            st.info("Default credentials: admin / admin")


# Initialize authentication manager
auth = Auth()


def login_required(func):
    """
    Decorator to require authentication for a function.
    
    Args:
        func: Function to protect
    
    Returns:
        Wrapped function that requires authentication
    """
    def wrapper(*args, **kwargs):
        if not auth.is_authenticated():
            auth.show_login_form()
            st.stop()
        return func(*args, **kwargs)
    return wrapper