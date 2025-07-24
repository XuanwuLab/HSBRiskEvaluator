"""
GitHub token management for load balancing and rate limit avoidance.
"""

import logging
import threading
import time
from typing import List, Optional, Dict, Any
from itertools import cycle
from datetime import datetime, timedelta

from github import Github, RateLimitExceededException

logger = logging.getLogger(__name__)


class GitHubTokenManager:
    """Manages multiple GitHub tokens for load balancing and rate limit handling"""
    
    def __init__(self, tokens: List[str], proxy_url: Optional[str] = None):
        """
        Initialize token manager
        
        Args:
            tokens: List of GitHub tokens
            proxy_url: Optional proxy URL for requests
        """
        if not tokens:
            raise ValueError("At least one GitHub token is required")
            
        self.tokens = tokens
        self.proxy_url = proxy_url
        self._token_cycle = cycle(tokens)
        self._current_token = next(self._token_cycle)
        self._token_usage: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        # Initialize token usage tracking
        for token in tokens:
            self._token_usage[token] = {
                'requests_made': 0,
                'rate_limited_until': None,
                'last_used': None,
                'github_client': None
            }
    
    def get_github_client(self) -> Github:
        """
        Get a GitHub client with an available token
        
        Returns:
            Github: PyGithub client instance
        """
        with self._lock:
            # Try to find an available token (not rate limited)
            available_token = self._find_available_token()
            
            if available_token:
                self._current_token = available_token
            else:
                # All tokens are rate limited, use the least recently rate limited
                self._current_token = self._get_least_rate_limited_token()
                logger.warning(f"All tokens rate limited, using {self._current_token[:8]}...")
            
            # Get or create GitHub client for this token
            token_info = self._token_usage[self._current_token]
            if token_info['github_client'] is None:
                # Create GitHub client with proxy if specified
                if self.proxy_url:
                    raise NotImplementedError("Proxy support is not implemented yet")
                else:
                    token_info['github_client'] = Github(
                        login_or_token=self._current_token,
                        per_page=100
                    )
            
            # Update usage tracking
            token_info['requests_made'] += 1
            token_info['last_used'] = datetime.now()
            
            logger.debug(f"Using token {self._current_token[:8]}... (requests: {token_info['requests_made']})")
            return token_info['github_client']
    
    def handle_rate_limit(self, token: Optional[str] = None) -> None:
        """
        Mark a token as rate limited
        
        Args:
            token: Token to mark as rate limited (defaults to current token)
        """
        if token is None:
            token = self._current_token
            
        with self._lock:
            if token in self._token_usage:
                # GitHub rate limit resets every hour
                reset_time = datetime.now() + timedelta(hours=1)
                self._token_usage[token]['rate_limited_until'] = reset_time
                logger.warning(f"Token {token[:8]}... rate limited until {reset_time}")
    
    def _find_available_token(self) -> Optional[str]:
        """Find a token that is not currently rate limited"""
        current_time = datetime.now()
        
        for token in self.tokens:
            token_info = self._token_usage[token]
            rate_limited_until = token_info['rate_limited_until']
            
            if rate_limited_until is None or current_time > rate_limited_until:
                return token
        
        return None
    
    def _get_least_rate_limited_token(self) -> str:
        """Get the token that will be available soonest"""
        current_time = datetime.now()
        earliest_available = None
        earliest_time = None
        
        for token in self.tokens:
            token_info = self._token_usage[token]
            rate_limited_until = token_info['rate_limited_until']
            
            if rate_limited_until is None:
                return token
            
            if earliest_time is None or rate_limited_until < earliest_time:
                earliest_time = rate_limited_until
                earliest_available = token
        
        return earliest_available or self.tokens[0]
    
    def rotate_token(self) -> None:
        """Manually rotate to the next token"""
        with self._lock:
            self._current_token = next(self._token_cycle)
            logger.debug(f"Rotated to token {self._current_token[:8]}...")
    
    def get_token_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tokens"""
        with self._lock:
            stats = {}
            current_time = datetime.now()
            
            for token in self.tokens:
                token_info = self._token_usage[token]
                is_rate_limited = (
                    token_info['rate_limited_until'] is not None and 
                    current_time < token_info['rate_limited_until']
                )
                
                stats[token[:8] + "..."] = {
                    'requests_made': token_info['requests_made'],
                    'is_rate_limited': is_rate_limited,
                    'rate_limited_until': token_info['rate_limited_until'],
                    'last_used': token_info['last_used']
                }
            
            return stats
    
    def close_all_clients(self) -> None:
        """Close all GitHub clients"""
        with self._lock:
            for token_info in self._token_usage.values():
                if token_info['github_client'] is not None:
                    try:
                        token_info['github_client'].close()
                    except Exception as e:
                        logger.debug(f"Error closing GitHub client: {e}")
                    token_info['github_client'] = None