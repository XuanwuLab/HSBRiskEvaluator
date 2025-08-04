"""
GitHub token management for load balancing and rate limit avoidance.
"""

import logging
import os
import threading
import time
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from github.Rate import Rate
from github.RateLimit import RateLimit

if TYPE_CHECKING:
    from ..settings import CollectorSettings
import random
from itertools import cycle
from datetime import datetime, timedelta

from github import Github

logger = logging.getLogger(__name__)


class GitHubTokenManager:
    """Manages multiple GitHub tokens for load balancing and rate limit handling"""

    def __init__(self, settings: "CollectorSettings"):
        """
        Initialize token manager from settings

        Args:
            settings: CollectorSettings containing github_tokens and github_proxy_url

        Raises:
            ValueError: If no tokens are found
        """
        tokens = []

        # Load tokens from settings and environment
        if settings.github_tokens:
            tokens += settings.github_tokens
        env_token = os.getenv("GITHUB_TOKEN")
        if env_token:
            tokens.append(env_token)

        # Remove duplicates
        tokens = list(set(tokens))
        logger.info(f"Using {len(tokens)} GitHub tokens for API requests")

        if not tokens:
            raise ValueError(
                "GitHub token(s) required. Set GITHUB_TOKEN environment variable, "
                "or configure github_tokens in settings."
            )

        random.shuffle(tokens)
        self.proxy_url = settings.github_proxy_url
        self._token_cycle = cycle(tokens)
        self._current_token = next(self._token_cycle)
        self._github_clients: Dict[str, Github] = {}
        self._lock = threading.Lock()

        # Initialize GitHub clients for each token
        for token in tokens:
            if self.proxy_url:
                raise NotImplementedError(
                    "Proxy support is not implemented yet")
            else:
                self._github_clients[token] = Github(
                    login_or_token=token, per_page=100)

    def get_github_client(self) -> Github:
        """
        Get a GitHub client with the current token

        Returns:
            Github: PyGithub client instance
        """
        with self._lock:
            return self._github_clients[self._current_token]

    def rotate_token(self) -> None:
        """Manually rotate to the next token"""
        with self._lock:
            self._current_token = next(self._token_cycle)
            logger.debug(f"Rotated to token {self._current_token[:8]}...")

    def get_token_stats(self) -> Optional[RateLimit]:
        """Get real rate limit statistics for all tokens from GitHub API"""
        with self._lock:
            try:
                github_client = self._github_clients[self._current_token]
                rate_limit: RateLimit = github_client.get_rate_limit()
                # Get core rate limit info (the main API rate limit)
            except Exception as e:
                logger.warning(
                    f"Failed to get rate limit for token {self._current_token[:8]}...: {e}"
                )
                return None

            return rate_limit

    def close_all_clients(self) -> None:
        """Close all GitHub clients"""
        with self._lock:
            for token, github_client in self._github_clients.items():
                try:
                    github_client.close()
                except Exception as e:
                    logger.debug(
                        f"Error closing GitHub client for token {token[:8]}...: {e}"
                    )
            self._github_clients.clear()
