import os
import aiohttp
import requests
import asyncio
import time
import os
import logging
from typing import Optional, Dict, Any, List, Union
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

DEFAULT_RETRIES = 3
logger_format = "%(levelname)s %(asctime)s %(message)s"
logging.basicConfig(format=logger_format, level=logging.INFO)
logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Base exception for retryable errors"""


class RateLimitExceeded(RetryableError):
    """Rate limit exceeded exception"""


class ServerError(RetryableError):
    """5xx server errors"""


class ClientError(Exception):
    """4xx client errors"""


class GitHubAPIBase:
    """Base class containing shared GitHub API logic"""

    GITHUB_API = "https://api.github.com"
    RAW_CONTENT_URL = "https://raw.githubusercontent.com"
    PER_PAGE = 100  # GitHub API maximum per_page value
    DEFAULT_RETRIES = 3

    def __init__(self):
        self._init_headers()

    def _init_headers(self):
        """Initialize common headers"""
        self.headers = {
            "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github.v3+json",
        }

    def _build_url(self, endpoint: str) -> str:
        """Build full API URL"""
        return f"{self.GITHUB_API}{endpoint}"

    def strip_repo_name(self, repo_name: str) -> str:
        """Normalize repository name format"""
        return repo_name.lstrip("https://github.com/").rstrip("/")

    def _should_retry(self, status_code: int) -> bool:
        """Determine if a request should be retried"""
        if 500 <= status_code < 600:
            return True
        if status_code in [403, 429]:
            return True
        return False

    def _parse_response(self, data: Union[Dict, List]) -> List[Dict]:
        """Parse API response structure"""
        if isinstance(data, list):
            return data
        return data.get("items", [])

    def _handle_api_error(self, status_code: int, message: str):
        """Handle different types of API errors"""
        if self._should_retry(status_code):
            if status_code in [403, 429]:
                raise RateLimitExceeded(message)
            raise ServerError(message)
        if 400 <= status_code < 500:
            raise ClientError(message)
        raise Exception(f"Unknown error: {status_code} {message}")


class GitHubCrawler(GitHubAPIBase):
    """Asynchronous GitHub API client with auto-retry"""

    def __init__(self):
        super().__init__()
        self.session = aiohttp.ClientSession(headers=self.headers)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.session.close()

    async def _check_rate_limit(self, endpoint: str = "/rate_limit") -> Dict:
        """Check current rate limit status"""
        url = self._build_url(endpoint)
        return await self._request("GET", url)

    async def _wait_for_rate_limit_reset(self, rate_limit: Dict) -> None:
        """Handle rate limiting by waiting until reset time"""
        remaining = rate_limit["rate"]["remaining"]
        reset_time = rate_limit["rate"]["reset"]

        if remaining == 0:
            wait_duration = max(reset_time - time.time(), 0)
            logger.warning(f"Rate limit exceeded. Waiting {wait_duration:.2f} seconds.")
            await asyncio.sleep(wait_duration)

    async def _handle_rate_limit(self, endpoint: str = "/rate_limit") -> None:
        """Unified rate limit handler"""
        rate_limit = await self._check_rate_limit(endpoint)
        await self._wait_for_rate_limit_reset(rate_limit)

    @retry(
        retry=retry_if_exception_type(RetryableError),
        stop=stop_after_attempt(DEFAULT_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def _request(
        self, method: str, url: str, parse_json: bool = True, **kwargs
    ) -> Any:
        """Core async request method with retry logic"""
        async with self.session.request(method, url, **kwargs) as response:
            if response.status == 200:
                logger.info(f"Request succeeded: {method} {url}")
                return await response.json() if parse_json else await response.text()

            error_msg = f"{method} {url} failed: {response.status}"
            if response.status == 404:
                raise ClientError(f"{error_msg} - Not Found")

            response_data = await response.text()

            try:
                self._handle_api_error(
                    response.status, f"{error_msg} - {response_data}"
                )
            except RateLimitExceeded:
                await self._handle_rate_limit()
                raise RateLimitExceeded(f"{error_msg} - {response_data}")

    async def get_repo(self, repo_name: str) -> Dict:
        """Get repository details"""
        repo_name = self.strip_repo_name(repo_name)
        url = self._build_url(f"/repos/{repo_name}")
        return await self._request("GET", url)

    async def get_readme(self, repo_name: str, branch: str = "main") -> str:
        """Get repository README content"""
        repo_name = self.strip_repo_name(repo_name)
        readme_names = [
            "README.md",
            "README.rst",
            "README.txt",
            "README",
            "README.MD",
            "README.TXT",
            "README.RST",
            "readme.md",
            "readme.txt",
            "readme.rst",
            "readme",
        ]
        for name in readme_names:
            url = f"{self.RAW_CONTENT_URL}/{repo_name}/{branch}/{name}"
            try:
                return await self._request("GET", url, parse_json=False)
            except ClientError as e:
                continue
        logger.warning(f"No README found for repo {repo_name}")
        return ""

    async def search_repos(
        self, query: str, sort: str = "", limit: int = 100
    ) -> List[Dict]:
        """Generic repository search"""
        results = []
        page = 1

        while len(results) < limit:
            url = self._build_url(
                f"/search/repositories?q={query}&per_page={self.PER_PAGE}&page={page}"
            )
            if sort:
                url += f"&sort={sort}"

            try:
                data = await self._request("GET", url)
                batch = [
                    repo for repo in self._parse_response(data) if not repo["fork"]
                ]
                results.extend(batch[: limit - len(results)])

                if len(batch) < self.PER_PAGE:
                    break
                page += 1
            except RateLimitExceeded:
                logger.warning("Rate limit hit during repository search. Handling...")
                await self._handle_rate_limit(endpoint="/rate_limit")

        logger.info(
            f"Search completed: {len(results)} repositories found for query '{query}'."
        )
        return results
