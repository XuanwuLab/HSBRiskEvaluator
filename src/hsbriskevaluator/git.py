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

    async def get_contributors(self, repo_name: str, limit: int = 100) -> List[Dict]:
        """Get repository contributors with detailed user information"""
        repo_name = self.strip_repo_name(repo_name)
        contributors = []
        page = 1

        while len(contributors) < limit:
            url = self._build_url(
                f"/repos/{repo_name}/contributors?per_page={self.PER_PAGE}&page={page}"
            )
            
            try:
                data = await self._request("GET", url)
                if not data:
                    break
                
                # Get detailed user info for each contributor
                for contributor in data:
                    if len(contributors) >= limit:
                        break
                    
                    user_detail = await self.get_user(contributor["login"])
                    contributors.append({
                        "username": contributor["login"],
                        "name": user_detail.get("name", ""),
                        "email": user_detail.get("email", ""),
                        "contributions": contributor.get("contributions", 0)
                    })
                
                if len(data) < self.PER_PAGE:
                    break
                page += 1
                
            except RateLimitExceeded:
                logger.warning("Rate limit hit during contributors fetch. Handling...")
                await self._handle_rate_limit()
            except ClientError as e:
                logger.error(f"Error fetching contributors: {e}")
                break

        logger.info(f"Retrieved {len(contributors)} contributors for {repo_name}")
        return contributors

    async def get_user(self, username: str) -> Dict:
        """Get detailed user information"""
        url = self._build_url(f"/users/{username}")
        try:
            return await self._request("GET", url)
        except ClientError:
            logger.warning(f"Could not fetch user details for {username}")
            return {"login": username, "name": "", "email": ""}

    async def get_issues(self, repo_name: str, state: str = "all", limit: int = 100) -> List[Dict]:
        """Get repository issues with comments"""
        repo_name = self.strip_repo_name(repo_name)
        issues = []
        page = 1

        while len(issues) < limit:
            url = self._build_url(
                f"/repos/{repo_name}/issues?state={state}&per_page={self.PER_PAGE}&page={page}"
            )
            
            try:
                data = await self._request("GET", url)
                if not data:
                    break
                
                for issue in data:
                    if len(issues) >= limit:
                        break
                    
                    # Skip pull requests (they appear in issues endpoint)
                    if "pull_request" in issue:
                        continue
                    
                    # Get comments for this issue
                    comments = await self.get_issue_comments(repo_name, issue["number"])
                    
                    issue_data = {
                        "number": issue["number"],
                        "author": issue["user"]["login"],
                        "title": issue["title"],
                        "body": issue.get("body", ""),
                        "comments": comments,
                        "status": issue["state"],
                        "url": issue["html_url"],
                        "created_at": issue["created_at"],
                        "updated_at": issue["updated_at"]
                    }
                    issues.append(issue_data)
                
                if len(data) < self.PER_PAGE:
                    break
                page += 1
                
            except RateLimitExceeded:
                logger.warning("Rate limit hit during issues fetch. Handling...")
                await self._handle_rate_limit()
            except ClientError as e:
                logger.error(f"Error fetching issues: {e}")
                break

        logger.info(f"Retrieved {len(issues)} issues for {repo_name}")
        return issues

    async def get_issue_comments(self, repo_name: str, issue_number: int) -> List[Dict]:
        """Get comments for a specific issue"""
        repo_name = self.strip_repo_name(repo_name)
        url = self._build_url(f"/repos/{repo_name}/issues/{issue_number}/comments")
        
        try:
            data = await self._request("GET", url)
            comments = []
            for comment in data:
                comments.append({
                    "username": comment["user"]["login"],
                    "content": comment["body"],
                    "created_at": comment["created_at"]
                })
            return comments
        except ClientError:
            logger.warning(f"Could not fetch comments for issue #{issue_number}")
            return []

    async def get_pull_requests(self, repo_name: str, state: str = "all", limit: int = 100) -> List[Dict]:
        """Get repository pull requests with review information"""
        repo_name = self.strip_repo_name(repo_name)
        pull_requests = []
        page = 1

        while len(pull_requests) < limit:
            url = self._build_url(
                f"/repos/{repo_name}/pulls?state={state}&per_page={self.PER_PAGE}&page={page}"
            )
            
            try:
                data = await self._request("GET", url)
                if not data:
                    break
                
                for pr in data:
                    if len(pull_requests) >= limit:
                        break
                    
                    # Get reviews for this PR
                    reviews = await self.get_pr_reviews(repo_name, pr["number"])
                    
                    # Find approver and merger
                    approver = ""
                    merger = pr.get("merged_by", {}).get("login", "") if pr.get("merged_by") else ""
                    
                    # Find the last approving review
                    for review in reversed(reviews):
                        if review.get("state") == "APPROVED":
                            approver = review.get("user", {}).get("login", "")
                            break
                    
                    pr_data = {
                        "number": pr["number"],
                        "title": pr["title"],
                        "author": pr["user"]["login"],
                        "approver": approver,
                        "merger": merger,
                        "status": "merged" if pr["merged_at"] else pr["state"],
                        "url": pr["html_url"],
                        "created_at": pr["created_at"],
                        "updated_at": pr["updated_at"],
                        "merged_at": pr.get("merged_at")
                    }
                    pull_requests.append(pr_data)
                
                if len(data) < self.PER_PAGE:
                    break
                page += 1
                
            except RateLimitExceeded:
                logger.warning("Rate limit hit during pull requests fetch. Handling...")
                await self._handle_rate_limit()
            except ClientError as e:
                logger.error(f"Error fetching pull requests: {e}")
                break

        logger.info(f"Retrieved {len(pull_requests)} pull requests for {repo_name}")
        return pull_requests

    async def get_pr_reviews(self, repo_name: str, pr_number: int) -> List[Dict]:
        """Get reviews for a specific pull request"""
        repo_name = self.strip_repo_name(repo_name)
        url = self._build_url(f"/repos/{repo_name}/pulls/{pr_number}/reviews")
        
        try:
            return await self._request("GET", url)
        except ClientError:
            logger.warning(f"Could not fetch reviews for PR #{pr_number}")
            return []

    async def get_repository_contents(self, repo_name: str, path: str = "", ref: str = "main") -> List[Dict]:
        """Get repository contents at a specific path"""
        repo_name = self.strip_repo_name(repo_name)
        url = self._build_url(f"/repos/{repo_name}/contents/{path}")
        if ref:
            url += f"?ref={ref}"
        
        try:
            return await self._request("GET", url)
        except ClientError:
            logger.warning(f"Could not fetch contents for path: {path}")
            return []

    async def find_binary_files(self, repo_name: str, max_depth: int = 3) -> List[str]:
        """Find binary files in the repository by traversing the file tree"""
        binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.a', '.lib', '.bin', '.dat',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
            '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac',
            '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.jar', '.war', '.ear', '.class', '.pyc', '.pyo',
            '.o', '.obj', '.deb', '.rpm', '.dmg', '.pkg', '.msi'
        }
        
        binary_files = []
        
        async def traverse_directory(path: str = "", current_depth: int = 0):
            if current_depth > max_depth:
                return
            
            contents = await self.get_repository_contents(repo_name, path)
            
            for item in contents:
                if item["type"] == "file":
                    file_path = item["path"]
                    file_ext = os.path.splitext(file_path)[1].lower()
                    
                    if file_ext in binary_extensions:
                        binary_files.append(file_path)
                elif item["type"] == "dir" and current_depth < max_depth:
                    await traverse_directory(item["path"], current_depth + 1)
        
        try:
            await traverse_directory()
            logger.info(f"Found {len(binary_files)} binary files in {repo_name}")
        except Exception as e:
            logger.error(f"Error finding binary files: {e}")
        
        return binary_files

    async def get_repo_info_complete(self, repo_name: str, pkt_name: str = "") -> Dict:
        """Get complete repository information for RepoInfo model"""
        repo_name = self.strip_repo_name(repo_name)
        
        try:
            # Get basic repo info
            repo_data = await self.get_repo(repo_name)
            
            # Gather all required information concurrently
            contributors_task = asyncio.create_task(self.get_contributors(repo_name))
            issues_task = asyncio.create_task(self.get_issues(repo_name))
            prs_task = asyncio.create_task(self.get_pull_requests(repo_name))
            binary_files_task = asyncio.create_task(self.find_binary_files(repo_name))
            
            # Wait for all tasks to complete
            contributors = await contributors_task
            issues = await issues_task
            pull_requests = await prs_task
            binary_files = await binary_files_task
            
            repo_info = {
                "pkt_type": "debian",
                "pkt_name": pkt_name or repo_name.split("/")[-1],
                "repo_id": repo_name.replace("/", "-"),
                "url": repo_data["html_url"],
                "contributor_list": contributors,
                "pr_list": pull_requests,
                "issue_list": issues,
                "binary_file_list": binary_files
            }
            
            logger.info(f"Successfully collected complete repo info for {repo_name}")
            return repo_info
            
        except Exception as e:
            logger.error(f"Error collecting complete repo info: {e}")
            raise
