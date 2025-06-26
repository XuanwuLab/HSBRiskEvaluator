"""
GitHub repository information collector using the official PyGithub library.
This module provides functionality to collect repository information
and convert it to the Pydantic models defined in repo_info.py.
"""

import asyncio
import logging
import os
import subprocess
from typing import List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from github import Github, GithubException, RateLimitExceededException
from github.Repository import Repository
from github.NamedUser import NamedUser
from github.Issue import Issue as GithubIssue
from github.PullRequest import PullRequest as GithubPR
from github.PullRequestComment import PullRequestComment
from github.IssueComment import IssueComment
from github.PullRequestReview import PullRequestReview
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

from .repo_info import RepoInfo, GithubUser, Issue, PullRequest, Comment
from ..utils.file import get_data_dir, is_binary

load_dotenv()
logger = logging.getLogger(__name__)


class GitHubRepoCollector:
    """Collector for GitHub repository information using the official PyGithub library"""

    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize the GitHub collector
        
        Args:
            github_token: GitHub API token. If not provided, will use GITHUB_TOKEN env var
        """
        self.token = github_token or os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN environment variable or pass token directly.")
        
        self.github = Github(self.token)
        self.executor = ThreadPoolExecutor(max_workers=5)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.executor.shutdown(wait=True)
        self.github.close()

    def _convert_to_github_user(self, user: NamedUser) -> GithubUser:
        """Convert PyGithub NamedUser to GithubUser model"""
        return GithubUser(
            username=user.login,
            name=user.name or "",
            email=user.email or ""
        )

    def _convert_to_comment(self, comment: IssueComment | PullRequestComment) -> Comment:
        """Convert PyGithub IssueComment to Comment model"""
        return Comment(
            username=comment.user.login,
            content=comment.body or ""
        )

    def _convert_to_issue(self, issue: GithubIssue) -> Issue:
        """Convert PyGithub Issue to Issue model"""
        comments = [
            self._convert_to_comment(comment) 
            for comment in issue.get_comments()
        ]
        
        # Ensure status is valid for Issue model
        status = "open" if issue.state == "open" else "closed"
        
        return Issue(
            number=issue.number,
            author=issue.user.login,
            title=issue.title,
            body=issue.body or "",
            comments=comments,
            status=status,
            url=issue.html_url,
            created_at=issue.created_at.isoformat() if issue.created_at else "",
            updated_at=issue.updated_at.isoformat() if issue.updated_at else ""
        )

    def _convert_to_pull_request(self, pr: GithubPR) -> PullRequest:
        """Convert PyGithub PullRequest to PullRequest model"""
        # Get the last approving review
        approvers:list[str] = []
        reviewers:list[str] = []

        try:
            for review in reversed(list(pr.get_reviews())):
                reviewers.append(review.user.login)
                if review.state == "APPROVED":
                    approvers.append(review.user.login)
                    break
        except GithubException:
            logger.warning(f"Could not fetch reviews for PR #{pr.number}")

        # Get merger information
        merger = ""
        if pr.merged and pr.merged_by:
            merger = pr.merged_by.login

        # Determine status - ensure it matches the Literal type
        if pr.merged:
            status = "merged"
        elif pr.state == "open":
            status = "open"
        else:
            status = "closed"

        comments = [
            self._convert_to_comment(comment) 
            for comment in pr.get_comments()
        ]
        # Handle timestamps safely
        created_at_str = pr.created_at.isoformat() if pr.created_at else ""
        updated_at_str = pr.updated_at.isoformat() if pr.updated_at else ""
        merged_at_str = pr.merged_at.isoformat() if pr.merged_at else None

        return PullRequest(
            number=pr.number,
            title=pr.title,
            author=pr.user.login,
            reviewers=reviewers,
            body = pr.body or "",
            comments=comments,
            approvers=approvers,
            merger=merger,
            status=status,
            url=pr.html_url,
            created_at=created_at_str,
            updated_at=updated_at_str,
            merged_at=merged_at_str
        )

    @retry(
        retry=retry_if_exception_type(RateLimitExceededException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=60, max=300)
    )
    def _get_repository(self, repo_name: str) -> Repository:
        """Get repository with retry logic for rate limiting"""
        return self.github.get_repo(repo_name)

    async def _run_in_executor(self, func, *args):
        """Run blocking GitHub API calls in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def _clone_repository(self, repo_name: str, repo_url: str) -> Optional[str]:
        """
        Clone repository to local data directory
        
        Args:
            repo_name: Repository name in format 'owner/repo'
            repo_url: Repository clone URL
            
        Returns:
            str: Relative path to cloned repository from data_dir, or None if failed
        """
        def _clone_repo():
            try:
                data_dir = get_data_dir()
                repo_dir_name = repo_name.replace("/", "-")
                local_repo_path = data_dir / repo_dir_name
                
                # Remove existing directory if it exists
                if local_repo_path.exists():
                    import shutil
                    shutil.rmtree(local_repo_path)
                
                # Clone the repository
                cmd = ["git", "clone", repo_url, str(local_repo_path)]
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    logger.info(f"Successfully cloned {repo_name} to {local_repo_path}")
                    return repo_dir_name  # Return relative path
                else:
                    logger.error(f"Failed to clone {repo_name}: {result.stderr}")
                    return None
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout cloning repository {repo_name}")
                return None
            except Exception as e:
                logger.error(f"Error cloning repository {repo_name}: {str(e)}")
                return None
        
        return await self._run_in_executor(_clone_repo)

    async def _find_binary_files_local(self, local_repo_path: Path) -> List[str]:
        """
        Find binary files in the locally cloned repository using the is_binary function
        This method scans all files and uses content-based detection rather than extension-based
        
        Args:
            local_repo_path: Path to the local repository
            
        Returns:
            List[str]: List of binary file paths relative to repository root
        """
        def _scan_local_files():
            binary_files = []
            try:
                # Scan all files in the repository
                for file_path in local_repo_path.rglob("*"):
                    if file_path.is_file():
                        # Skip .git directory files
                        if '.git' in file_path.parts:
                            continue
                            
                        # Check if file is binary using the is_binary function
                        # This function includes is_dir check and content-based binary detection
                        if is_binary(str(file_path)):
                            # Get relative path from repository root
                            relative_path = file_path.relative_to(local_repo_path)
                            binary_files.append(str(relative_path))
                
                logger.info(f"Found {len(binary_files)} binary files in local repository using content-based detection")
                return binary_files
            except Exception as e:
                logger.error(f"Error scanning local repository for binary files: {str(e)}")
                return []
        
        return await self._run_in_executor(_scan_local_files)

    async def _get_contributors(self, repo: Repository, limit: int = 100) -> List[GithubUser]:
        """Get repository contributors with detailed user information"""
        def _fetch_contributors():
            contributors = []
            try:
                for contributor in repo.get_contributors():
                    if len(contributors) >= limit:
                        break
                    contributors.append(self._convert_to_github_user(contributor))
                logger.info(f"Retrieved {len(contributors)} contributors for {repo.full_name}")
                return contributors
            except GithubException as e:
                logger.error(f"Error fetching contributors: {e}")
                return []

        return await self._run_in_executor(_fetch_contributors)

    async def _get_issues(self, repo: Repository, limit: int = 100) -> List[Issue]:
        """Get repository issues (excluding pull requests)"""
        def _fetch_issues():
            issues = []
            try:
                # Get issues excluding pull requests
                for issue in repo.get_issues(state="all"):
                    if len(issues) >= limit:
                        break
                    # Skip pull requests (they appear in issues endpoint)
                    if issue.pull_request is None:
                        issues.append(self._convert_to_issue(issue))
                
                logger.info(f"Retrieved {len(issues)} issues for {repo.full_name}")
                return issues
            except GithubException as e:
                logger.error(f"Error fetching issues: {e}")
                return []

        return await self._run_in_executor(_fetch_issues)

    async def _get_pull_requests(self, repo: Repository, limit: int = 100) -> List[PullRequest]:
        """Get repository pull requests using search API as suggested"""
        def _fetch_pull_requests():
            pull_requests = []
            try:
                # Use search API to get pull requests as suggested in the task
                query = f"repo:{repo.full_name} is:pr"
                search_results = self.github.search_issues(query)
                
                # Convert search results to actual PR objects and process
                for issue in search_results[:limit]:
                    if issue.pull_request:
                        # Get the actual PR object
                        pr = repo.get_pull(issue.number)
                        pull_requests.append(self._convert_to_pull_request(pr))
                
                logger.info(f"Retrieved {len(pull_requests)} pull requests for {repo.full_name}")
                return pull_requests
            except GithubException as e:
                logger.error(f"Error fetching pull requests: {e}")
                return []

        return await self._run_in_executor(_fetch_pull_requests)

    async def _find_binary_files(self, repo: Repository, max_depth: int = 3) -> List[str]:
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
        
        def _traverse_directory(path: str = "", current_depth: int = 0) -> List[str]:
            binary_files = []
            
            if current_depth > max_depth:
                return binary_files
            
            try:
                contents = repo.get_contents(path)
                if not isinstance(contents, list):
                    contents = [contents]
                
                for item in contents:
                    if item.type == "file":
                        file_path = item.path
                        file_ext = os.path.splitext(file_path)[1].lower()
                        
                        if file_ext in binary_extensions:
                            binary_files.append(file_path)
                    elif item.type == "dir" and current_depth < max_depth:
                        binary_files.extend(_traverse_directory(item.path, current_depth + 1))
                        
            except GithubException as e:
                logger.warning(f"Error accessing path {path}: {e}")
            
            return binary_files

        def _fetch_binary_files():
            try:
                binary_files = _traverse_directory()
                logger.info(f"Found {len(binary_files)} binary files in {repo.full_name}")
                return binary_files
            except Exception as e:
                logger.error(f"Error finding binary files: {e}")
                return []

        return await self._run_in_executor(_fetch_binary_files)

    async def collect_repo_info(
        self, 
        repo_name: str, 
        pkt_name: str = "",
        max_contributors: int = 100,
        max_issues: int = 100,
        max_prs: int = 100,
        clone_repo: bool = True
    ) -> RepoInfo:
        """
        Collect complete repository information and return as RepoInfo model
        
        Args:
            repo_name: Repository name in format 'owner/repo'
            pkt_name: Package name (optional, defaults to repo name)
            max_contributors: Maximum number of contributors to fetch
            max_issues: Maximum number of issues to fetch
            max_prs: Maximum number of pull requests to fetch
            clone_repo: Whether to clone the repository locally for better binary file detection
            
        Returns:
            RepoInfo: Complete repository information
        """
        logger.info(f"Starting collection for repository: {repo_name}")

        try:
            # Get repository object
            repo = await self._run_in_executor(self._get_repository, repo_name)
            
            # Clone repository if requested
            local_repo_dir = None
            if clone_repo:
                local_repo_dir = await self._clone_repository(repo_name, repo.clone_url)
            
            # Gather all required information concurrently
            contributors_task = asyncio.create_task(
                self._get_contributors(repo, max_contributors)
            )
            issues_task = asyncio.create_task(
                self._get_issues(repo, max_issues)
            )
            prs_task = asyncio.create_task(
                self._get_pull_requests(repo, max_prs)
            )
            
            # Use local binary file detection if repository was cloned, otherwise use API
            if local_repo_dir:
                data_dir = get_data_dir()
                local_repo_path = data_dir / local_repo_dir
                binary_files_task = asyncio.create_task(
                    self._find_binary_files_local(local_repo_path)
                )
            else:
                binary_files_task = asyncio.create_task(
                    self._find_binary_files(repo)
                )
            
            # Wait for all tasks to complete
            contributors, issues, pull_requests, binary_files = await asyncio.gather(
                contributors_task,
                issues_task,
                prs_task,
                binary_files_task
            )

            # Create RepoInfo model
            repo_info = RepoInfo(
                pkt_type="debian",
                pkt_name=pkt_name or repo_name.split("/")[-1],
                repo_id=repo_name.replace("/", "-"),
                url=repo.html_url,
                contributor_list=contributors,
                pr_list=pull_requests,
                issue_list=issues,
                binary_file_list=binary_files,
                local_repo_dir=local_repo_dir
            )

            logger.info(f"Successfully collected repo info for {repo_name}")
            logger.info(f"  - Contributors: {len(contributors)}")
            logger.info(f"  - Issues: {len(issues)}")
            logger.info(f"  - Pull Requests: {len(pull_requests)}")
            logger.info(f"  - Binary Files: {len(binary_files)}")
            if local_repo_dir:
                logger.info(f"  - Local Repository: {local_repo_dir}")

            return repo_info

        except GithubException as e:
            logger.error(f"GitHub API error collecting repo info for {repo_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error collecting repo info for {repo_name}: {e}")
            raise

    async def collect_multiple_repos(
        self, 
        repo_names: List[str],
        max_concurrent: int = 3,
        **kwargs
    ) -> List[RepoInfo]:
        """
        Collect information for multiple repositories concurrently
        
        Args:
            repo_names: List of repository names
            max_concurrent: Maximum number of concurrent requests
            **kwargs: Additional arguments passed to collect_repo_info
            
        Returns:
            List[RepoInfo]: List of repository information
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def collect_single_repo(repo_name: str) -> Optional[RepoInfo]:
            async with semaphore:
                try:
                    return await self.collect_repo_info(repo_name, **kwargs)
                except Exception as e:
                    logger.error(f"Failed to collect info for {repo_name}: {e}")
                    return None

        logger.info(f"Starting collection for {len(repo_names)} repositories")
        
        tasks = [collect_single_repo(repo_name) for repo_name in repo_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        repo_infos = [
            result for result in results 
            if isinstance(result, RepoInfo)
        ]
        
        logger.info(f"Successfully collected info for {len(repo_infos)}/{len(repo_names)} repositories")
        return repo_infos


# Convenience function for single repository collection
async def collect_github_repo_info(
    repo_name: str, 
    pkt_name: str = "",
    github_token: Optional[str] = None,
    **kwargs
) -> RepoInfo:
    """
    Convenience function to collect information for a single repository
    
    Args:
        repo_name: Repository name in format 'owner/repo'
        pkt_name: Package name (optional)
        github_token: GitHub API token (optional, uses env var if not provided)
        **kwargs: Additional arguments
        
    Returns:
        RepoInfo: Repository information
    """
    async with GitHubRepoCollector(github_token) as collector:
        return await collector.collect_repo_info(repo_name, pkt_name, **kwargs)


# Convenience function for multiple repositories collection
async def collect_multiple_github_repos(
    repo_names: List[str],
    github_token: Optional[str] = None,
    **kwargs
) -> List[RepoInfo]:
    """
    Convenience function to collect information for multiple repositories
    
    Args:
        repo_names: List of repository names
        github_token: GitHub API token (optional, uses env var if not provided)
        **kwargs: Additional arguments
        
    Returns:
        List[RepoInfo]: List of repository information
    """
    async with GitHubRepoCollector(github_token) as collector:
        return await collector.collect_multiple_repos(repo_names, **kwargs)
