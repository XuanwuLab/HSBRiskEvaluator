"""
GitHub repository information collector using the official PyGithub library.
This module provides functionality to collect repository information
and convert it to the Pydantic models defined in repo_info.py.
"""

import asyncio
import logging
import os
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

from github import Github, GithubException, RateLimitExceededException
from github.Repository import Repository
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from dotenv import load_dotenv

from ..repo_info import RepoInfo
from ..utils.file import get_data_dir
from .data_collectors import GitHubDataCollector
from .utils import LocalRepoUtils

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
        self.token = github_token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError(
                "GitHub token is required. Set GITHUB_TOKEN environment variable or pass token directly."
            )

        self.github = Github(self.token)
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.data_collector = GitHubDataCollector(self.executor)
        self.local_utils = LocalRepoUtils()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.executor.shutdown(wait=True)
        self.github.close()

    @retry(
        retry=retry_if_exception_type(RateLimitExceededException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=60, max=300),
    )
    def _get_repository(self, repo_name: str) -> Repository:
        """Get repository with retry logic for rate limiting"""
        return self.github.get_repo(repo_name)

    async def _run_in_executor(self, func, *args):
        """Run blocking GitHub API calls in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    def search_repo(self, pkg_name: str) -> Repository:
        """
        Search for a repository by name using the GitHub search API
        Args:
            pkg_name: Repository name in format 'owner/repo'
        Returns:
            Repository: PyGithub Repository object if found, otherwise raises GithubException
        """
        try:
            return self.github.search_repositories(pkg_name)[0]
        except IndexError:
            logger.error(f"Error searching for repository {pkg_name}")
            raise

    async def collect_repo_info(
        self,
        repo_name: str,
        pkt_type: str = "debian",
        pkt_name: str = "",
        time_window: Optional[timedelta] = None,
    ) -> RepoInfo:
        """
        Collect complete repository information and return as RepoInfo model

        Args:
            repo_name: Repository name in format 'owner/repo'
            pkt_name: Package name (optional, defaults to repo name)
            time_window: Time window for filtering issues, PRs, and events.

        Returns:
            RepoInfo: Complete repository information
        """
        logger.info(f"Starting collection for repository: {repo_name}")

        try:
            if time_window:
                since_timestamp = datetime.now(tz=timezone.utc) - time_window
            else:
                since_timestamp = None

            # Get repository object
            repo = await self._run_in_executor(self._get_repository, repo_name)
            data_dir = get_data_dir()
            local_repo_dir = await self.local_utils.clone_repository(
                repo_name, repo.clone_url, self.executor
            )
            local_repo_path = data_dir / local_repo_dir

            # Gather all required information concurrently
            basic_info_task = asyncio.create_task(
                self.data_collector.get_basic_info(repo)
            )
            issues_task = asyncio.create_task(
                self.data_collector.get_issues(repo, since_timestamp)
            )
            prs_task = asyncio.create_task(
                self.data_collector.get_pull_requests(repo, since_timestamp)
            )
            # events are not used in the current implementation, but can be uncommented if needed
            # events_task = asyncio.create_task(
            #    self.data_collector.get_events(repo, since_timestamp))
            # commits are not used since it will make repo_info too large
            # commits_task = asyncio.create_task(
            #    self.data_collector.get_local_commits(local_repo_path, since_timestamp))

            # Use local binary file detection if repository was cloned
            binary_files_task = asyncio.create_task(
                self.data_collector.find_binary_files_local(local_repo_path)
            )
            workflows_task = asyncio.create_task(
                self.data_collector.get_workflows(repo, local_repo_path)
            )
            check_runs_task = asyncio.create_task(
                self.data_collector.get_check_runs(repo)
            )

            # Wait for all tasks to complete
            (
                basic_info,
                issues,
                pull_requests,
                binary_files,
                workflows,
                check_runs,
            ) = await asyncio.gather(
                basic_info_task,
                issues_task,
                prs_task,
                binary_files_task,
                workflows_task,
                check_runs_task,
            )

            events = []  # events_task.result()  # Uncomment if events are needed
            commits = []

            # Create RepoInfo model
            repo_info = RepoInfo(
                pkt_type=pkt_type,  # type: ignore
                pkt_name=pkt_name or repo_name.split("/")[-1],
                repo_id=repo_name.replace("/", "-"),
                url=repo.html_url,
                basic_info=basic_info,
                commit_list=commits,
                pr_list=pull_requests,
                issue_list=issues,
                binary_file_list=binary_files,
                local_repo_dir=local_repo_dir,
                event_list=events,
                workflow_list=workflows,
                check_run_list=check_runs,
            )

            logger.info(f"Successfully collected repo info for {repo_name}")
            logger.info(f"  - Issues: {len(issues)}")
            logger.info(f"  - Pull Requests: {len(pull_requests)}")
            logger.info(f"  - Commits: {len(commits)}")
            logger.info(f"  - Events: {len(events)}")
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
        self, repo_names: List[str], max_concurrent: int = 3, **kwargs
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
        repo_infos = [result for result in results if isinstance(result, RepoInfo)]

        logger.info(
            f"Successfully collected info for {len(repo_infos)}/{len(repo_names)} repositories"
        )
        return repo_infos


# Convenience function for single repository collection
async def collect_github_repo_info(
    repo_name: str,
    pkt_type: str = "debian",
    pkt_name: str = "",
    github_token: Optional[str] = None,
    **kwargs,
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
        return await collector.collect_repo_info(
            repo_name, pkt_type, pkt_name, **kwargs
        )


# Convenience function for multiple repositories collection
async def collect_multiple_github_repos(
    repo_names: List[str], github_token: Optional[str] = None, **kwargs
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
