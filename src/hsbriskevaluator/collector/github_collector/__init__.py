"""
GitHub repository information collector using the official PyGithub library.
This module provides functionality to collect repository information
and convert it to the Pydantic models defined in repo_info.py.
"""

import asyncio
import logging
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

from github import GithubException
from github.Repository import Repository
from dotenv import load_dotenv

from ..repo_info import RepoInfo
from ..settings import CollectorSettings
from ...utils.file import get_data_dir
from ...utils.progress_manager import get_progress_manager
from .data_collectors import GitHubDataCollector
from .utils import LocalRepoUtils

load_dotenv()
progress_manager = get_progress_manager()
logger = progress_manager.logger


class GitHubRepoCollector:
    """Collector for GitHub repository information using the official PyGithub library"""

    def __init__(
        self, settings: Optional[CollectorSettings] = None, show_progress: bool = True
    ):
        """
        Initialize the GitHub collector

        Args:
            settings: Collector settings (if not provided, defaults will be used)
            show_progress: Whether to show progress tracking
        """
        if settings is None:
            settings = CollectorSettings()
        self.settings = settings
        self.show_progress = show_progress

        self.executor = ThreadPoolExecutor(max_workers=settings.github_max_workers)
        self.local_utils = LocalRepoUtils(settings)

    async def _run_in_executor(self, func, *args):
        """Run blocking GitHub API calls in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def collect_repo_info(
        self,
        repo_name: str,
        pkt_type: str = "debian",
        pkt_name: str = "",
    ) -> RepoInfo:
        """
        Collect complete repository information and return as RepoInfo model

        Args:
            repo_name: Repository name in format 'owner/repo'
            pkt_name: Package name (optional, defaults to repo name)

        Returns:
            RepoInfo: Complete repository information
        """
        logger.info(f"Starting collection for repository: {repo_name}")

        try:
            async with GitHubDataCollector(
                self.executor, self.settings
            ) as data_collector:
                # Initialize repository
                logger.info(f"Getting basic info for {repo_name}")
                basic_info = await data_collector.get_basic_info(repo_name)

                # Clone repository
                logger.info(f"Cloning repository {repo_name}")
                data_dir = get_data_dir()
                local_repo_dir = await self.local_utils.clone_repository(
                    repo_name,
                    basic_info.clone_url,
                    self.executor,
                )
                local_repo_path = data_dir / local_repo_dir

                # Execute all collection tasks concurrently
                logger.info(f"Starting concurrent collection for {repo_name}")
                (
                    issues,
                    issues_without_comment,
                    merged_pull_requests,
                    closed_pull_requests,
                    open_pull_requests,
                    pr_without_comment,
                    binary_files,
                    workflows,
                    check_runs,
                ) = await asyncio.gather(
                    data_collector.get_issues(repo_name),
                    data_collector.get_issues_without_comment(repo_name),
                    data_collector.get_merged_pull_requests(repo_name),
                    data_collector.get_closed_pull_requests(repo_name),
                    data_collector.get_open_pull_requests(repo_name),
                    data_collector.get_pull_requests_without_comment(repo_name),
                    data_collector.find_binary_files_local(local_repo_path),
                    data_collector.get_workflows(repo_name, local_repo_path),
                    data_collector.get_check_runs(repo_name),
                )

                # Combine all pull requests as requested
                pull_requests = merged_pull_requests + closed_pull_requests + open_pull_requests

                # Finalize repo info  
                logger.info(f"Finalizing repository info for {repo_name}")
                events = []  # events not used in current implementation
                commits = []  # commits not used to avoid large repo_info

                # Create RepoInfo model
                repo_info = RepoInfo(
                    pkt_type=pkt_type,  # type: ignore
                    pkt_name=pkt_name or repo_name.split("/")[-1],
                    repo_id=repo_name.replace("/", "-"),
                    basic_info=basic_info,
                    commit_list=commits,
                    pr_list=pull_requests,
                    issue_list=issues,
                    issue_without_comment_list=issues_without_comment,
                    pr_without_comment_list=pr_without_comment,
                    binary_file_list=binary_files,
                    local_repo_dir=local_repo_dir,
                    event_list=events,
                    workflow_list=workflows,
                    check_run_list=check_runs,
                )

            logger.info(f"Successfully collected repo info for {repo_name}")
            logger.info(f"  - Issues: {len(issues)}")
            logger.info(f"  - Issues without comment: {len(issues_without_comment)}")
            logger.info(f"  - Pull Requests: {len(pull_requests)} (merged: {len(merged_pull_requests)}, closed: {len(closed_pull_requests)}, open: {len(open_pull_requests)})")
            logger.info(f"  - Pull Requests without comment: {len(pr_without_comment)}")
            logger.info(f"  - Commits: {len(commits)}")
            logger.info(f"  - Events: {len(events)}")
            logger.info(f"  - Binary Files: {len(binary_files)}")
            logger.info(f"  - Workflows: {len(workflows)}")
            logger.info(f"  - Check Runs: {len(check_runs)}")
            if local_repo_dir:
                logger.info(f"  - Local Repository: {local_repo_dir}")

            return repo_info

        finally:
            pass

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
        repo_infos = []

        async def collect_single_repo(repo_name: str) -> Optional[RepoInfo]:
            async with semaphore:
                try:
                    return await self.collect_repo_info(repo_name, **kwargs)
                except Exception as e:
                    logger.error(f"Failed to collect info for {repo_name}: {e}")
                    return None

        logger.info(f"Starting collection for {len(repo_names)} repositories")

        # Process repositories with controlled concurrency
        tasks = [
            asyncio.create_task(collect_single_repo(repo_name))
            for repo_name in repo_names
        ]

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
    settings: Optional[CollectorSettings] = None,
    show_progress: bool = True,
    **kwargs,
) -> RepoInfo:
    """
    Convenience function to collect information for a single repository

    Args:
        repo_name: Repository name in format 'owner/repo'
        pkt_name: Package name (optional)
        settings: Collector settings
        show_progress: Whether to show progress tracking
        **kwargs: Additional arguments

    Returns:
        RepoInfo: Repository information
    """
    collector = GitHubRepoCollector(settings, show_progress)
    return await collector.collect_repo_info(repo_name, pkt_type, pkt_name)


# Convenience function for multiple repositories collection
async def collect_multiple_github_repos(
    repo_names: List[str],
    settings: Optional[CollectorSettings] = None,
    show_progress: bool = True,
    **kwargs,
) -> List[RepoInfo]:
    """
    Convenience function to collect information for multiple repositories

    Args:
        repo_names: List of repository names
        settings: Collector settings
        show_progress: Whether to show progress tracking
        **kwargs: Additional arguments

    Returns:
        List[RepoInfo]: List of repository information
    """
    collector = GitHubRepoCollector(settings, show_progress)
    return await collector.collect_multiple_repos(repo_names)
