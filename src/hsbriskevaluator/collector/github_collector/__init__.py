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
from ..settings import CollectorSettings
from ...utils.file import get_data_dir
from ...utils.progress import create_progress_tracker, ProgressContext, ProgressTracker
from .data_collectors import GitHubDataCollector
from .utils import LocalRepoUtils
from .token_manager import GitHubTokenManager

load_dotenv()
logger = logging.getLogger(__name__)


class GitHubRepoCollector:
    """Collector for GitHub repository information using the official PyGithub library"""

    def __init__(self, github_token: Optional[str] = None, settings: Optional[CollectorSettings] = None, 
                 show_progress: bool = True):
        """
        Initialize the GitHub collector

        Args:
            github_token: GitHub API token. If not provided, will use GITHUB_TOKEN env var or settings
            settings: Collector settings (if not provided, defaults will be used)
            show_progress: Whether to show progress tracking
        """
        if settings is None:
            settings = CollectorSettings()
        self.settings = settings
        self.show_progress = show_progress
        
        # Setup token management
        tokens = []
        
        # Priority: 1. Direct parameter, 2. Settings multi-tokens, 3. GITHUB_TOKEN env var
        if github_token:
            tokens = [github_token]
        elif settings.github_tokens:
            tokens = settings.github_tokens
        else:
            env_token = os.getenv("GITHUB_TOKEN")
            if env_token:
                tokens = [env_token]
        
        if not tokens:
            raise ValueError(
                "GitHub token(s) required. Set GITHUB_TOKEN environment variable, "
                "pass token directly, or configure github_tokens in settings."
            )
        
        # Initialize token manager
        self.token_manager = GitHubTokenManager(tokens, settings.github_proxy_url)
        
        # For backward compatibility, expose a github client property
        self._current_github = None
        
        self.executor = ThreadPoolExecutor(max_workers=settings.github_max_workers)
        self.data_collector = GitHubDataCollector(self.executor, settings, self.token_manager)
        self.local_utils = LocalRepoUtils(settings)
    
    @property
    def github(self) -> Github:
        """Get current GitHub client (for backward compatibility)"""
        if self._current_github is None:
            self._current_github = self.token_manager.get_github_client()
        return self._current_github
    
    def _get_fresh_github_client(self) -> Github:
        """Get a fresh GitHub client, potentially with a different token"""
        self._current_github = self.token_manager.get_github_client()
        return self._current_github
    
    def get_token_stats(self) -> dict:
        """Get usage statistics for all configured tokens"""
        return self.token_manager.get_token_stats()
    
    def rotate_token(self) -> None:
        """Manually rotate to the next token"""
        self.token_manager.rotate_token()
        self._current_github = None  # Force refresh on next access

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.executor.shutdown(wait=True)
        self.token_manager.close_all_clients()

    @retry(
        retry=retry_if_exception_type(RateLimitExceededException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=60, max=300),
    )
    def _get_repository(self, repo_name: str) -> Repository:
        """Get repository with retry logic for rate limiting and token rotation"""
        try:
            github_client = self._get_fresh_github_client()
            return github_client.get_repo(repo_name)
        except RateLimitExceededException as e:
            # Mark current token as rate limited and rotate
            self.token_manager.handle_rate_limit()
            self.token_manager.rotate_token()
            logger.warning(f"Rate limit exceeded, rotated to next token")
            raise e

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
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> RepoInfo:
        """
        Collect complete repository information and return as RepoInfo model

        Args:
            repo_name: Repository name in format 'owner/repo'
            pkt_name: Package name (optional, defaults to repo name)
            progress_tracker: Optional progress tracker for external tracking

        Returns:
            RepoInfo: Complete repository information
        """
        # Create progress tracker if not provided
        if progress_tracker is None:
            progress_tracker = create_progress_tracker(
                name=f"Collecting {repo_name}", 
                show_progress=self.show_progress
            )
            show_summary = True
        else:
            show_summary = False

        # Setup todos
        progress_tracker.add_todos([
            ("init", "Initialize repository"),
            ("clone", "Clone repository"),
            ("basic_info", "Collect basic info"),
            ("issues", "Collect issues"),
            ("prs", "Collect pull requests"),
            ("binary_files", "Find binary files"),
            ("workflows", "Collect workflows"),
            ("check_runs", "Collect check runs"),
            ("finalize", "Finalize repository info")
        ])

        logger.info(f"Starting collection for repository: {repo_name}")

        try:
            # Initialize repository
            with ProgressContext(progress_tracker, "init") as init_ctx:
                with init_ctx.step("github_auth", "Authenticate with GitHub API"):
                    repo = await self._run_in_executor(self._get_repository, repo_name)
                
                with init_ctx.step("setup_dirs", "Setup local directories"):
                    data_dir = get_data_dir()

            # Clone repository
            with ProgressContext(progress_tracker, "clone") as clone_ctx:
                local_repo_dir = await self.local_utils.clone_repository(
                    repo_name, repo.clone_url, self.executor, progress_tracker
                )
                local_repo_path = data_dir / local_repo_dir

            # Collect data concurrently with progress tracking
            async def collect_basic_info():
                with ProgressContext(progress_tracker, "basic_info") as ctx:
                    with ctx.step("fetch_repo_info", "Fetch repository metadata"):
                        return await self.data_collector.get_basic_info(repo, progress_tracker)

            async def collect_issues():
                with ProgressContext(progress_tracker, "issues") as ctx:
                    max_issues = self.settings.get_max_issues()
                    details = f"Max: {max_issues}" if max_issues else "Unlimited"
                    with ctx.step("fetch_issues", "Fetch issues from GitHub API", details):
                        return await self.data_collector.get_issues(repo, progress_tracker)

            async def collect_prs():
                with ProgressContext(progress_tracker, "prs") as ctx:
                    max_prs = self.settings.get_max_pull_requests() 
                    details = f"Max: {max_prs}" if max_prs else "Unlimited"
                    with ctx.step("fetch_prs", "Fetch pull requests from GitHub API", details):
                        return await self.data_collector.get_pull_requests(repo, progress_tracker)

            async def collect_binary_files():
                with ProgressContext(progress_tracker, "binary_files") as ctx:
                    with ctx.step("scan_files", "Scan local repository for binary files", f"Scanning {local_repo_path}"):
                        return await self.data_collector.find_binary_files_local(local_repo_path)

            async def collect_workflows():
                with ProgressContext(progress_tracker, "workflows") as ctx:
                    with ctx.step("fetch_workflows", "Fetch GitHub Actions workflows"):
                        return await self.data_collector.get_workflows(repo, local_repo_path, progress_tracker)

            async def collect_check_runs():
                with ProgressContext(progress_tracker, "check_runs") as ctx:
                    limit = self.settings.check_runs_commit_limit
                    with ctx.step("fetch_check_runs", "Fetch check runs from recent commits", f"Last {limit} commits"):
                        return await self.data_collector.get_check_runs(repo, progress_tracker)

            # Execute all collection tasks concurrently
            (
                basic_info,
                issues,
                pull_requests,
                binary_files,
                workflows,
                check_runs,
            ) = await asyncio.gather(
                collect_basic_info(),
                collect_issues(),
                collect_prs(),
                collect_binary_files(),
                collect_workflows(),
                collect_check_runs(),
            )

            # Finalize repo info
            with ProgressContext(progress_tracker, "finalize") as final_ctx:
                with final_ctx.step("prepare_data", "Prepare collected data"):
                    events = []  # events not used in current implementation
                    commits = []  # commits not used to avoid large repo_info

                with final_ctx.step("create_repoinfo", "Create final RepoInfo object", f"Combining data for {repo_name}"):
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
            logger.info(f"  - Workflows: {len(workflows)}")
            logger.info(f"  - Check Runs: {len(check_runs)}")
            if local_repo_dir:
                logger.info(f"  - Local Repository: {local_repo_dir}")

            if show_summary:
                progress_tracker.print_summary()

            return repo_info

        except GithubException as e:
            logger.error(f"GitHub API error collecting repo info for {repo_name}: {e}")
            if show_summary:
                progress_tracker.print_summary()
            raise
        except Exception as e:
            logger.error(f"Error collecting repo info for {repo_name}: {e}")
            if show_summary:
                progress_tracker.print_summary()
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
        # Create master progress tracker
        master_tracker = create_progress_tracker(
            name=f"Collecting {len(repo_names)} repositories",
            show_progress=self.show_progress
        )
        
        # Add todos for each repository
        todos = [(f"repo_{i}", f"Collect {repo_name}") for i, repo_name in enumerate(repo_names)]
        master_tracker.add_todos(todos)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        repo_infos = []

        async def collect_single_repo(i: int, repo_name: str) -> Optional[RepoInfo]:
            async with semaphore:
                todo_id = f"repo_{i}"
                try:
                    with ProgressContext(master_tracker, todo_id):
                        # Create sub-tracker for this repository
                        sub_tracker = create_progress_tracker(
                            name=f"{repo_name}",
                            show_progress=False  # Don't show sub-progress for multiple repos
                        )
                        return await self.collect_repo_info(repo_name, progress_tracker=sub_tracker, **kwargs)
                except Exception as e:
                    logger.error(f"Failed to collect info for {repo_name}: {e}")
                    return None

        logger.info(f"Starting collection for {len(repo_names)} repositories")

        # Use tqdm for overall progress if available
        if self.show_progress:
            progress_bar = master_tracker.create_progress_bar(
                len(repo_names), 
                desc="Collecting repositories"
            )
        else:
            progress_bar = None

        # Process repositories with controlled concurrency
        tasks = []
        for i, repo_name in enumerate(repo_names):
            task = asyncio.create_task(collect_single_repo(i, repo_name))
            tasks.append(task)
            
            # Update progress as tasks complete
            if progress_bar:
                async def update_progress(task):
                    result = await task
                    progress_bar.update(1)
                    return result
                tasks[-1] = asyncio.create_task(update_progress(task))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        

        # Filter out None results and exceptions
        repo_infos = [result for result in results if isinstance(result, RepoInfo)]

        logger.info(
            f"Successfully collected info for {len(repo_infos)}/{len(repo_names)} repositories"
        )
        
        master_tracker.print_summary()
        return repo_infos


# Convenience function for single repository collection
async def collect_github_repo_info(
    repo_name: str,
    pkt_type: str = "debian",
    pkt_name: str = "",
    github_token: Optional[str] = None,
    settings: Optional[CollectorSettings] = None,
    show_progress: bool = True,
    **kwargs,
) -> RepoInfo:
    """
    Convenience function to collect information for a single repository

    Args:
        repo_name: Repository name in format 'owner/repo'
        pkt_name: Package name (optional)
        github_token: GitHub API token (optional, uses env var if not provided)
        settings: Collector settings
        show_progress: Whether to show progress tracking
        **kwargs: Additional arguments

    Returns:
        RepoInfo: Repository information
    """
    async with GitHubRepoCollector(github_token, settings, show_progress) as collector:
        return await collector.collect_repo_info(
            repo_name, pkt_type, pkt_name, **kwargs
        )


# Convenience function for multiple repositories collection
async def collect_multiple_github_repos(
    repo_names: List[str], 
    github_token: Optional[str] = None, 
    settings: Optional[CollectorSettings] = None, 
    show_progress: bool = True,
    **kwargs
) -> List[RepoInfo]:
    """
    Convenience function to collect information for multiple repositories

    Args:
        repo_names: List of repository names
        github_token: GitHub API token (optional, uses env var if not provided)
        settings: Collector settings
        show_progress: Whether to show progress tracking
        **kwargs: Additional arguments

    Returns:
        List[RepoInfo]: List of repository information
    """
    async with GitHubRepoCollector(github_token, settings, show_progress) as collector:
        return await collector.collect_multiple_repos(repo_names, **kwargs)
