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
from ...utils.progress import create_progress_tracker, ProgressContext, ProgressTracker
from .data_collectors import GitHubDataCollector
from .utils import LocalRepoUtils

load_dotenv()
logger = logging.getLogger(__name__)


class GitHubRepoCollector:
    """Collector for GitHub repository information using the official PyGithub library"""

    def __init__(self, settings: Optional[CollectorSettings] = None, 
                 show_progress: bool = True):
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
            ("clone_repository", "Clone repository"),
            ("get_basic_info", "Collect basic info"),
            ("issues", "Collect issues"),
            ("prs", "Collect pull requests"),
            ("binary_files", "Find binary files"),
            ("workflows", "Collect workflows"),
            ("check_runs", "Collect check runs"),
            ("finalize", "Finalize repository info")
        ])

        logger.info(f"Starting collection for repository: {repo_name}")

        async with GitHubDataCollector(self.executor, self.settings) as data_collector:
        # Initialize repository
            with ProgressContext(progress_tracker, "get_basic_info") as ctx:
                with ctx.step("get_basic_info", "Fetch repository metadata"):
                        basic_info =await  data_collector.get_basic_info(repo_name, progress_tracker)
                
            # Clone repository
            with ProgressContext(progress_tracker, "clone_repository") as clone_ctx:
                with clone_ctx.step("clone_repository", "Clone repository to local storage"):
                    data_dir = get_data_dir()
                    local_repo_dir = await self.local_utils.clone_repository(
                        repo_name, basic_info.clone_url, self.executor, progress_tracker
                    )
                    local_repo_path = data_dir / local_repo_dir


            async def collect_issues():
                with ProgressContext(progress_tracker, "issues") as ctx:
                    max_issues = self.settings.get_max_issues()
                    details = f"Max: {max_issues}" if max_issues else "Unlimited"
                    with ctx.step("fetch_issues", "Fetch issues from GitHub API", details):
                        return await data_collector.get_issues(repo_name, progress_tracker)

            async def collect_prs():
                with ProgressContext(progress_tracker, "prs") as ctx:
                    max_prs = self.settings.get_max_pull_requests() 
                    details = f"Max: {max_prs}" if max_prs else "Unlimited"
                    with ctx.step("fetch_prs", "Fetch pull requests from GitHub API", details):
                        return await data_collector.get_pull_requests(repo_name, progress_tracker)

            async def collect_binary_files():
                with ProgressContext(progress_tracker, "binary_files") as ctx:
                    with ctx.step("scan_files", "Scan local repository for binary files", f"Scanning {local_repo_path}"):
                        return await data_collector.find_binary_files_local(local_repo_path)

            async def collect_workflows():
                with ProgressContext(progress_tracker, "workflows") as ctx:
                    with ctx.step("fetch_workflows", "Fetch GitHub Actions workflows"):
                        return await data_collector.get_workflows(repo_name, local_repo_path, progress_tracker)

            async def collect_check_runs():
                with ProgressContext(progress_tracker, "check_runs") as ctx:
                    limit = self.settings.check_runs_commit_limit
                    with ctx.step("fetch_check_runs", "Fetch check runs from recent commits", f"Last {limit} commits"):
                        return await data_collector.get_check_runs(repo_name, progress_tracker)

            # Execute all collection tasks concurrently
            (
                issues,
                pull_requests,
                binary_files,
                workflows,
                check_runs,
            ) = await asyncio.gather(
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
    return await collector.collect_repo_info(
            repo_name, pkt_type, pkt_name, **kwargs
    )


# Convenience function for multiple repositories collection
async def collect_multiple_github_repos(
    repo_names: List[str], 
    settings: Optional[CollectorSettings] = None, 
    show_progress: bool = True,
    **kwargs
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
    return await collector.collect_multiple_repos(repo_names, **kwargs)
