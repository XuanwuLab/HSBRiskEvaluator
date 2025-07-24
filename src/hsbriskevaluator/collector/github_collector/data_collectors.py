"""
Data collection methods for GitHub repository information.
"""

import asyncio
import logging
from typing import List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..settings import CollectorSettings
    from .token_manager import GitHubTokenManager
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from github.Repository import Repository
from github.Commit import Commit as GithubCommit
from github import GithubException

from ..repo_info import (
    BasicInfo,
    Issue,
    PullRequest,
    GithubEvent,
    Commit,
    Workflow,
    CheckRun,
)
from .converters import GitHubConverter
from .utils import LocalRepoUtils

logger = logging.getLogger(__name__)


class GitHubDataCollector:
    """Handles data collection from GitHub API and local repositories"""

    def __init__(self, executor: ThreadPoolExecutor, settings: Optional["CollectorSettings"] = None, 
                 token_manager: Optional["GitHubTokenManager"] = None):
        self.executor = executor
        if settings is None:
            from ..settings import CollectorSettings
            settings = CollectorSettings()
        self.settings = settings
        self.token_manager = token_manager
        self.converter = GitHubConverter()
        self.local_utils = LocalRepoUtils(self.settings)

    async def _run_in_executor(self, func, *args):
        """Run blocking GitHub API calls in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def get_basic_info(self, repo: Repository) -> BasicInfo:
        """Get basic repository information"""
        return BasicInfo(
            description=repo.description or "",
            stargazers_count=repo.stargazers_count,
            watchers_count=repo.watchers_count,
            forks_count=repo.forks_count,
        )

    async def get_issues(self, repo: Repository) -> List[Issue]:
        """
        Get repository issues (excluding pull requests) with proper pagination support

        Args:
            repo: Repository object
        """

        def _fetch_issues():
            issues = []
            max_issues = self.settings.get_max_issues()
            since_time = self.settings.get_issues_since_time()
            since_timestamp = None
            if since_time:
                since_timestamp = datetime.now(tz=timezone.utc) - since_time
            count = 0
            
            try:
                # Get issues excluding pull requests
                issues_paginated = repo.get_issues(state="all")

                # Iterate through all issues across all pages
                for issue in issues_paginated:
                    # Check max limit
                    if max_issues is not None and count >= max_issues:
                        break
                    
                    # Check time limit before processing to avoid unnecessary API calls
                    if (
                        since_timestamp is not None
                        and issue.created_at < since_timestamp
                    ):
                        break
                        
                    # Skip pull requests (they appear in issues endpoint)
                    if issue.pull_request is None:
                        issues.append(self.converter.to_issue(issue))
                        count += 1

                logger.info(f"Retrieved {len(issues)} issues for {repo.full_name}")
                return issues
            except GithubException as e:
                logger.error(f"Error fetching issues: {e}")
                return []

        return await self._run_in_executor(_fetch_issues)

    async def get_pull_requests(self, repo: Repository) -> List[PullRequest]:
        """
        Get repository pull requests

        Args:
            repo: Repository object
        """

        def _fetch_pull_requests():
            pull_requests = []
            max_prs = self.settings.get_max_pull_requests()
            since_time = self.settings.get_pull_requests_since_time()
            since_timestamp = None
            if since_time:
                since_timestamp = datetime.now(tz=timezone.utc) - since_time
            count = 0
            
            try:
                pulls_paginated = repo.get_pulls(state="all")
                # Convert pull requests to model objects
                for pr in pulls_paginated:
                    # Check max limit
                    if max_prs is not None and count >= max_prs:
                        break
                        
                    if since_timestamp is not None and pr.created_at < since_timestamp:
                        break
                    pull_requests.append(self.converter.to_pull_request(pr))
                    count += 1

                logger.info(
                    f"Retrieved {len(pull_requests)} pull requests for {repo.full_name}"
                )
                return pull_requests
            except GithubException as e:
                logger.error(f"Error fetching pull requests: {e}")
                return []

        return await self._run_in_executor(_fetch_pull_requests)

    async def get_events(self, repo: Repository) -> List[GithubEvent]:
        """
        Get repository events using GitHub Events API with proper pagination support

        Args:
            repo: Repository object

        Returns:
            List[GithubEvent]: List of repository events (excluding WatchEvent, ForkEvent, SponsorshipEvent)
        """

        def _fetch_events():
            events = []
            max_events = self.settings.get_max_events()
            since_time = self.settings.get_events_since_time()
            since_timestamp = None
            if since_time:
                since_timestamp = datetime.now(tz=timezone.utc) - since_time
            count = 0
            
            try:
                # Get repository events
                events_paginated = repo.get_events()
                logger.info(
                    f"Fetching events for {repo.full_name} get {events_paginated.totalCount} events from github api."
                )

                # Iterate through all events across all pages
                for event in events_paginated:
                    # Check max limit
                    if max_events is not None and count >= max_events:
                        break
                        
                    # Check time limit before processing to avoid unnecessary API calls
                    if (
                        since_timestamp is not None
                        and event.created_at < since_timestamp
                    ):
                        break

                    # Convert event and filter out unwanted types
                    converted_event = self.converter.to_github_event(event)
                    if converted_event:
                        events.append(converted_event)
                        count += 1

                logger.info(f"Retrieved {len(events)} events for {repo.full_name}")
                return events
            except GithubException as e:
                logger.error(f"Error fetching events: {e}")
                return []

        return await self._run_in_executor(_fetch_events)

    async def get_commits(self, repo: Repository) -> List[Commit]:
        """
        Get repository commits with detailed information and proper pagination support

        Args:
            repo: Repository object
        """

        def _fetch_commits():
            commits = []
            max_commits = self.settings.get_max_commits()
            since_time = self.settings.get_commits_since_time()
            since_timestamp = None
            if since_time:
                since_timestamp = datetime.now(tz=timezone.utc) - since_time
            count = 0
            
            try:
                # Get repository commits
                commits_paginated = repo.get_commits()

                # Iterate through all commits across all pages
                for commit in commits_paginated:
                    # Check max limit
                    if max_commits is not None and count >= max_commits:
                        break
                        
                    # Check time limit before processing to avoid unnecessary API calls
                    if (
                        since_timestamp is not None
                        and commit.commit.committer.date < since_timestamp
                    ):
                        break

                    commits.append(self.converter.to_commit(commit))
                    count += 1

                logger.info(f"Retrieved {len(commits)} commits for {repo.full_name}")
                return commits
            except GithubException as e:
                logger.error(f"Error fetching commits: {e}")
                return []

        return await self._run_in_executor(_fetch_commits)

    async def get_local_commits(self, local_repo_path: Path) -> List[Commit]:
        """
        Get commits from a local Git repository and map them to the Commit class.

        Args:
            local_repo_path: Path to the local Git repository.

        Returns:
            List of Commit objects.
        """
        return await self.local_utils.get_local_commits(local_repo_path)

    async def get_workflows(
        self, repo: Repository, local_repo_path: Path
    ) -> List[Workflow]:
        """Get GitHub Actions workflows"""

        def _fetch_workflows():
            workflows = []
            max_workflows = self.settings.get_max_workflows()
            count = 0
            
            try:
                for workflow in repo.get_workflows():
                    # Check max limit
                    if max_workflows is not None and count >= max_workflows:
                        break
                        
                    file_path = local_repo_path / workflow.path
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                    except FileNotFoundError:
                        content = "Not from a file in repo"
                    workflows.append(
                        Workflow(
                            name=workflow.name, content=content, path=workflow.path
                        )
                    )
                    count += 1
                    
                logger.info(
                    f"Retrieved {len(workflows)} workflows for {repo.full_name}"
                )
                return workflows
            except GithubException as e:
                logger.error(f"Error fetching workflows: {e}")
                return []

        return await self._run_in_executor(_fetch_workflows)

    async def get_check_runs(self, repo: Repository) -> List[CheckRun]:
        """Get check runs from recent commits and pull requests"""

        def _fetch_check_runs():
            check_runs: Set[str] = set()
            max_check_runs = self.settings.get_max_check_runs()
            
            try:

                def add_check_run(commit: GithubCommit):
                    for check_run in commit.get_check_runs():
                        if max_check_runs is not None and len(check_runs) >= max_check_runs:
                            return True  # Signal to break outer loop
                        check_runs.add(check_run.name)
                    for status in commit.get_statuses():
                        if max_check_runs is not None and len(check_runs) >= max_check_runs:
                            return True  # Signal to break outer loop
                        check_runs.add(status.context)
                    return False

                # Check commits
                for commit in repo.get_commits()[:self.settings.check_runs_commit_limit]:
                    if add_check_run(commit):
                        break
                
                # Check pull requests if we haven't reached the limit
                if max_check_runs is None or len(check_runs) < max_check_runs:
                    for pull in repo.get_pulls(state="all")[:self.settings.check_runs_pr_limit]:
                        for commit in pull.get_commits():
                            if add_check_run(commit):
                                break
                        if max_check_runs is not None and len(check_runs) >= max_check_runs:
                            break

                logger.info(
                    f"Retrieved {len(check_runs)} check runs for {repo.full_name}"
                )
                return list(map(lambda name: CheckRun(name=name), check_runs))
            except GithubException as e:
                logger.error(f"Error fetching checkruns: {e}")
                return []

        return await self._run_in_executor(_fetch_check_runs)

    async def find_binary_files_local(self, local_repo_path: Path) -> List[str]:
        """Find binary files in the locally cloned repository"""
        return await self.local_utils.find_binary_files_local(local_repo_path)
