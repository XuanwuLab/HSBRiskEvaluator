"""
Data collection methods for GitHub repository information.
"""

import asyncio
import logging
from typing import List, Optional, Set
from datetime import datetime
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

    def __init__(self, executor: ThreadPoolExecutor):
        self.executor = executor
        self.converter = GitHubConverter()
        self.local_utils = LocalRepoUtils()

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

    async def get_issues(
        self, repo: Repository, since_timestamp: Optional[datetime] = None
    ) -> List[Issue]:
        """
        Get repository issues (excluding pull requests) with proper pagination support

        Args:
            repo: Repository object
            since_timestamp: Fetch issues only after this timestamp
        """

        def _fetch_issues():
            issues = []
            try:
                # Get issues excluding pull requests
                issues_paginated = repo.get_issues(state="all")

                # Iterate through all issues across all pages
                for issue in issues_paginated:
                    # Check limit before processing to avoid unnecessary API calls
                    if (
                        since_timestamp is not None
                        and issue.created_at < since_timestamp
                    ):
                        break
                    # Skip pull requests (they appear in issues endpoint)
                    if issue.pull_request is None:
                        issues.append(self.converter.to_issue(issue))

                logger.info(f"Retrieved {len(issues)} issues for {repo.full_name}")
                return issues
            except GithubException as e:
                logger.error(f"Error fetching issues: {e}")
                return []

        return await self._run_in_executor(_fetch_issues)

    async def get_pull_requests(
        self, repo: Repository, since_timestamp: Optional[datetime] = None
    ) -> List[PullRequest]:
        """
        Get repository pull requests

        Args:
            repo: Repository object
            since_timestamp: Fetch PRs only after this timestamp
        """

        def _fetch_pull_requests():
            pull_requests = []
            try:
                pulls_paginated = repo.get_pulls(state="all")
                # Convert pull requests to model objects
                for pr in pulls_paginated:
                    if since_timestamp is not None and pr.created_at < since_timestamp:
                        break
                    pull_requests.append(self.converter.to_pull_request(pr))

                logger.info(
                    f"Retrieved {len(pull_requests)} pull requests for {repo.full_name}"
                )
                return pull_requests
            except GithubException as e:
                logger.error(f"Error fetching pull requests: {e}")
                return []

        return await self._run_in_executor(_fetch_pull_requests)

    async def get_events(
        self, repo: Repository, since_timestamp: Optional[datetime] = None
    ) -> List[GithubEvent]:
        """
        Get repository events using GitHub Events API with proper pagination support

        Args:
            repo: Repository object
            since_timestamp: Fetch events only after this timestamp

        Returns:
            List[GithubEvent]: List of repository events (excluding WatchEvent, ForkEvent, SponsorshipEvent)
        """

        def _fetch_events():
            events = []
            try:
                # Get repository events
                events_paginated = repo.get_events()
                logger.info(
                    f"Fetching events for {repo.full_name} get {events_paginated.totalCount} events from github api."
                )

                # Iterate through all events across all pages
                for event in events_paginated:
                    # Check limit before processing to avoid unnecessary API calls
                    if (
                        since_timestamp is not None
                        and event.created_at < since_timestamp
                    ):
                        break

                    # Convert event and filter out unwanted types
                    converted_event = self.converter.to_github_event(event)
                    if converted_event:
                        events.append(converted_event)

                logger.info(f"Retrieved {len(events)} events for {repo.full_name}")
                return events
            except GithubException as e:
                logger.error(f"Error fetching events: {e}")
                return []

        return await self._run_in_executor(_fetch_events)

    async def get_commits(
        self, repo: Repository, since_timestamp: Optional[datetime] = None
    ) -> List[Commit]:
        """
        Get repository commits with detailed information and proper pagination support

        Args:
            repo: Repository object
            since_timestamp: Fetch commits only after this timestamp
        """

        def _fetch_commits():
            commits = []
            try:
                # Get repository commits
                commits_paginated = repo.get_commits()

                # Iterate through all commits across all pages
                for commit in commits_paginated:
                    # Check limit before processing to avoid unnecessary API calls
                    if (
                        since_timestamp is not None
                        and commit.commit.committer.date < since_timestamp
                    ):
                        break

                    commits.append(self.converter.to_commit(commit))

                logger.info(f"Retrieved {len(commits)} commits for {repo.full_name}")
                return commits
            except GithubException as e:
                logger.error(f"Error fetching commits: {e}")
                return []

        return await self._run_in_executor(_fetch_commits)

    async def get_local_commits(
        self,
        local_repo_path: Path,
        since_timestamp: Optional[datetime] = None,
    ) -> List[Commit]:
        """
        Get commits from a local Git repository and map them to the Commit class.

        Args:
            local_repo_path: Path to the local Git repository.
            since_timestamp: Fetch commits only after this timestamp.

        Returns:
            List of Commit objects.
        """
        return await self.local_utils.get_local_commits(
            local_repo_path, since_timestamp
        )

    async def get_workflows(
        self, repo: Repository, local_repo_path: Path
    ) -> List[Workflow]:
        """Get GitHub Actions workflows"""

        def _fetch_workflows():
            workflows = []
            try:
                for workflow in repo.get_workflows():
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
            try:

                def add_check_run(commit: GithubCommit):
                    for check_run in commit.get_check_runs():
                        check_runs.add(check_run.name)
                    for status in commit.get_statuses():
                        check_runs.add(status.context)

                for commit in repo.get_commits()[:3]:
                    add_check_run(commit)
                for pull in repo.get_pulls(state="all")[:3]:
                    for commit in pull.get_commits():
                        add_check_run(commit)

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
