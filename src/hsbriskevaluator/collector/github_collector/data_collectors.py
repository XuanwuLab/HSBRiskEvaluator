"""
Data collection methods for GitHub repository information.
"""

import asyncio
import logging
import time
from typing import List, Optional, Sequence, Set, Dict, TYPE_CHECKING

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
from functools import lru_cache
from tqdm import tqdm
from ...utils.progress_manager import get_progress_manager

progress_manager = get_progress_manager()
logger = progress_manager.logger


class GitHubDataCollector:
    """Handles data collection from GitHub API and local repositories"""

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        settings: Optional["CollectorSettings"] = None,
    ):
        self.executor = executor
        if settings is None:
            from ..settings import CollectorSettings

            settings = CollectorSettings()
        self.settings = settings

        # Initialize token manager
        from .token_manager import GitHubTokenManager

        self.token_manager = GitHubTokenManager(settings)

        self.converter = GitHubConverter()
        self.local_utils = LocalRepoUtils(self.settings)
        self._iteration_count = 0
        self._repo_cache: Dict[str, Repository] = {}

    @property
    def github(self):
        """Get GitHub client from token manager (if available) or fallback"""
        # Rotate token on every call
        self._iteration_count += 1
        logger.info(f"Rotating token on call {self._iteration_count}")
        if self._iteration_count % self.settings.token_rotation_interval == 0:
            logger.debug("Rotating GitHub token")
            self.token_manager.rotate_token()

        token_status = self.token_manager.get_token_stats()
        while token_status and token_status.core.remaining == 0:
            logger.warning(
                f"Token {self.token_manager._current_token[:8]}... exhausted, rotating to next token."
            )
            self.token_manager.rotate_token()
            token_status = self.token_manager.get_token_stats()
            time.sleep(60)

        return self.token_manager.get_github_client()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.executor.shutdown(wait=True)
        self.token_manager.close_all_clients()

    def _get_repository(self, repo_name: str) -> Repository:
        """Get repository with in-memory caching"""
        logger.debug(f"Fetching repository from GitHub API: {repo_name}")
        repo = self.github.get_repo(repo_name)
        logger.debug(f"Cached repository: {repo_name}")
        return repo

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

    async def get_basic_info(
        self, repo_name: str
    ) -> BasicInfo:
        """Get basic repository information"""
        logger.info(f"Collecting basic info for repository: {repo_name}")

        start_time = time.time()
        try:
            repo = self._get_repository(repo_name)

            basic_info = BasicInfo(
                description=repo.description or "",
                stargazers_count=repo.stargazers_count,
                watchers_count=repo.watchers_count,
                forks_count=repo.forks_count,
                url=repo.html_url or "",
                clone_url=repo.clone_url or "",
            )

            elapsed = time.time() - start_time
            logger.info(
                f"Successfully collected basic info for {repo_name} in {elapsed:.2f}s"
            )
            logger.debug(
                f"Basic info: stars={basic_info.stargazers_count}, "
                f"watchers={basic_info.watchers_count}, forks={basic_info.forks_count}"
            )

            return basic_info

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Failed to collect basic info for {repo_name} after {elapsed:.2f}s: {e}"
            )
            raise

    async def get_issues(
        self, repo_name: str, with_comment: bool = True
    ) -> List[Issue]:
        """
        Get repository issues (excluding pull requests)

        Args:
            repo_name: Repository name in format 'owner/repo'
            with_comment: Whether to include comments in the response
        """
        logger.info(f"Starting issue collection for repository: {repo_name} (with_comment={with_comment})")

        def _fetch_issues():
            issues = []
            if with_comment:
                max_issues = self.settings.get_max_issues()
                since_time = self.settings.get_issues_since_time()
            else:
                max_issues = self.settings.get_max_issues_without_comment()
                since_time = self.settings.get_issues_without_comment_since_time()
            since_timestamp = None
            if since_time:
                since_timestamp = datetime.now(tz=timezone.utc) - since_time

            start_time = time.time()
            count = 0
            page_count = 0
            rate_limit_hits = 0

            logger.info(
                f"Issue collection parameters: max_issues={max_issues}, since_time={since_time}"
            )

            try:
                # Get issues excluding pull requests
                logger.debug(f"Fetching issues for {repo_name} (state=all)")
                repo = self._get_repository(repo_name)
                issues_paginated = repo.get_issues(state="all")

                # Get total count if available (for progress tracking)
                try:
                    total_issues = issues_paginated.totalCount
                    logger.info(f"Total issues available in repository: {total_issues}")
                except Exception:
                    total_issues = None
                    logger.debug("Could not determine total issue count")

                # Determine effective limit for progress bar
                effective_limit = min(
                    max_issues or float("inf"), total_issues or float("inf")
                )
                if effective_limit == float("inf"):
                    effective_limit = None

                # Create progress bar with progress manager
                issue_desc = f"Issues ({'with' if with_comment else 'without'} comment) from {repo_name}"
                
                # Use progress manager for sub-progress tracking
                with progress_manager.create_sub_progress(
                    total=effective_limit,
                    desc=issue_desc,
                    unit="items",
                    parent_id="issues"
                ) as issues_progress:

                    # Iterate through all issues across all pages
                    for issue in issues_paginated:
                        page_count += 1
                        issues_progress.update(1)

                        # Check max limit
                        if max_issues is not None and count >= max_issues:
                            logger.info(f"Reached maximum issue limit: {max_issues}")
                            break

                        # Check time limit before processing to avoid unnecessary API calls
                        if (
                            since_timestamp is not None
                            and issue.created_at < since_timestamp
                        ):
                            logger.info(
                                f"Reached time limit: issue created at {issue.created_at} is older than {since_timestamp}"
                            )
                            break

                        # Skip pull requests (they appear in issues endpoint)
                        if issue.pull_request is None:
                            try:
                                converted_issue = self.converter.to_issue(issue, with_comment=with_comment)
                                issues.append(converted_issue)
                                count += 1
                                logger.debug(
                                    f"Collected issue #{issue.number}: {issue.title[:50]}..."
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to convert issue #{issue.number}: {e}"
                                )
                        else:
                            logger.debug(
                                f"Skipping PR #{issue.number} (appears in issues endpoint)"
                            )

                elapsed = time.time() - start_time
                logger.info(
                    f"Successfully retrieved {len(issues)} issues for {repo_name} in {elapsed:.2f}s"
                )
                logger.info(
                    f"Collection stats: processed {page_count} total items, "
                    f"collected {count} issues, rate limit hits: {rate_limit_hits}"
                )

                return issues

            except GithubException as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"GitHub API error fetching issues for {repo_name} after {elapsed:.2f}s: {e}"
                )
                return issues  # Return what we have so far

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Unexpected error fetching issues for {repo_name} after {elapsed:.2f}s: {e}"
                )
                return []

        return await self._run_in_executor(_fetch_issues)

    async def get_pull_requests(
        self, repo_name: str, with_comment: bool = True
    ) -> List[PullRequest]:
        """
        Get repository pull requests

        Args:
            repo_name: Repository name in format 'owner/repo'
            with_comment: Whether to include comments in the response
        """
        return await self._get_pull_requests_by_status(repo_name, "all", with_comment)

    async def get_merged_pull_requests(self, repo_name: str) -> List[PullRequest]:
        """Get merged pull requests only"""
        return await self._get_pull_requests_by_status(repo_name, "merged")

    async def get_closed_pull_requests(self, repo_name: str) -> List[PullRequest]:
        """Get closed (but not merged) pull requests only"""
        return await self._get_pull_requests_by_status(repo_name, "closed")

    async def get_open_pull_requests(self, repo_name: str) -> List[PullRequest]:
        """Get open pull requests only"""  
        return await self._get_pull_requests_by_status(repo_name, "open")

    async def _get_pull_requests_by_status(self, repo_name: str, status: str, with_comment: bool = True) -> List[PullRequest]:
        """
        Helper method to get pull requests by specific status
        
        Args:
            repo_name: Repository name in format 'owner/repo'
            status: PR status - 'merged', 'closed', or 'open'
            with_comment: Whether to include comments in the response
        """
        logger.info(f"Starting {status} pull request collection for repository: {repo_name} (with_comment={with_comment})")

        def _fetch_pull_requests_by_status():
            pull_requests = []
            
            # Get appropriate settings based on status and comment inclusion
            if with_comment:
                if status == "merged":
                    max_prs = self.settings.get_max_merged_pull_requests()
                    since_time = self.settings.get_merged_pull_requests_since_time()
                elif status == "closed":
                    max_prs = self.settings.get_max_closed_pull_requests()
                    since_time = self.settings.get_closed_pull_requests_since_time()
                elif status == "open":
                    max_prs = self.settings.get_max_open_pull_requests()
                    since_time = self.settings.get_open_pull_requests_since_time()
                else:  # all
                    max_prs = self.settings.get_max_pull_requests()
                    since_time = self.settings.get_pull_requests_since_time()
            else:
                max_prs = self.settings.get_max_pull_requests_without_comment()
                since_time = self.settings.get_pull_requests_without_comment_since_time()
            
            since_timestamp = None
            if since_time:
                since_timestamp = datetime.now(tz=timezone.utc) - since_time

            start_time = time.time()
            count = 0
            page_count = 0

            logger.info(f"{status.capitalize()} PR collection parameters: max_prs={max_prs}, since_time={since_time}")

            try:
                repo = self._get_repository(repo_name)
                # Get PRs with appropriate state filter
                if status == "open":
                    pulls_paginated = repo.get_pulls(state="open")
                else:
                    pulls_paginated = repo.get_pulls(state="closed")

                try:
                    total_prs = pulls_paginated.totalCount
                    logger.info(f"Total {status} PRs available in repository: {total_prs}")
                except Exception:
                    total_prs = None

                # Determine effective limit for progress bar
                effective_limit = min(max_prs or float("inf"), total_prs or float("inf"))
                if effective_limit == float("inf"):
                    effective_limit = None

                # Create progress bar
                pr_desc = f"{status.capitalize()} PRs ({'with' if with_comment else 'without'} comment) from {repo_name}"
                
                with progress_manager.create_sub_progress(
                    total=effective_limit,
                    desc=pr_desc,
                    unit="PRs",
                    parent_id=f"{status}_prs"
                ) as prs_progress:

                    for pr in pulls_paginated:
                        page_count += 1
                        prs_progress.update(1)

                        # Check max limit
                        if max_prs is not None and count >= max_prs:
                            logger.info(f"Reached maximum {status} PR limit: {max_prs}")
                            break

                        # Check time limit
                        if since_timestamp is not None and pr.created_at < since_timestamp:
                            logger.info(f"Reached time limit for {status} PRs")
                            break

                        # Filter by status (for closed state, we need to distinguish merged vs closed)
                        if status == "merged" and not pr.merged:
                            continue
                        elif status == "closed" and pr.merged:
                            continue

                        try:
                            converted_pr = self.converter.to_pull_request(pr, with_comment=with_comment)
                            pull_requests.append(converted_pr)
                            count += 1
                            logger.debug(f"Collected {status} PR #{pr.number}: {pr.title[:50]}...")
                        except Exception as e:
                            logger.warning(f"Failed to convert {status} PR #{pr.number}: {e}")

                elapsed = time.time() - start_time
                logger.info(f"Successfully retrieved {len(pull_requests)} {status} pull requests for {repo_name} in {elapsed:.2f}s")

                return pull_requests

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Error fetching {status} pull requests for {repo_name} after {elapsed:.2f}s: {e}")
                return []

        return await self._run_in_executor(_fetch_pull_requests_by_status)


    async def get_issues_without_comment(self, repo_name: str) -> List[Issue]:
        """Get repository issues without comments"""
        return await self.get_issues(repo_name, with_comment=False)

    async def get_pull_requests_without_comment(self, repo_name: str) -> List[PullRequest]:
        """Get repository pull requests without comments"""
        return await self._get_pull_requests_by_status(repo_name, "all", with_comment=False)

    async def get_events(
        self, repo_name: str
    ) -> List[GithubEvent]:
        """
        Get repository events with comprehensive progress tracking and logging

        Args:
            repo_name: Repository name in format 'owner/repo'

        Returns:
            List[GithubEvent]: List of repository events (excluding WatchEvent, ForkEvent, SponsorshipEvent)
        """
        logger.info(f"Starting event collection for repository: {repo_name}")

        def _fetch_events():
            events = []
            max_events = self.settings.get_max_events()
            since_time = self.settings.get_events_since_time()
            since_timestamp = None
            if since_time:
                since_timestamp = datetime.now(tz=timezone.utc) - since_time

            start_time = time.time()
            count = 0
            page_count = 0
            filtered_count = 0
            rate_limit_hits = 0

            logger.info(
                f"Event collection parameters: max_events={max_events}, since_time={since_time}"
            )

            try:
                # Get repository events
                logger.debug(f"Fetching events for {repo_name}")
                repo = self._get_repository(repo_name)
                events_paginated = repo.get_events()

                try:
                    total_events = events_paginated.totalCount
                    logger.info(f"Total events available in repository: {total_events}")
                except Exception:
                    total_events = None
                    logger.debug("Could not determine total event count")

                # Determine effective limit for progress bar
                effective_limit = min(
                    max_events or float("inf"), total_events or float("inf")
                )
                if effective_limit == float("inf"):
                    effective_limit = None

                # Create progress bar with progress manager
                events_desc = f"Events from {repo_name}"
                
                with progress_manager.create_sub_progress(
                    total=effective_limit,
                    desc=events_desc,
                    unit="events",
                    parent_id="events"
                ) as events_progress:

                    # Iterate through all events across all pages
                    for event in events_paginated:
                        page_count += 1
                        events_progress.update(1)

                        # Check max limit
                        if max_events is not None and count >= max_events:
                            logger.info(f"Reached maximum event limit: {max_events}")
                            break

                        # Check time limit before processing to avoid unnecessary API calls
                        if (
                            since_timestamp is not None
                            and event.created_at < since_timestamp
                        ):
                            logger.info(
                                f"Reached time limit: event created at {event.created_at} is older than {since_timestamp}"
                            )
                            break

                        # Convert event and filter out unwanted types
                        try:
                            converted_event = self.converter.to_github_event(event)
                            if converted_event:
                                events.append(converted_event)
                                count += 1
                                logger.debug(
                                    f"Collected event: {event.type} by {event.actor.login if event.actor else 'unknown'}"
                                )
                            else:
                                filtered_count += 1
                                logger.debug(f"Filtered out event: {event.type}")
                        except Exception as e:
                            logger.warning(f"Failed to convert event {event.type}: {e}")

                elapsed = time.time() - start_time
                logger.info(
                    f"Successfully retrieved {len(events)} events for {repo_name} in {elapsed:.2f}s"
                )
                logger.info(
                    f"Collection stats: processed {page_count} total events, "
                    f"collected {count} events, filtered {filtered_count}, rate limit hits: {rate_limit_hits}"
                )

                return events

            except GithubException as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"GitHub API error fetching events for {repo_name} after {elapsed:.2f}s: {e}"
                )
                return events  # Return what we have so far

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Unexpected error fetching events for {repo_name} after {elapsed:.2f}s: {e}"
                )
                return []

        return await self._run_in_executor(_fetch_events)

    async def get_commits(self, repo_name: str) -> List[Commit]:
        """
        Get repository commits with detailed information and proper pagination support

        Args:
            repo_name: Repository name in format 'owner/repo'
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
                repo = self._get_repository(repo_name)
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

                logger.info(f"Retrieved {len(commits)} commits for {repo_name}")
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
        self,
        repo_name: str,
        local_repo_path: Path,
    ) -> List[Workflow]:
        """Get GitHub Actions workflows with comprehensive progress tracking"""
        logger.info(f"Starting workflow collection for repository: {repo_name}")

        def _fetch_workflows():
            workflows = []
            max_workflows = self.settings.get_max_workflows()
            start_time = time.time()
            count = 0
            file_read_errors = 0
            rate_limit_hits = 0

            logger.info(
                f"Workflow collection parameters: max_workflows={max_workflows}"
            )

            try:
                logger.debug(f"Fetching workflows for {repo_name}")
                repo = self._get_repository(repo_name)
                workflows_paginated = repo.get_workflows()

                try:
                    total_workflows = workflows_paginated.totalCount
                    logger.info(
                        f"Total workflows available in repository: {total_workflows}"
                    )
                except Exception:
                    total_workflows = None
                    logger.debug("Could not determine total workflow count")

                # Create progress bar with progress manager
                workflows_desc = f"Workflows from {repo_name}"
                
                with progress_manager.create_sub_progress(
                    total=total_workflows,
                    desc=workflows_desc,
                    unit="workflows",
                    parent_id="workflows"
                ) as workflows_progress:

                    for workflow in workflows_paginated:
                        workflows_progress.update(1)
                        
                        # Check max limit
                        if max_workflows is not None and count >= max_workflows:
                            logger.info(f"Reached maximum workflow limit: {max_workflows}")
                            break

                        logger.debug(
                            f"Processing workflow: {workflow.name} (path: {workflow.path})"
                        )

                        # Read workflow file from local repository
                        file_path = local_repo_path / workflow.path
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            logger.debug(
                                f"Successfully read workflow file: {file_path} ({len(content)} chars)"
                            )
                        except FileNotFoundError:
                            file_read_errors += 1
                            content = "Not from a file in repo"
                            logger.warning(f"Workflow file not found locally: {file_path}")
                        except Exception as e:
                            file_read_errors += 1
                            content = f"Error reading file: {e}"
                            logger.warning(f"Error reading workflow file {file_path}: {e}")

                        try:
                            workflow_obj = Workflow(
                                name=workflow.name, content=content, path=workflow.path
                            )
                            workflows.append(workflow_obj)
                            count += 1
                            logger.debug(
                                f"Successfully collected workflow: {workflow.name}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to create workflow object for {workflow.name}: {e}"
                            )


                elapsed = time.time() - start_time
                logger.info(
                    f"Successfully retrieved {len(workflows)} workflows for {repo_name} in {elapsed:.2f}s"
                )
                logger.info(
                    f"Collection stats: processed {count} workflows, "
                    f"file read errors: {file_read_errors}, rate limit hits: {rate_limit_hits}"
                )

                return workflows

            except GithubException as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"GitHub API error fetching workflows for {repo_name} after {elapsed:.2f}s: {e}"
                )
                return workflows  # Return what we have so far

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Unexpected error fetching workflows for {repo_name} after {elapsed:.2f}s: {e}"
                )
                return []

        return await self._run_in_executor(_fetch_workflows)

    async def get_check_runs(
        self, repo_name: str
    ) -> List[CheckRun]:
        """Get check runs from recent commits and pull requests with comprehensive tracking"""
        logger.info(f"Starting check run collection for repository: {repo_name}")

        def _fetch_check_runs():
            check_runs: Set[str] = set()
            max_check_runs = self.settings.get_max_check_runs()
            start_time = time.time()
            commits_processed = 0
            prs_processed = 0
            rate_limit_hits = 0

            logger.info(
                f"Check run collection parameters: max_check_runs={max_check_runs}, "
                f"commit_limit={self.settings.check_runs_commit_limit}, "
                f"pr_limit={self.settings.check_runs_pr_limit}"
            )

            try:

                def add_check_run(commit: GithubCommit):
                    """Add check runs and statuses from a commit"""
                    local_adds = 0
                    try:
                        # Add GitHub check runs
                        for check_run in commit.get_check_runs():
                            if (
                                max_check_runs is not None
                                and len(check_runs) >= max_check_runs
                            ):
                                return True, local_adds  # Signal to break outer loop
                            if check_run.name not in check_runs:
                                check_runs.add(check_run.name)
                                local_adds += 1
                                logger.debug(
                                    f"Added check run: {check_run.name} from commit {commit.sha[:8]}"
                                )

                        # Add legacy status checks
                        for status in commit.get_statuses():
                            if (
                                max_check_runs is not None
                                and len(check_runs) >= max_check_runs
                            ):
                                return True, local_adds  # Signal to break outer loop
                            if status.context not in check_runs:
                                check_runs.add(status.context)
                                local_adds += 1
                                logger.debug(
                                    f"Added status check: {status.context} from commit {commit.sha[:8]}"
                                )

                    except Exception as e:
                        logger.warning(
                            f"Error processing check runs for commit {commit.sha[:8]}: {e}"
                        )

                    return False, local_adds

                # Check recent commits with tqdm
                logger.debug(
                    f"Processing recent commits (limit: {self.settings.check_runs_commit_limit})"
                )
                try:
                    repo = self._get_repository(repo_name)
                    commits_list = list(
                        repo.get_commits()[: self.settings.check_runs_commit_limit]
                    )
                    
                    with progress_manager.create_sub_progress(
                        total=len(commits_list),
                        desc=f"Check runs from commits",
                        unit="commits",
                        parent_id="check_runs_commits"
                    ) as commits_progress:

                        for commit in commits_list:
                            commits_progress.update(1)
                            commits_processed += 1
                            should_break, added = add_check_run(commit)

                            if should_break:
                                logger.info(
                                    f"Reached max check runs limit from commits: {max_check_runs}"
                                )
                                break
                except Exception as e:
                    logger.warning(f"Error processing commits for check runs: {e}")

                # Check pull requests if we haven't reached the limit
                if max_check_runs is None or len(check_runs) < max_check_runs:
                    
                    logger.debug(
                        f"Processing recent pull requests (limit: {self.settings.check_runs_pr_limit})"
                    )

                    try:
                        repo = self._get_repository(repo_name)
                        prs_list = list(
                            repo.get_pulls(state="all")[
                                : self.settings.check_runs_pr_limit
                            ]
                        )
                        
                        with progress_manager.create_sub_progress(
                            total=len(prs_list),
                            desc=f"Check runs from PRs",
                            unit="PRs",
                            parent_id="check_runs_prs"
                        ) as prs_progress:

                            for pull in prs_list:
                                prs_progress.update(1)
                                prs_processed += 1
                                pr_commits_processed = 0

                                try:
                                    for commit in pull.get_commits():
                                        pr_commits_processed += 1
                                        should_break, added = add_check_run(commit)
                                        if should_break:
                                            logger.info(
                                                f"Reached max check runs limit from PR #{pull.number}: {max_check_runs}"
                                            )
                                            break

                                    logger.debug(
                                        f"Processed PR #{pull.number} with {pr_commits_processed} commits"
                                    )

                                except Exception as e:
                                    logger.warning(
                                        f"Error processing commits for PR #{pull.number}: {e}"
                                    )

                                if (
                                    max_check_runs is not None
                                    and len(check_runs) >= max_check_runs
                                ):
                                    logger.info(
                                        f"Reached max check runs limit: {max_check_runs}"
                                    )
                                    break

                    except Exception as e:
                        logger.warning(
                            f"Error processing pull requests for check runs: {e}"
                        )


                elapsed = time.time() - start_time
                check_run_list = list(map(lambda name: CheckRun(name=name), check_runs))

                logger.info(
                    f"Successfully retrieved {len(check_run_list)} unique check runs for {repo_name} in {elapsed:.2f}s"
                )
                logger.info(
                    f"Collection stats: processed {commits_processed} commits, "
                    f"{prs_processed} PRs, rate limit hits: {rate_limit_hits}"
                )

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Check run names found: {sorted(list(check_runs))}")

                return check_run_list

            except GithubException as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"GitHub API error fetching check runs for {repo_name} after {elapsed:.2f}s: {e}"
                )
                return list(
                    map(lambda name: CheckRun(name=name), check_runs)
                )  # Return what we have so far

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Unexpected error fetching check runs for {repo_name} after {elapsed:.2f}s: {e}"
                )
                return []

        return await self._run_in_executor(_fetch_check_runs)

    async def find_binary_files_local(self, local_repo_path: Path) -> List[str]:
        """Find binary files in the locally cloned repository"""
        return await self.local_utils.find_binary_files_local(local_repo_path)
