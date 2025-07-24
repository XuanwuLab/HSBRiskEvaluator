"""
Data collection methods for GitHub repository information.
"""

import asyncio
import logging
import time
from typing import List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..settings import CollectorSettings
    from .token_manager import GitHubTokenManager
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from github.Repository import Repository
from github.Commit import Commit as GithubCommit
from github import GithubException, RateLimitExceededException

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
from ...utils.progress import ProgressTracker, ProgressContext

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback for when tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

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

    def _get_github_client(self):
        """Get GitHub client from token manager (if available) or fallback"""
        if self.token_manager:
            return self.token_manager.get_github_client()
        return None
    
    def _handle_rate_limit_exception(self, e: RateLimitExceededException):
        """Handle rate limit exceptions with token rotation"""
        if self.token_manager:
            logger.warning(f"Rate limit hit, rotating token: {e}")
            self.token_manager.handle_rate_limit()
            self.token_manager.rotate_token()
        else:
            logger.warning(f"Rate limit hit (no token manager): {e}")
            
    async def _run_in_executor(self, func, *args):
        """Run blocking GitHub API calls in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def get_basic_info(self, repo: Repository, progress_tracker: Optional[ProgressTracker] = None) -> BasicInfo:
        """Get basic repository information with progress tracking"""
        logger.info(f"Collecting basic info for repository: {repo.full_name}")
        
        start_time = time.time()
        try:
            basic_info = BasicInfo(
                description=repo.description or "",
                stargazers_count=repo.stargazers_count,
                watchers_count=repo.watchers_count,
                forks_count=repo.forks_count,
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Successfully collected basic info for {repo.full_name} in {elapsed:.2f}s")
            logger.debug(f"Basic info: stars={basic_info.stargazers_count}, "
                        f"watchers={basic_info.watchers_count}, forks={basic_info.forks_count}")
            
            return basic_info
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to collect basic info for {repo.full_name} after {elapsed:.2f}s: {e}")
            raise

    async def get_issues(self, repo: Repository, progress_tracker: Optional[ProgressTracker] = None) -> List[Issue]:
        """
        Get repository issues (excluding pull requests) with comprehensive progress tracking

        Args:
            repo: Repository object
            progress_tracker: Optional progress tracker for detailed progress reporting
        """
        logger.info(f"Starting issue collection for repository: {repo.full_name}")
        
        def _fetch_issues():
            issues = []
            max_issues = self.settings.get_max_issues()
            since_time = self.settings.get_issues_since_time()
            since_timestamp = None
            if since_time:
                since_timestamp = datetime.now(tz=timezone.utc) - since_time
            
            start_time = time.time()
            count = 0
            page_count = 0
            rate_limit_hits = 0
            
            logger.info(f"Issue collection parameters: max_issues={max_issues}, since_time={since_time}")
            
            try:
                # Get issues excluding pull requests
                logger.debug(f"Fetching issues for {repo.full_name} (state=all)")
                issues_paginated = repo.get_issues(state="all")
                
                # Get total count if available (for progress tracking)
                try:
                    total_issues = issues_paginated.totalCount
                    logger.info(f"Total issues available in repository: {total_issues}")
                except Exception:
                    total_issues = None
                    logger.debug("Could not determine total issue count")

                # Determine effective limit for progress bar
                effective_limit = min(max_issues or float('inf'), total_issues or float('inf'))
                if effective_limit == float('inf'):
                    effective_limit = None

                # Create progress bar with tqdm
                issue_desc = f"Issues from {repo.full_name}"
                issues_iter = tqdm(
                    issues_paginated, 
                    desc=issue_desc,
                    total=effective_limit,
                    disable=not TQDM_AVAILABLE,
                    unit="items",
                    leave=False
                )

                # Iterate through all issues across all pages
                for issue in issues_iter:
                    page_count += 1
                    
                    # Check max limit
                    if max_issues is not None and count >= max_issues:
                        logger.info(f"Reached maximum issue limit: {max_issues}")
                        break
                    
                    # Check time limit before processing to avoid unnecessary API calls
                    if (
                        since_timestamp is not None
                        and issue.created_at < since_timestamp
                    ):
                        logger.info(f"Reached time limit: issue created at {issue.created_at} is older than {since_timestamp}")
                        break
                        
                    # Skip pull requests (they appear in issues endpoint)
                    if issue.pull_request is None:
                        try:
                            converted_issue = self.converter.to_issue(issue)
                            issues.append(converted_issue)
                            count += 1
                            logger.debug(f"Collected issue #{issue.number}: {issue.title[:50]}...")
                        except Exception as e:
                            logger.warning(f"Failed to convert issue #{issue.number}: {e}")
                    else:
                        logger.debug(f"Skipping PR #{issue.number} (appears in issues endpoint)")

                elapsed = time.time() - start_time
                logger.info(f"Successfully retrieved {len(issues)} issues for {repo.full_name} in {elapsed:.2f}s")
                logger.info(f"Collection stats: processed {page_count} total items, "
                           f"collected {count} issues, rate limit hits: {rate_limit_hits}")
                
                return issues
                
            except RateLimitExceededException as e:
                rate_limit_hits += 1
                self._handle_rate_limit_exception(e)
                elapsed = time.time() - start_time
                logger.warning(f"Rate limit hit during issue collection for {repo.full_name} after {elapsed:.2f}s")
                return issues  # Return what we have so far
                
            except GithubException as e:
                elapsed = time.time() - start_time
                logger.error(f"GitHub API error fetching issues for {repo.full_name} after {elapsed:.2f}s: {e}")
                return issues  # Return what we have so far
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Unexpected error fetching issues for {repo.full_name} after {elapsed:.2f}s: {e}")
                return []

        return await self._run_in_executor(_fetch_issues)

    async def get_pull_requests(self, repo: Repository, progress_tracker: Optional[ProgressTracker] = None) -> List[PullRequest]:
        """
        Get repository pull requests with comprehensive progress tracking

        Args:
            repo: Repository object
            progress_tracker: Optional progress tracker for detailed progress reporting
        """
        logger.info(f"Starting pull request collection for repository: {repo.full_name}")

        def _fetch_pull_requests():
            pull_requests = []
            max_prs = self.settings.get_max_pull_requests()
            since_time = self.settings.get_pull_requests_since_time()
            since_timestamp = None
            if since_time:
                since_timestamp = datetime.now(tz=timezone.utc) - since_time
            
            start_time = time.time()
            count = 0
            page_count = 0
            rate_limit_hits = 0
            
            logger.info(f"PR collection parameters: max_prs={max_prs}, since_time={since_time}")
            
            try:
                logger.debug(f"Fetching pull requests for {repo.full_name} (state=all)")
                pulls_paginated = repo.get_pulls(state="all")
                
                # Get total count if available
                try:
                    total_prs = pulls_paginated.totalCount
                    logger.info(f"Total pull requests available in repository: {total_prs}")
                except Exception:
                    total_prs = None
                    logger.debug("Could not determine total PR count")

                # Determine effective limit for progress bar
                effective_limit = min(max_prs or float('inf'), total_prs or float('inf'))
                if effective_limit == float('inf'):
                    effective_limit = None

                # Create progress bar with tqdm
                pr_desc = f"PRs from {repo.full_name}"
                prs_iter = tqdm(
                    pulls_paginated, 
                    desc=pr_desc,
                    total=effective_limit,
                    disable=not TQDM_AVAILABLE,
                    unit="PRs",
                    leave=False
                )
                
                # Convert pull requests to model objects
                for pr in prs_iter:
                    page_count += 1
                    
                    # Check max limit
                    if max_prs is not None and count >= max_prs:
                        logger.info(f"Reached maximum PR limit: {max_prs}")
                        break
                        
                    # Check time limit
                    if since_timestamp is not None and pr.created_at < since_timestamp:
                        logger.info(f"Reached time limit: PR created at {pr.created_at} is older than {since_timestamp}")
                        break
                    
                    try:
                        converted_pr = self.converter.to_pull_request(pr)
                        pull_requests.append(converted_pr)
                        count += 1
                        logger.debug(f"Collected PR #{pr.number}: {pr.title[:50]}... "
                                   f"(state: {pr.state}, merged: {pr.merged})")
                    except Exception as e:
                        logger.warning(f"Failed to convert PR #{pr.number}: {e}")

                elapsed = time.time() - start_time
                logger.info(f"Successfully retrieved {len(pull_requests)} pull requests for {repo.full_name} in {elapsed:.2f}s")
                logger.info(f"Collection stats: processed {page_count} total PRs, "
                           f"collected {count} PRs, rate limit hits: {rate_limit_hits}")
                
                return pull_requests
                
            except RateLimitExceededException as e:
                rate_limit_hits += 1
                self._handle_rate_limit_exception(e)
                elapsed = time.time() - start_time
                logger.warning(f"Rate limit hit during PR collection for {repo.full_name} after {elapsed:.2f}s")
                return pull_requests  # Return what we have so far
                
            except GithubException as e:
                elapsed = time.time() - start_time
                logger.error(f"GitHub API error fetching pull requests for {repo.full_name} after {elapsed:.2f}s: {e}")
                return pull_requests  # Return what we have so far
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Unexpected error fetching pull requests for {repo.full_name} after {elapsed:.2f}s: {e}")
                return []

        return await self._run_in_executor(_fetch_pull_requests)

    async def get_events(self, repo: Repository, progress_tracker: Optional[ProgressTracker] = None) -> List[GithubEvent]:
        """
        Get repository events with comprehensive progress tracking and logging

        Args:
            repo: Repository object
            progress_tracker: Optional progress tracker for detailed progress reporting

        Returns:
            List[GithubEvent]: List of repository events (excluding WatchEvent, ForkEvent, SponsorshipEvent)
        """
        logger.info(f"Starting event collection for repository: {repo.full_name}")

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
            
            logger.info(f"Event collection parameters: max_events={max_events}, since_time={since_time}")
            
            try:
                # Get repository events
                logger.debug(f"Fetching events for {repo.full_name}")
                events_paginated = repo.get_events()
                
                try:
                    total_events = events_paginated.totalCount
                    logger.info(f"Total events available in repository: {total_events}")
                except Exception:
                    total_events = None
                    logger.debug("Could not determine total event count")

                # Determine effective limit for progress bar
                effective_limit = min(max_events or float('inf'), total_events or float('inf'))
                if effective_limit == float('inf'):
                    effective_limit = None

                # Create progress bar with tqdm
                events_desc = f"Events from {repo.full_name}"
                events_iter = tqdm(
                    events_paginated, 
                    desc=events_desc,
                    total=effective_limit,
                    disable=not TQDM_AVAILABLE,
                    unit="events",
                    leave=False
                )

                # Iterate through all events across all pages
                for event in events_iter:
                    page_count += 1
                    
                    # Check max limit
                    if max_events is not None and count >= max_events:
                        logger.info(f"Reached maximum event limit: {max_events}")
                        break
                        
                    # Check time limit before processing to avoid unnecessary API calls
                    if (
                        since_timestamp is not None
                        and event.created_at < since_timestamp
                    ):
                        logger.info(f"Reached time limit: event created at {event.created_at} is older than {since_timestamp}")
                        break

                    # Convert event and filter out unwanted types
                    try:
                        converted_event = self.converter.to_github_event(event)
                        if converted_event:
                            events.append(converted_event)
                            count += 1
                            logger.debug(f"Collected event: {event.type} by {event.actor.login if event.actor else 'unknown'}")
                        else:
                            filtered_count += 1
                            logger.debug(f"Filtered out event: {event.type}")
                    except Exception as e:
                        logger.warning(f"Failed to convert event {event.type}: {e}")

                elapsed = time.time() - start_time
                logger.info(f"Successfully retrieved {len(events)} events for {repo.full_name} in {elapsed:.2f}s")
                logger.info(f"Collection stats: processed {page_count} total events, "
                           f"collected {count} events, filtered {filtered_count}, rate limit hits: {rate_limit_hits}")
                
                return events
                
            except RateLimitExceededException as e:
                rate_limit_hits += 1
                self._handle_rate_limit_exception(e)
                elapsed = time.time() - start_time
                logger.warning(f"Rate limit hit during event collection for {repo.full_name} after {elapsed:.2f}s")
                return events  # Return what we have so far
                
            except GithubException as e:
                elapsed = time.time() - start_time
                logger.error(f"GitHub API error fetching events for {repo.full_name} after {elapsed:.2f}s: {e}")
                return events  # Return what we have so far
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Unexpected error fetching events for {repo.full_name} after {elapsed:.2f}s: {e}")
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
        self, repo: Repository, local_repo_path: Path, progress_tracker: Optional[ProgressTracker] = None
    ) -> List[Workflow]:
        """Get GitHub Actions workflows with comprehensive progress tracking"""
        logger.info(f"Starting workflow collection for repository: {repo.full_name}")

        def _fetch_workflows():
            workflows = []
            max_workflows = self.settings.get_max_workflows()
            start_time = time.time()
            count = 0
            file_read_errors = 0
            rate_limit_hits = 0
            
            logger.info(f"Workflow collection parameters: max_workflows={max_workflows}")
            
            try:
                logger.debug(f"Fetching workflows for {repo.full_name}")
                workflows_paginated = repo.get_workflows()
                
                try:
                    total_workflows = workflows_paginated.totalCount
                    logger.info(f"Total workflows available in repository: {total_workflows}")
                except Exception:
                    total_workflows = None
                    logger.debug("Could not determine total workflow count")
                
                # Create progress bar with tqdm
                workflows_desc = f"Workflows from {repo.full_name}"
                workflows_iter = tqdm(
                    workflows_paginated, 
                    desc=workflows_desc,
                    total=total_workflows,
                    disable=not TQDM_AVAILABLE,
                    unit="workflows",
                    leave=False
                )

                for workflow in workflows_iter:
                    # Check max limit
                    if max_workflows is not None and count >= max_workflows:
                        logger.info(f"Reached maximum workflow limit: {max_workflows}")
                        break
                    
                    logger.debug(f"Processing workflow: {workflow.name} (path: {workflow.path})")
                    
                    # Read workflow file from local repository
                    file_path = local_repo_path / workflow.path
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        logger.debug(f"Successfully read workflow file: {file_path} ({len(content)} chars)")
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
                        logger.debug(f"Successfully collected workflow: {workflow.name}")
                    except Exception as e:
                        logger.warning(f"Failed to create workflow object for {workflow.name}: {e}")

                elapsed = time.time() - start_time
                logger.info(f"Successfully retrieved {len(workflows)} workflows for {repo.full_name} in {elapsed:.2f}s")
                logger.info(f"Collection stats: processed {count} workflows, "
                           f"file read errors: {file_read_errors}, rate limit hits: {rate_limit_hits}")
                
                return workflows
                
            except RateLimitExceededException as e:
                rate_limit_hits += 1
                self._handle_rate_limit_exception(e)
                elapsed = time.time() - start_time
                logger.warning(f"Rate limit hit during workflow collection for {repo.full_name} after {elapsed:.2f}s")
                return workflows  # Return what we have so far
                
            except GithubException as e:
                elapsed = time.time() - start_time
                logger.error(f"GitHub API error fetching workflows for {repo.full_name} after {elapsed:.2f}s: {e}")
                return workflows  # Return what we have so far
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Unexpected error fetching workflows for {repo.full_name} after {elapsed:.2f}s: {e}")
                return []

        return await self._run_in_executor(_fetch_workflows)

    async def get_check_runs(self, repo: Repository, progress_tracker: Optional[ProgressTracker] = None) -> List[CheckRun]:
        """Get check runs from recent commits and pull requests with comprehensive tracking"""
        logger.info(f"Starting check run collection for repository: {repo.full_name}")

        def _fetch_check_runs():
            check_runs: Set[str] = set()
            max_check_runs = self.settings.get_max_check_runs()
            start_time = time.time()
            commits_processed = 0
            prs_processed = 0
            rate_limit_hits = 0
            
            logger.info(f"Check run collection parameters: max_check_runs={max_check_runs}, "
                       f"commit_limit={self.settings.check_runs_commit_limit}, "
                       f"pr_limit={self.settings.check_runs_pr_limit}")
            
            try:
                def add_check_run(commit: GithubCommit):
                    """Add check runs and statuses from a commit"""
                    local_adds = 0
                    try:
                        # Add GitHub check runs
                        for check_run in commit.get_check_runs():
                            if max_check_runs is not None and len(check_runs) >= max_check_runs:
                                return True, local_adds  # Signal to break outer loop
                            if check_run.name not in check_runs:
                                check_runs.add(check_run.name)
                                local_adds += 1
                                logger.debug(f"Added check run: {check_run.name} from commit {commit.sha[:8]}")
                        
                        # Add legacy status checks
                        for status in commit.get_statuses():
                            if max_check_runs is not None and len(check_runs) >= max_check_runs:
                                return True, local_adds  # Signal to break outer loop
                            if status.context not in check_runs:
                                check_runs.add(status.context)
                                local_adds += 1
                                logger.debug(f"Added status check: {status.context} from commit {commit.sha[:8]}")
                                
                    except Exception as e:
                        logger.warning(f"Error processing check runs for commit {commit.sha[:8]}: {e}")
                    
                    return False, local_adds

                # Check recent commits with tqdm
                logger.debug(f"Processing recent commits (limit: {self.settings.check_runs_commit_limit})")
                try:
                    commits_list = list(repo.get_commits()[:self.settings.check_runs_commit_limit])
                    commits_iter = tqdm(
                        commits_list, 
                        desc=f"Check runs from commits",
                        disable=not TQDM_AVAILABLE,
                        unit="commits",
                        leave=False
                    )
                    
                    for commit in commits_iter:
                        commits_processed += 1
                        should_break, added = add_check_run(commit)
                        
                        
                        if should_break:
                            logger.info(f"Reached max check runs limit from commits: {max_check_runs}")
                            break
                except Exception as e:
                    logger.warning(f"Error processing commits for check runs: {e}")
                
                # Check pull requests if we haven't reached the limit
                if max_check_runs is None or len(check_runs) < max_check_runs:
                    logger.debug(f"Processing recent pull requests (limit: {self.settings.check_runs_pr_limit})")
                    
                    try:
                        prs_list = list(repo.get_pulls(state="all")[:self.settings.check_runs_pr_limit])
                        prs_iter = tqdm(
                            prs_list, 
                            desc=f"Check runs from PRs",
                            disable=not TQDM_AVAILABLE,
                            unit="PRs",
                            leave=False
                        )
                        
                        for pull in prs_iter:
                            prs_processed += 1
                            pr_commits_processed = 0
                            
                            try:
                                for commit in pull.get_commits():
                                    pr_commits_processed += 1
                                    should_break, added = add_check_run(commit)
                                    if should_break:
                                        logger.info(f"Reached max check runs limit from PR #{pull.number}: {max_check_runs}")
                                        break
                                        
                                logger.debug(f"Processed PR #{pull.number} with {pr_commits_processed} commits")
                                        
                            except Exception as e:
                                logger.warning(f"Error processing commits for PR #{pull.number}: {e}")
                                
                            if max_check_runs is not None and len(check_runs) >= max_check_runs:
                                logger.info(f"Reached max check runs limit: {max_check_runs}")
                                break
                                
                    except Exception as e:
                        logger.warning(f"Error processing pull requests for check runs: {e}")

                elapsed = time.time() - start_time
                check_run_list = list(map(lambda name: CheckRun(name=name), check_runs))
                
                logger.info(f"Successfully retrieved {len(check_run_list)} unique check runs for {repo.full_name} in {elapsed:.2f}s")
                logger.info(f"Collection stats: processed {commits_processed} commits, "
                           f"{prs_processed} PRs, rate limit hits: {rate_limit_hits}")
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Check run names found: {sorted(list(check_runs))}")
                
                return check_run_list
                
            except RateLimitExceededException as e:
                rate_limit_hits += 1
                self._handle_rate_limit_exception(e)
                elapsed = time.time() - start_time
                logger.warning(f"Rate limit hit during check run collection for {repo.full_name} after {elapsed:.2f}s")
                return list(map(lambda name: CheckRun(name=name), check_runs))  # Return what we have so far
                
            except GithubException as e:
                elapsed = time.time() - start_time
                logger.error(f"GitHub API error fetching check runs for {repo.full_name} after {elapsed:.2f}s: {e}")
                return list(map(lambda name: CheckRun(name=name), check_runs))  # Return what we have so far
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Unexpected error fetching check runs for {repo.full_name} after {elapsed:.2f}s: {e}")
                return []

        return await self._run_in_executor(_fetch_check_runs)

    async def find_binary_files_local(self, local_repo_path: Path) -> List[str]:
        """Find binary files in the locally cloned repository"""
        return await self.local_utils.find_binary_files_local(local_repo_path)
