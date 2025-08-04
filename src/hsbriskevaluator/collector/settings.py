from typing import Optional, List, Dict
from datetime import timedelta
from pydantic_settings import BaseSettings, SettingsConfigDict


class CollectorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HSB_COLLECTOR_")

    # APT Collector settings
    apt_max_concurrency: int = 3

    # GitHub Collector settings
    github_max_workers: int = 5
    # Multiple tokens for load balancing
    github_tokens: Optional[List[str]] = None
    github_proxy_url: Optional[str] = None  # HTTP/HTTPS proxy URL

    # Local repository settings
    git_clone_timeout_seconds: int = 300

    # Global defaults (None means fetch all)
    global_max_count: Optional[int] = None
    global_since_days: Optional[int] = None

    # Specific limits for each data type (None falls back to global)
    issues_max_count: Optional[int] = 100
    issues_since_days: Optional[int] = None
    issues_without_comment_max_count: Optional[int] = 0
    issues_without_comment_since_days: Optional[int] = None

    pull_requests_max_count: Optional[int] = 100
    pull_requests_since_days: Optional[int] = None
    pull_requests_without_comment_max_count: Optional[int] = 1000
    pull_requests_without_comment_since_days: Optional[int] = 365*5

    # Separate limits for merged, closed, and open PRs
    merged_pull_requests_max_count: Optional[int] = 100
    merged_pull_requests_since_days: Optional[int] = None

    closed_pull_requests_max_count: Optional[int] = 100
    closed_pull_requests_since_days: Optional[int] = None

    open_pull_requests_max_count: Optional[int] = 100
    open_pull_requests_since_days: Optional[int] = None

    events_max_count: Optional[int] = None
    events_since_days: Optional[int] = None

    commits_max_count: Optional[int] = 1000
    commits_since_days: Optional[int] = 365 * 5

    workflows_max_count: Optional[int] = None

    check_runs_max_count: Optional[int] = 3
    check_runs_commit_limit: int = 3
    check_runs_pr_limit: int = 3

    token_rotation_interval: int = 5

    def get_max_issues(self) -> Optional[int]:
        """Get max issues count with fallback to global default"""
        return self.issues_max_count if self.issues_max_count is not None else self.global_max_count

    def get_max_pull_requests(self) -> Optional[int]:
        """Get max pull requests count with fallback to global default"""
        return self.pull_requests_max_count if self.pull_requests_max_count is not None else self.global_max_count

    def get_max_events(self) -> Optional[int]:
        """Get max events count with fallback to global default"""
        return self.events_max_count if self.events_max_count is not None else self.global_max_count

    def get_max_commits(self) -> Optional[int]:
        """Get max commits count with fallback to global default"""
        return self.commits_max_count if self.commits_max_count is not None else self.global_max_count

    def get_max_workflows(self) -> Optional[int]:
        """Get max workflows count with fallback to global default"""
        return self.workflows_max_count if self.workflows_max_count is not None else self.global_max_count

    def get_max_check_runs(self) -> Optional[int]:
        """Get max check runs count with fallback to global default"""
        return self.check_runs_max_count if self.check_runs_max_count is not None else self.global_max_count

    def get_issues_since_time(self) -> Optional[timedelta]:
        """Get issues since time with fallback to global default"""
        days = self.issues_since_days if self.issues_since_days is not None else self.global_since_days
        return timedelta(days=days) if days is not None else None

    def get_pull_requests_since_time(self) -> Optional[timedelta]:
        """Get pull requests since time with fallback to global default"""
        days = self.pull_requests_since_days if self.pull_requests_since_days is not None else self.global_since_days
        return timedelta(days=days) if days is not None else None

    def get_events_since_time(self) -> Optional[timedelta]:
        """Get events since time with fallback to global default"""
        days = self.events_since_days if self.events_since_days is not None else self.global_since_days
        return timedelta(days=days) if days is not None else None

    def get_commits_since_time(self) -> Optional[timedelta]:
        """Get commits since time with fallback to global default"""
        days = self.commits_since_days if self.commits_since_days is not None else self.global_since_days
        return timedelta(days=days) if days is not None else None

    # New getter methods for issues without comments
    def get_max_issues_without_comment(self) -> Optional[int]:
        """Get max issues without comment count with fallback to global default"""
        return self.issues_without_comment_max_count if self.issues_without_comment_max_count is not None else self.global_max_count

    def get_issues_without_comment_since_time(self) -> Optional[timedelta]:
        """Get issues without comment since time with fallback to global default"""
        days = self.issues_without_comment_since_days if self.issues_without_comment_since_days is not None else self.global_since_days
        return timedelta(days=days) if days is not None else None

    def get_max_pull_requests_without_comment(self) -> Optional[int]:
        """Get max pull requests without comment count with fallback to global default"""
        return self.pull_requests_without_comment_max_count if self.pull_requests_without_comment_max_count is not None else self.global_max_count

    def get_pull_requests_without_comment_since_time(self) -> Optional[timedelta]:
        """Get pull requests without comment since time with fallback to global default"""
        days = self.pull_requests_without_comment_since_days if self.pull_requests_without_comment_since_days is not None else self.global_since_days
        return timedelta(days=days) if days is not None else None

    def get_max_merged_pull_requests(self) -> Optional[int]:
        """Get max merged pull requests count with fallback to global default"""
        return self.merged_pull_requests_max_count if self.merged_pull_requests_max_count is not None else self.global_max_count

    def get_merged_pull_requests_since_time(self) -> Optional[timedelta]:
        """Get merged pull requests since time with fallback to global default"""
        days = self.merged_pull_requests_since_days if self.merged_pull_requests_since_days is not None else self.global_since_days
        return timedelta(days=days) if days is not None else None

    def get_max_closed_pull_requests(self) -> Optional[int]:
        """Get max closed pull requests count with fallback to global default"""
        return self.closed_pull_requests_max_count if self.closed_pull_requests_max_count is not None else self.global_max_count

    def get_closed_pull_requests_since_time(self) -> Optional[timedelta]:
        """Get closed pull requests since time with fallback to global default"""
        days = self.closed_pull_requests_since_days if self.closed_pull_requests_since_days is not None else self.global_since_days
        return timedelta(days=days) if days is not None else None

    def get_max_open_pull_requests(self) -> Optional[int]:
        """Get max open pull requests count with fallback to global default"""
        return self.open_pull_requests_max_count if self.open_pull_requests_max_count is not None else self.global_max_count

    def get_open_pull_requests_since_time(self) -> Optional[timedelta]:
        """Get open pull requests since time with fallback to global default"""
        days = self.open_pull_requests_since_days if self.open_pull_requests_since_days is not None else self.global_since_days
        return timedelta(days=days) if days is not None else None
