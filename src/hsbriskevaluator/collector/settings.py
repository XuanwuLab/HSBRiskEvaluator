from typing import Optional, List, Dict
from datetime import timedelta
from pydantic_settings import BaseSettings, SettingsConfigDict

class CollectorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HSB_COLLECTOR_")
    
    # APT Collector settings
    apt_max_concurrency: int = 3
    
    # GitHub Collector settings
    github_max_workers: int = 5
    github_tokens: Optional[List[str]] = None  # Multiple tokens for load balancing
    github_proxy_url: Optional[str] = None  # HTTP/HTTPS proxy URL
    
    # Local repository settings
    git_clone_timeout_seconds: int = 300
    
    # Global defaults (None means fetch all)
    global_max_count: Optional[int] = 100
    global_since_days: Optional[int] = None
    
    # Specific limits for each data type (None falls back to global)
    issues_max_count: Optional[int] = None
    issues_since_days: Optional[int] = None
    
    pull_requests_max_count: Optional[int] = None
    pull_requests_since_days: Optional[int] = None
    
    events_max_count: Optional[int] = None
    events_since_days: Optional[int] = None
    
    commits_max_count: Optional[int] = None
    commits_since_days: Optional[int] = None
    
    workflows_max_count: Optional[int] = None
    
    check_runs_max_count: Optional[int] = None
    check_runs_commit_limit: int = 3
    check_runs_pr_limit: int = 3
    
    def get_max_issues(self) -> Optional[int]:
        """Get max issues count with fallback to global default"""
        return self.issues_max_count or self.global_max_count
    
    def get_max_pull_requests(self) -> Optional[int]:
        """Get max pull requests count with fallback to global default"""
        return self.pull_requests_max_count or self.global_max_count
    
    def get_max_events(self) -> Optional[int]:
        """Get max events count with fallback to global default"""
        return self.events_max_count or self.global_max_count
    
    def get_max_commits(self) -> Optional[int]:
        """Get max commits count with fallback to global default"""
        return self.commits_max_count or self.global_max_count
    
    def get_max_workflows(self) -> Optional[int]:
        """Get max workflows count with fallback to global default"""
        return self.workflows_max_count or self.global_max_count
    
    def get_max_check_runs(self) -> Optional[int]:
        """Get max check runs count with fallback to global default"""
        return self.check_runs_max_count or self.global_max_count
    
    def get_issues_since_time(self) -> Optional[timedelta]:
        """Get issues since time with fallback to global default"""
        days = self.issues_since_days or self.global_since_days
        return timedelta(days=days) if days is not None else None
    
    def get_pull_requests_since_time(self) -> Optional[timedelta]:
        """Get pull requests since time with fallback to global default"""
        days = self.pull_requests_since_days or self.global_since_days
        return timedelta(days=days) if days is not None else None
    
    def get_events_since_time(self) -> Optional[timedelta]:
        """Get events since time with fallback to global default"""
        days = self.events_since_days or self.global_since_days
        return timedelta(days=days) if days is not None else None
    
    def get_commits_since_time(self) -> Optional[timedelta]:
        """Get commits since time with fallback to global default"""
        days = self.commits_since_days or self.global_since_days
        return timedelta(days=days) if days is not None else None