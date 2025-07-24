"""
Information gathering module for HSBRiskEvaluator.

This module provides functionality to collect repository information
from various sources, particularly GitHub repositories, and package
dependency information from APT.
"""

from typing import Optional
from datetime import timedelta
from .repo_info import Comment, Issue, PullRequest, RepoInfo, Dependent, User
from .settings import CollectorSettings
from ..utils.progress import create_progress_tracker, ProgressContext, ProgressTracker

from .github_collector import (
    GitHubRepoCollector,
    collect_github_repo_info,
    collect_multiple_github_repos,
)

from .apt_collector import (
    APTCollector,
    collect_package_dependencies,
    enrich_repo_with_dependencies,
    enrich_multiple_repos_with_dependencies,
)

__all__ = [
    # Pydantic models
    "User",
    "Comment",
    "Issue",
    "PullRequest",
    "RepoInfo",
    "Dependent",
    # Settings
    "CollectorSettings",
    # Main collector function
    "collect_all",
    # GitHub collector
    "GitHubRepoCollector",
    "collect_github_repo_info",
    "collect_multiple_github_repos",
    # Dependency collector
    "APTCollector",
    "collect_package_dependencies",
    "enrich_repo_with_dependencies",
    "enrich_multiple_repos_with_dependencies",
]


async def collect_all(
    pkt_type: str,
    pkt_name: str,
    repo_name: str,
    settings: Optional[CollectorSettings] = None,
    show_progress: bool = True,
):
    """
    Collect all information about a repository with progress tracking
    
    Args:
        pkt_type: Package type (e.g., "debian")
        pkt_name: Package name
        repo_name: Repository name in format 'owner/repo'
        settings: Collector settings
        show_progress: Whether to show progress tracking
        
    Returns:
        RepoInfo: Complete repository information
    """
    # Create progress tracker
    progress_tracker = create_progress_tracker(
        name=f"Collecting {repo_name}",
        show_progress=show_progress
    )
    
    # Setup todos
    todos = [("github", "Collect GitHub information")]
    if pkt_type == "debian":
        todos.append(("dependencies", "Collect dependencies"))
    
    progress_tracker.add_todos(todos)
    
    try:
        # Collect GitHub repo info
        with ProgressContext(progress_tracker, "github"):
            repo_info = await collect_github_repo_info(
                repo_name=repo_name,
                pkt_type=pkt_type,
                pkt_name=pkt_name,
                settings=settings,
                show_progress=False  # Don't show sub-progress
            )
        
        # Enrich with dependencies if Debian package
        if pkt_type == "debian":
            with ProgressContext(progress_tracker, "dependencies"):
                repo_info = await enrich_repo_with_dependencies(
                    repo_info, 
                    settings=settings,
                    show_progress=False  # Don't show sub-progress
                )
        
        progress_tracker.print_summary()
        return repo_info
        
    except Exception as e:
        progress_tracker.print_summary()
        raise
