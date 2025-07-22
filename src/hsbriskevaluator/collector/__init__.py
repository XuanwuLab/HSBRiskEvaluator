"""
Information gathering module for HSBRiskEvaluator.

This module provides functionality to collect repository information
from various sources, particularly GitHub repositories, and package
dependency information from APT.
"""

from typing import Optional
from datetime import timedelta
from .repo_info import Comment, Issue, PullRequest, RepoInfo, Dependent, User

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
    time_window: Optional[timedelta] = None,
):
    """Collect all information about a repository"""
    repo_info = await collect_github_repo_info(
        repo_name=repo_name,
        pkt_type=pkt_type,
        pkt_name=pkt_name,
        time_window=time_window
    )
    if pkt_type == "debian":
        repo_info = await enrich_repo_with_dependencies(repo_info)
    return repo_info
