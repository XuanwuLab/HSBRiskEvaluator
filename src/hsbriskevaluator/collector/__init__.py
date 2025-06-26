"""
Information gathering module for HSBRiskEvaluator.

This module provides functionality to collect repository information
from various sources, particularly GitHub repositories.
"""

from .repo_info import (
    GithubUser,
    Comment,
    Issue,
    PullRequest,
    RepoInfo
)

from .github_collector import (
    GitHubRepoCollector,
    collect_github_repo_info,
    collect_multiple_github_repos
)

__all__ = [
    # Pydantic models
    "GithubUser",
    "Comment", 
    "Issue",
    "PullRequest",
    "RepoInfo",
    
    # GitHub collector
    "GitHubRepoCollector",
    "collect_github_repo_info",
    "collect_multiple_github_repos"
]
