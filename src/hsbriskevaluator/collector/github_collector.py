"""
GitHub repository information collector using the GitHub API.
This module provides functionality to collect repository information
and convert it to the Pydantic models defined in repo_info.py.
"""

import asyncio
import logging
from typing import List, Optional
from ..git import GitHubCrawler
from .repo_info import RepoInfo, GithubUser, Issue, PullRequest, Comment

logger = logging.getLogger(__name__)


class GitHubRepoCollector:
    """Collector for GitHub repository information using the GitHub API"""

    def __init__(self):
        self.crawler: Optional[GitHubCrawler] = None

    async def __aenter__(self):
        self.crawler = GitHubCrawler()
        await self.crawler.__aenter__()
        return self

    async def __aexit__(self, *args):
        if self.crawler:
            await self.crawler.__aexit__(*args)

    def _convert_to_github_user(self, user_data: dict) -> GithubUser:
        """Convert raw user data to GithubUser model"""
        return GithubUser(
            username=user_data.get("username", ""),
            name=user_data.get("name", ""),
            email=user_data.get("email", "")
        )

    def _convert_to_comment(self, comment_data: dict) -> Comment:
        """Convert raw comment data to Comment model"""
        return Comment(
            username=comment_data.get("username", ""),
            content=comment_data.get("content", "")
        )

    def _convert_to_issue(self, issue_data: dict) -> Issue:
        """Convert raw issue data to Issue model"""
        comments = [
            self._convert_to_comment(comment) 
            for comment in issue_data.get("comments", [])
        ]
        
        return Issue(
            number=issue_data.get("number", 0),
            author=issue_data.get("author", ""),
            title=issue_data.get("title", ""),
            body=issue_data.get("body", ""),
            comments=comments,
            status=issue_data.get("status", "open"),
            url=issue_data.get("url", ""),
            created_at=issue_data.get("created_at", ""),
            updated_at=issue_data.get("updated_at", "")
        )

    def _convert_to_pull_request(self, pr_data: dict) -> PullRequest:
        """Convert raw pull request data to PullRequest model"""
        return PullRequest(
            number=pr_data.get("number", 0),
            title=pr_data.get("title", ""),
            author=pr_data.get("author", ""),
            approver=pr_data.get("approver", ""),
            merger=pr_data.get("merger", ""),
            status=pr_data.get("status", "open"),
            url=pr_data.get("url", ""),
            created_at=pr_data.get("created_at", ""),
            updated_at=pr_data.get("updated_at", ""),
            merged_at=pr_data.get("merged_at")
        )

    async def collect_repo_info(
        self, 
        repo_name: str, 
        pkt_name: str = "",
        max_contributors: int = 100,
        max_issues: int = 100,
        max_prs: int = 100
    ) -> RepoInfo:
        """
        Collect complete repository information and return as RepoInfo model
        
        Args:
            repo_name: Repository name in format 'owner/repo'
            pkt_name: Package name (optional, defaults to repo name)
            max_contributors: Maximum number of contributors to fetch
            max_issues: Maximum number of issues to fetch
            max_prs: Maximum number of pull requests to fetch
            
        Returns:
            RepoInfo: Complete repository information
        """
        if not self.crawler:
            raise RuntimeError("Collector not initialized. Use async context manager.")

        logger.info(f"Starting collection for repository: {repo_name}")

        try:
            # Get complete repo info from the crawler
            raw_repo_info = await self.crawler.get_repo_info_complete(repo_name, pkt_name)
            
            # Convert contributors to GithubUser models
            contributors = [
                self._convert_to_github_user(contributor)
                for contributor in raw_repo_info.get("contributor_list", [])[:max_contributors]
            ]

            # Convert issues to Issue models
            issues = [
                self._convert_to_issue(issue)
                for issue in raw_repo_info.get("issue_list", [])[:max_issues]
            ]

            # Convert pull requests to PullRequest models
            pull_requests = [
                self._convert_to_pull_request(pr)
                for pr in raw_repo_info.get("pr_list", [])[:max_prs]
            ]

            # Create RepoInfo model
            repo_info = RepoInfo(
                pkt_type="debian",
                pkt_name=raw_repo_info.get("pkt_name", pkt_name or repo_name.split("/")[-1]),
                repo_id=raw_repo_info.get("repo_id", repo_name.replace("/", "-")),
                url=raw_repo_info.get("url", ""),
                contributor_list=contributors,
                pr_list=pull_requests,
                issue_list=issues,
                binary_file_list=raw_repo_info.get("binary_file_list", [])
            )

            logger.info(f"Successfully collected repo info for {repo_name}")
            logger.info(f"  - Contributors: {len(contributors)}")
            logger.info(f"  - Issues: {len(issues)}")
            logger.info(f"  - Pull Requests: {len(pull_requests)}")
            logger.info(f"  - Binary Files: {len(repo_info.binary_file_list)}")

            return repo_info

        except Exception as e:
            logger.error(f"Error collecting repo info for {repo_name}: {e}")
            raise

    async def collect_multiple_repos(
        self, 
        repo_names: List[str],
        max_concurrent: int = 3,
        **kwargs
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
        if not self.crawler:
            raise RuntimeError("Collector not initialized. Use async context manager.")

        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def collect_single_repo(repo_name: str) -> Optional[RepoInfo]:
            async with semaphore:
                try:
                    return await self.collect_repo_info(repo_name, **kwargs)
                except Exception as e:
                    logger.error(f"Failed to collect info for {repo_name}: {e}")
                    return None

        logger.info(f"Starting collection for {len(repo_names)} repositories")
        
        tasks = [collect_single_repo(repo_name) for repo_name in repo_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        repo_infos = [
            result for result in results 
            if isinstance(result, RepoInfo)
        ]
        
        logger.info(f"Successfully collected info for {len(repo_infos)}/{len(repo_names)} repositories")
        return repo_infos


# Convenience function for single repository collection
async def collect_github_repo_info(
    repo_name: str, 
    pkt_name: str = "",
    **kwargs
) -> RepoInfo:
    """
    Convenience function to collect information for a single repository
    
    Args:
        repo_name: Repository name in format 'owner/repo'
        pkt_name: Package name (optional)
        **kwargs: Additional arguments
        
    Returns:
        RepoInfo: Repository information
    """
    async with GitHubRepoCollector() as collector:
        return await collector.collect_repo_info(repo_name, pkt_name, **kwargs)


# Convenience function for multiple repositories collection
async def collect_multiple_github_repos(
    repo_names: List[str],
    **kwargs
) -> List[RepoInfo]:
    """
    Convenience function to collect information for multiple repositories
    
    Args:
        repo_names: List of repository names
        **kwargs: Additional arguments
        
    Returns:
        List[RepoInfo]: List of repository information
    """
    async with GitHubRepoCollector() as collector:
        return await collector.collect_multiple_repos(repo_names, **kwargs)
