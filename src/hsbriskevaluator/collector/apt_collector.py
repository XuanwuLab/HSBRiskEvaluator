"""
Dependency collector for gathering package dependencies using APT utilities.
This module provides functionality to collect recursive dependencies for Debian packages
and populate the dependent_list field in RepoInfo models.
"""

import asyncio
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from hsbriskevaluator.collector.repo_info import Dependent, RepoInfo
from hsbriskevaluator.collector.settings import CollectorSettings
from hsbriskevaluator.utils.apt_utils import AptUtils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class APTCollector:
    """Collector for package dependencies using APT utilities"""

    def __init__(self, max_concurrency: Optional[int] = None, settings: Optional[CollectorSettings] = None):
        """
        Initialize the dependency collector

        Args:
            max_concurrency: Maximum number of concurrent dependency collection operations
            settings: Collector settings (if not provided, defaults will be used)
        """
        if settings is None:
            settings = CollectorSettings()
        self.settings = settings
        self.max_concurrency = max_concurrency or settings.apt_max_concurrency
        self.apt_utils = AptUtils(max_concurrency=self.max_concurrency)
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrency)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.executor.shutdown(wait=True)

    async def _run_in_executor(self, func, *args):
        """Run blocking APT operations in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def collect_dependencies(self, package_name: str) -> List[Dependent]:
        """
        Collect recursive dependencies for a package

        Args:
            package_name: The Debian package name

        Returns:
            List[Dependent]: List of recursive dependencies

        Raises:
            RuntimeError: If dependency collection fails
        """
        logger.info(
            f"Starting dependency collection for package: {package_name}")

        try:
            dependencies = await self._run_in_executor(
                self.apt_utils.get_recursive_dependencies, package_name
            )

            logger.info(
                f"Successfully collected {len(dependencies)} dependencies for {package_name}"
            )
            return dependencies

        except Exception as e:
            logger.error(
                f"Failed to collect dependencies for {package_name}: {str(e)}")
            raise RuntimeError(
                f"Dependency collection failed for {package_name}"
            ) from e

    async def collect_multiple_dependencies(
        self, package_names: List[str]
    ) -> List[List[Dependent]]:
        """
        Collect dependencies for multiple packages concurrently

        Args:
            package_names: List of package names

        Returns:
            List[List[Dependent]]: List of dependency lists for each package
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def collect_single_package(package_name: str) -> List[Dependent]:
            async with semaphore:
                try:
                    return await self.collect_dependencies(package_name)
                except Exception as e:
                    logger.error(
                        f"Failed to collect dependencies for {package_name}: {e}"
                    )
                    return []

        logger.info(
            f"Starting dependency collection for {len(package_names)} packages")

        tasks = [collect_single_package(pkg) for pkg in package_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        dependency_lists = [
            result for result in results if isinstance(result, list)]

        logger.info(
            f"Successfully collected dependencies for {len(dependency_lists)}/{len(package_names)} packages"
        )
        return dependency_lists

    async def enrich_repo_info_with_dependencies(self, repo_info: RepoInfo) -> RepoInfo:
        """
        Enrich a RepoInfo object with dependency information

        Args:
            repo_info: RepoInfo object to enrich

        Returns:
            RepoInfo: Updated RepoInfo object with dependencies
        """
        if not repo_info.pkt_name:
            logger.warning(
                f"No package name provided for repo {repo_info.repo_id}, skipping dependency collection"
            )
            return repo_info

        try:
            dependencies = await self.collect_dependencies(repo_info.pkt_name)

            # Create a new RepoInfo object with updated dependencies
            updated_repo_info = repo_info.model_copy()
            updated_repo_info.dependent_list = dependencies

            logger.info(
                f"Successfully enriched {repo_info.repo_id} with {len(dependencies)} dependencies"
            )
            return updated_repo_info

        except Exception as e:
            logger.error(
                f"Failed to enrich repo {repo_info.repo_id} with dependencies: {str(e)}"
            )
            # Return original repo_info if dependency collection fails
            return repo_info

    async def enrich_multiple_repo_infos(
        self, repo_infos: List[RepoInfo]
    ) -> List[RepoInfo]:
        """
        Enrich multiple RepoInfo objects with dependency information concurrently

        Args:
            repo_infos: List of RepoInfo objects to enrich

        Returns:
            List[RepoInfo]: List of updated RepoInfo objects with dependencies
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def enrich_single_repo(repo_info: RepoInfo) -> RepoInfo:
            async with semaphore:
                return await self.enrich_repo_info_with_dependencies(repo_info)

        logger.info(
            f"Starting dependency enrichment for {len(repo_infos)} repositories"
        )

        tasks = [enrich_single_repo(repo) for repo in repo_infos]
        enriched_repos = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        valid_repos = [
            result for result in enriched_repos if isinstance(result, RepoInfo)
        ]

        logger.info(
            f"Successfully enriched {len(valid_repos)}/{len(repo_infos)} repositories with dependencies"
        )
        return valid_repos

    def get_package_classification(self, package_name: str) -> tuple[bool, bool]:
        """
        Get package classification (is_os_default, is_mainstream_os)

        Args:
            package_name: The package name to classify

        Returns:
            tuple[bool, bool]: (is_os_default, is_mainstream_os)
        """
        return self.apt_utils.get_package_classification(package_name)

    def is_os_default_package(self, package_name: str) -> bool:
        """
        Check if package is an OS default package

        Args:
            package_name: The package name to check

        Returns:
            bool: True if package is OS default
        """
        return self.apt_utils.is_os_default_package(package_name)

    def is_mainstream_os_package(self, package_name: str) -> bool:
        """
        Check if package is a mainstream OS package

        Args:
            package_name: The package name to check

        Returns:
            bool: True if package is mainstream OS
        """
        return self.apt_utils.is_mainstream_os_package(package_name)


# Convenience function for single package dependency collection
async def collect_package_dependencies(
    package_name: str, max_concurrency: Optional[int] = None, settings: Optional[CollectorSettings] = None
) -> List[Dependent]:
    """
    Convenience function to collect dependencies for a single package

    Args:
        package_name: The Debian package name
        max_concurrency: Maximum concurrency for operations
        settings: Collector settings

    Returns:
        List[Dependent]: List of recursive dependencies
    """
    async with APTCollector(max_concurrency, settings) as collector:
        return await collector.collect_dependencies(package_name)


# Convenience function for enriching RepoInfo with dependencies
async def enrich_repo_with_dependencies(
    repo_info: RepoInfo, max_concurrency: Optional[int] = None, settings: Optional[CollectorSettings] = None
) -> RepoInfo:
    """
    Convenience function to enrich a RepoInfo object with dependencies

    Args:
        repo_info: RepoInfo object to enrich
        max_concurrency: Maximum concurrency for operations
        settings: Collector settings

    Returns:
        RepoInfo: Updated RepoInfo object with dependencies
    """
    async with APTCollector(max_concurrency, settings) as collector:
        return await collector.enrich_repo_info_with_dependencies(repo_info)


# Convenience function for enriching multiple RepoInfos with dependencies
async def enrich_multiple_repos_with_dependencies(
    repo_infos: List[RepoInfo], max_concurrency: Optional[int] = None, settings: Optional[CollectorSettings] = None
) -> List[RepoInfo]:
    """
    Convenience function to enrich multiple RepoInfo objects with dependencies

    Args:
        repo_infos: List of RepoInfo objects to enrich
        max_concurrency: Maximum concurrency for operations
        settings: Collector settings

    Returns:
        List[RepoInfo]: List of updated RepoInfo objects with dependencies
    """
    async with APTCollector(max_concurrency, settings) as collector:
        return await collector.enrich_multiple_repo_infos(repo_infos)
