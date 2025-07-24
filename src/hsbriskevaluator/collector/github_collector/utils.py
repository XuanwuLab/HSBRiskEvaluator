"""
Utility functions for GitHub collector operations.
"""

import asyncio
import logging
import subprocess
from typing import List, Optional
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from git import Repo, GitCommandError

from ..repo_info import Commit, User
from ...utils.file import get_data_dir, is_binary

logger = logging.getLogger(__name__)


class LocalRepoUtils:
    """Utility class for local repository operations"""

    @staticmethod
    async def _run_in_executor(executor: ThreadPoolExecutor, func, *args):
        """Run blocking operations in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, *args)

    async def clone_repository(
        self, repo_name: str, repo_url: str, executor: ThreadPoolExecutor
    ) -> str:
        """
        Clone repository to local data directory

        Args:
            repo_name: Repository name in format 'owner/repo'
            repo_url: Repository clone URL
            executor: ThreadPoolExecutor for async operations

        Returns:
            str: Relative path to cloned repository from data_dir, or None if failed
        """

        def _clone_repo():
            try:
                data_dir = get_data_dir()
                repo_dir_name = repo_name.replace("/", "-")
                local_repo_path = data_dir / repo_dir_name

                # Remove existing directory if it exists
                if local_repo_path.exists():
                    import shutil

                    shutil.rmtree(local_repo_path)

                # Clone the repository
                cmd = ["git", "clone", repo_url, str(local_repo_path)]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
                )

                if result.returncode == 0:
                    logger.info(f"Successfully cloned {repo_name} to {local_repo_path}")
                    return repo_dir_name  # Return relative path
                else:
                    logger.error(f"Failed to clone {repo_name}: {result.stderr}")
                    return None

            except subprocess.TimeoutExpired:
                logger.error(f"Timeout cloning repository {repo_name}")
                raise TimeoutError(
                    f"Cloning {repo_name} took too long and was aborted."
                )
            except Exception as e:
                logger.error(f"Error cloning repository {repo_name}: {str(e)}")
                raise e

        return await self._run_in_executor(executor, _clone_repo)

    async def find_binary_files_local(
        self, local_repo_path: Path, executor: Optional[ThreadPoolExecutor] = None
    ) -> List[str]:
        """
        Find binary files in the locally cloned repository using the is_binary function
        This method scans all files and uses content-based detection rather than extension-based

        Args:
            local_repo_path: Path to the local repository
            executor: ThreadPoolExecutor for async operations

        Returns:
            List[str]: List of binary file paths relative to repository root
        """

        def _scan_local_files():
            binary_files = []
            try:
                # Scan all files in the repository
                for file_path in local_repo_path.rglob("*"):
                    if file_path.is_file():
                        # Skip .git directory files
                        if ".git" in file_path.parts:
                            continue

                        # Check if file is binary using the is_binary function
                        # This function includes is_dir check and content-based binary detection
                        if is_binary(str(file_path)):
                            # Get relative path from repository root
                            relative_path = file_path.relative_to(local_repo_path)
                            binary_files.append(str(relative_path))

                logger.info(
                    f"Found {len(binary_files)} binary files in local repository using content-based detection"
                )
                return binary_files
            except Exception as e:
                logger.error(
                    f"Error scanning local repository for binary files: {str(e)}"
                )
                return []

        if executor:
            return await self._run_in_executor(executor, _scan_local_files)
        else:
            # Run synchronously if no executor provided
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _scan_local_files)

    async def get_local_commits(
        self,
        local_repo_path: Path,
        since_timestamp: Optional[datetime] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> List[Commit]:
        """
        Get commits from a local Git repository and map them to the Commit class.

        Args:
            local_repo_path: Path to the local Git repository.
            since_timestamp: Fetch commits only after this timestamp.
            executor: ThreadPoolExecutor for async operations

        Returns:
            List of Commit objects.
        """

        def _fetch_local_commits():
            """Fetch commits from the local Git repository."""
            commits = []
            try:
                # Open the local repository
                repo = Repo(local_repo_path)

                # Check if the repository is valid
                if repo.bare:
                    logger.error(
                        f"The repository at {local_repo_path} is not a valid Git repository."
                    )
                    return []

                # Iterate through commits
                for commit in repo.iter_commits():
                    # Filter commits by timestamp if specified
                    commit_date = datetime.fromtimestamp(commit.committed_date)
                    if since_timestamp and commit_date < since_timestamp:
                        break

                    # Extract pull request numbers from the commit message
                    pull_numbers = []

                    # Append commit information mapped to the Commit class
                    username = "" if not commit.author else commit.author.name or ""
                    email = "" if not commit.author else commit.author.email or ""
                    if type(commit.message) == str:
                        message = commit.message
                    elif type(commit.message) == bytes:
                        message = commit.message.decode()
                    else:
                        message = ""

                    commits.append(
                        Commit(
                            hash=commit.hexsha,
                            author=User(username=username, email=email, type=""),
                            message=message,
                            timestamp=commit_date.isoformat(),
                            pull_numbers=pull_numbers,
                        )
                    )

                logger.info(
                    f"Retrieved {len(commits)} commits from local repo: {local_repo_path}"
                )
                return commits
            except GitCommandError as e:
                logger.error(f"Error fetching commits from local repo: {e}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return []

        if executor:
            return await self._run_in_executor(executor, _fetch_local_commits)
        else:
            # Run synchronously if no executor provided
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _fetch_local_commits)
