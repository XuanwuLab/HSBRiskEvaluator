import os
import time
from datetime import timedelta
import asyncio
import yaml
from pathlib import Path
from typing import Dict
import logging

from hsbriskevaluator.collector.github_collector import CollectorSettings
from hsbriskevaluator.utils.file import get_data_dir
from hsbriskevaluator.collector import collect_all

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Utility functions
def load_yaml(file_path: Path) -> Dict:
    """Load a YAML file into a Python dictionary."""
    if not file_path.exists():
        logger.error(f"File {file_path} does not exist.")
        exit(1)
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        exit(1)


def save_yaml(data: Dict, file_path: Path):
    """Save a Python dictionary to a YAML file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(file_path, "w") as f:
            yaml.dump(data, f, allow_unicode=True)
        logger.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving YAML file {file_path}: {e}")


def get_repo_name(git_url: str) -> str:
    """
    Extract the repository name from a Git URL.

    Args:
        git_url (str): The Git URL of the repository.

    Returns:
        str: The repository name in the format 'owner/repo'.
    """
    try:
        repo_name = git_url.split("github.com/")[-1]
        return repo_name.rstrip(".git") if repo_name.endswith(".git") else repo_name
    except Exception as e:
        logger.error(f"Failed to extract repository name from URL '{git_url}': {e}")
        raise


async def collect_one(package_info: Dict, semaphore: asyncio.Semaphore):
    """
    Collect repository information for a single package file and update it if needed.

    Args:
        package_info (Dict): Information about the package.
        semaphore (asyncio.Semaphore): A semaphore to limit concurrency.
    """

    github_tokens = []
    with open(Path(__file__).parent.parent / ".github_tokens", "r") as f:
        github_tokens = [line.strip() for line in f if line.strip()]
    settings = CollectorSettings(
        github_tokens=github_tokens,
    )
    logger.info(f"github_tokens: {len(github_tokens)} tokens loaded.")
    # Create tasks for each package
    async with semaphore:  # Ensure concurrency limit
        try:
            # Validate package information
            package_name = package_info.get("parent_package") or package_info.get(
                "package"
            )
            if not package_name:
                raise ValueError(
                    "Missing 'package' or 'parent_package' in package info."
                )

            git_url = package_info.get("upstream_git_url")
            if not git_url:
                raise ValueError(
                    f"Missing 'upstream_git_url' for package: {package_name}"
                )

            # Skip non-GitHub URLs
            if not git_url.startswith("https://github"):
                logger.info(
                    f"Skipping package {package_name} due to non-GitHub git URL: {git_url}"
                )
                return

            # Define output path for repository info
            repo_info_path = get_data_dir() / "repo_info" / f"{package_name}.yaml"
            if repo_info_path.exists():
                logger.info(
                    f"Repository info for {package_name} already exists, skipping."
                )
                return

            if not repo_info_path.parent.exists():
                repo_info_path.parent.mkdir(parents=True, exist_ok=True)
            # Collect repository information
            repo_info = await collect_all(
                pkt_type="debian",
                pkt_name=package_info["package"],
                repo_name=get_repo_name(git_url),
                settings=settings,
            )

            # Save the collected repository information
            save_yaml(repo_info.model_dump(), repo_info_path)

        except Exception as e:
            # Log errors for individual packages
            logger.error(f"Error processing package {package_name}: {e}")
            raise e


async def collect_repo_info(package_file: Path, max_concurrency: int = 5):
    """
    Collect repository information for multiple packages and update YAML files.

    Args:
        package_file (Path): Path to the YAML file containing package information.
        max_concurrency (int): Maximum number of concurrent tasks allowed.
    """
    package_dict_by_name = load_yaml(package_file)
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        collect_one(package_info, semaphore)
        for package_name, package_info in package_dict_by_name.items()
    ]

    # Run tasks concurrently while logging exceptions
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")


# Main entry point
if __name__ == "__main__":
    package_file = get_data_dir() / "packages.yaml"
    dependency_file = get_data_dir() / "dependencies.yaml"

    for file in [package_file, dependency_file]:
        if not file.exists():
            logger.error(f"File {file} does not exist.")
        else:
            asyncio.run(collect_repo_info(file, 5))
