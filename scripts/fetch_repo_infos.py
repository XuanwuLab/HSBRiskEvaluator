import os
from datetime import timedelta
import asyncio
import yaml
from pathlib import Path
import logging

from hsbriskevaluator.utils.file import get_data_dir
from hsbriskevaluator.collector import collect_all

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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


async def collect_one(package_file: Path, semaphore: asyncio.Semaphore):
    """
    Collect repository information for a single package file and update it if needed.

    Args:
        package_file (Path): Path to the YAML file representing the package.
        semaphore (asyncio.Semaphore): A semaphore to limit concurrency.
    """
    async with semaphore:  # Ensure concurrency limit
        try:
            logger.info(f"Processing package file: {package_file}")

            # Read the YAML file
            with package_file.open("r", encoding="utf-8") as f:
                package_data = yaml.safe_load(f)

            # Validate the loaded YAML content
            if not isinstance(package_data, dict):
                raise ValueError(f"Invalid YAML format in file: {package_file}")

            # Ensure `git_url` exists in the YAML content
            git_url = package_data.get("git_url")
            if not git_url:
                raise ValueError(f"Missing 'git_url' in file: {package_file}")

            # If `repo_info` is missing and `git_url` meets the conditions, collect repository info
            if not package_data.get("repo_info") and git_url.startswith("https://github"):
                logger.info(f"Collecting repository info for package: {package_data['package']}")
                
                repo_info = await collect_all(
                    pkt_type="debian",
                    pkt_name=package_data["package"],
                    repo_name=get_repo_name(git_url),
                    time_window=timedelta(days=365),
                )
                package_data["repo_info"] = repo_info.model_dump()

            # Write the updated content back to the YAML file
            with package_file.open("w", encoding="utf-8") as f:
                yaml.dump(package_data, f, allow_unicode=True)
                logger.info(f"Updated repository info saved for package: {package_data['package']}")

        except Exception as e:
            # Log errors for individual files
            logger.error(f"Error processing file {package_file}: {e}")


async def collect_repo_info(max_concurrency: int = 5):
    """
    Collect repository information for multiple packages and update YAML files.

    Args:
        max_concurrency (int): Maximum number of concurrent tasks allowed.
    """
    try:
        # Get the directory where package data is stored
        package_dir = get_data_dir() / "package"
        if not package_dir.exists() or not package_dir.is_dir():
            raise FileNotFoundError(f"Package directory not found: {package_dir}")

        # Create a semaphore to limit the number of concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrency)

        # Collect repository information for each YAML file
        tasks = []
        for package_file in package_dir.glob("*.yaml"):
            tasks.append(collect_one(package_file, semaphore))

        # Run all tasks concurrently, respecting the semaphore limits
        await asyncio.gather(*tasks, return_exceptions=True)  # Handle all exceptions gracefully

    except Exception as e:
        # Catch and log errors in the main process
        logger.critical(f"Error occurred during repository information collection: {e}")


# Run the asynchronous task
if __name__ == "__main__":
    # Set the maximum concurrency level to 5 (or adjust as needed)
    try:
        asyncio.run(collect_repo_info(max_concurrency=5))
    except Exception as e:
        logger.critical(f"Unexpected error in main execution: {e}")