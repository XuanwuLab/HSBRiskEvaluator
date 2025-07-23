import logging
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from hsbriskevaluator.collector import GitHubRepoCollector
from hsbriskevaluator.utils.apt_utils import AptUtils, PackageInfo
from hsbriskevaluator.utils.file import get_data_dir
from hsbriskevaluator.utils.upstream import get_upstream_info_by_llm
from pathlib import Path
import yaml
from tqdm.asyncio import tqdm as async_tqdm
from typing import Optional, Dict
import aiohttp
from hsbriskevaluator.utils.llm import get_instructor_client
import asyncio

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
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
        with open(file_path, "w") as f:
            yaml.dump(data, f, allow_unicode=True)
        logger.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving YAML file {file_path}: {e}")


async def check_url(url: str, session: aiohttp.ClientSession) -> bool:
    """Check if a URL is reachable using aiohttp."""
    try:
        async with session.head(url, allow_redirects=True) as response:
            return response.status == 200
    except aiohttp.ClientError as e:
        logger.error(f"Error checking URL {url}: {e}")
        return False


# Main processing function
async def update_package_info(package_file: Path, max_concurrency: int = 10):
    """
    Update package information with upstream details and save the results.
    """
    # Load package and dependency data
    package_dict_by_name = load_yaml(package_file)

    # Semaphore for limiting concurrency
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_package(
        package_name: str, package_info: Dict, session: aiohttp.ClientSession
    ):
        """Process a single package."""
        async with semaphore:
            # Skip packages that already have upstream info
            upstream_url = package_info.get("upstream_git_url")
            if upstream_url:
                is_valid_url = await check_url(upstream_url, session)
                if is_valid_url:
                    logger.info(
                        f"Package {package_name} already has valid upstream info, skipping."
                    )
                    return

            # Fetch upstream information
            try:
                response = await get_upstream_info_by_llm(package_name)
                if response:
                    package_info["upstream_git_url"] = response.upstream_git_url
                    package_info["upstream_type"] = response.upstream_type
                    package_info["git_url"] = response.debian_downstream_git_url
                    package_info[
                        "parent_debian_package"
                    ] = response.parent_debian_package
                    logger.info(
                        f"Updated package {package_name} with upstream info: {response}"
                    )
            except Exception as e:
                logger.error(f"Error updating package {package_name}: {e}")

    # Process all packages asynchronously
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_package(package_name, package_info, session)
            for package_name, package_info in package_dict_by_name.items()
        ]
        await async_tqdm.gather(*tasks, desc="Analyzing packages")

    # Save updated package data
    save_yaml(package_dict_by_name, package_file)


# Main entry point
if __name__ == "__main__":
    package_file = get_data_dir() / "packages.yaml"
    dependency_file = get_data_dir() / "dependencies.yaml"

    for file in [package_file, dependency_file]:
        if not file.exists():
            logger.error(f"File {file} does not exist.")
        else:
            asyncio.run(update_package_info(file))
