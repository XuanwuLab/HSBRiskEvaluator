import os
import logging
from dotenv import load_dotenv
from hsbriskevaluator.collector import GitHubRepoCollector
from hsbriskevaluator.utils.diff import Comparator
from hsbriskevaluator.utils.apt_utils import AptUtils
from hsbriskevaluator.utils.file import get_data_dir
from pathlib import Path
import yaml
from tqdm import tqdm
from typing import List, Set

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize GitHub and APT utilities
github_collector = GitHubRepoCollector()
apt_utils = AptUtils()


def get_git_url(package: str) -> str:
    """
    Get the Git URL for a given package by comparing upstream repositories.

    Args:
        package (str): The package name.

    Returns:
        str: The Git URL of the package.
    """
    comparator = Comparator(github_collector, apt_utils)
    diff_result = comparator.clone_and_compare(package)

    # Return the appropriate repository URL based on the comparison result
    if diff_result.same_project:
        return diff_result.url2
    return diff_result.url1


def load_all_packages() -> List[str]:
    """
    Load and aggregate all packages from 'os_default.yaml' and 'mainstream_os.yaml'.

    Returns:
        List[str]: A list of unique package names.
    """
    evaluator_path = Path(__file__).resolve().parent.parent / "src" / "hsbriskevaluator" / "evaluator"
    os_default_yaml = evaluator_path / "os_default.yaml"
    mainstream_os_yaml = evaluator_path / "mainstream_os.yaml"

    packages = []

    # Load packages from 'os_default.yaml'
    if os_default_yaml.exists():
        with open(os_default_yaml, "r", encoding="utf-8") as f:
            os_default_data = yaml.safe_load(f)
            packages.extend(os_default_data.get("packages", []))
            packages.extend(os_default_data.get("dependencies", []))
    else:
        logger.warning(f"'{os_default_yaml}' does not exist.")

    # Load packages from 'mainstream_os.yaml'
    if mainstream_os_yaml.exists():
        with open(mainstream_os_yaml, "r", encoding="utf-8") as f:
            mainstream_os_data = yaml.safe_load(f)
            packages.extend(mainstream_os_data.get("packages", []))
            packages.extend(mainstream_os_data.get("dependencies", []))
    else:
        logger.warning(f"'{mainstream_os_yaml}' does not exist.")

    unique_packages = list(set(packages))  # Remove duplicates
    logger.info(f"Total unique packages loaded: {len(unique_packages)}")
    return unique_packages


def save_package_info(package: str, git_url: str, failed_message:str) -> None:
    """
    Save package information as a YAML file in the 'package' directory.

    Args:
        package (str): The package name.
        git_url (str): The Git URL of the package.
    """
    package_dir = get_data_dir() / "package"
    package_dir.mkdir(parents=True, exist_ok=True)

    package_file = package_dir / f"{package}.yaml"
    package_data = {
        "package": package,
        "git_url": git_url,
        "failed_message": failed_message,
        "repo_info": {}
    }
    with open(package_file, "w", encoding="utf-8") as f:
        yaml.dump(package_data, f, allow_unicode=True)
    logger.info(f"Saved package information: {package_file}")


def process_packages(packages: List[str]) -> Set[str]:
    """
    Process a list of packages, fetch their Git URLs, and save their information.

    Args:
        packages (List[str]): The list of packages to process.

    Returns:
        Set[str]: The set of failed packages for potential retries.
    """
    failed_packages = set()

    # Use tqdm to display progress
    for package in tqdm(packages, desc="Processing packages"):
        try:
            git_url = get_git_url(package)
            logger.info(f"Package: {package}, Git URL: {git_url}")
            save_package_info(package, git_url, failed_message="")
        except Exception as e:
            logger.warning(f"Failed to get Git URL for package '{package}', using APT URL: {git_url}")
            save_package_info(package, "", failed_message=str(e))
            failed_packages.add(package)

    return failed_packages


def main():
    """
    Main function to process all packages and handle retries for failed cases.
    """
        # Load all unique packages
    packages = load_all_packages()
    failed_packages = process_packages(packages)
    with open(get_data_dir() / "failed_packages.yaml", "w", encoding="utf-8") as f:
        yaml.dump(list(failed_packages), f, allow_unicode=True)


if __name__ == "__main__":
    main()