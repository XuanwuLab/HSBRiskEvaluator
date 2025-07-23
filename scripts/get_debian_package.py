import logging
from dotenv import load_dotenv
from hsbriskevaluator.collector import GitHubRepoCollector
from hsbriskevaluator.utils.apt_utils import AptUtils
from hsbriskevaluator.utils.file import get_data_dir
from pathlib import Path
import yaml
from tqdm import tqdm
from typing import Set, Dict, List, Union

# Initialize environment variables and logging
load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize utility classes
github_collector = GitHubRepoCollector()
apt_utils = AptUtils()


def save_to_yaml(data: Union[Dict, List], file_path: Path):
    """Save data to a YAML file."""
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    logger.info(f"Saved data to {file_path}")


def process_packages(priority: str) -> Dict:
    """
    Process packages of a given priority:
    - Fetch package information.
    - Return a dictionary of package details.
    """
    logger.info(f"Processing {priority} packages...")
    packages = apt_utils.get_package_by_priority(priority)  # type:ignore
    return {pkg.package: pkg.model_dump() for pkg in packages}


def process_dependencies(priority: str, package_dict: Dict) -> Dict:
    """
    Process dependencies for a given priority:
    - Extract dependency information from the package dictionary.
    - Fetch details of dependency packages.
    - Return a dictionary of dependency details.
    """
    logger.info(f"Processing {priority} package dependencies...")

    # Collect dependency names
    dependency_names: Set[str] = set()
    for package_name, package_info in tqdm(package_dict.items(), desc=f"Analyzing {priority} dependencies"):
        if "depends" in package_info:
            for dep in package_info["depends"]:
                dependency_names.add(dep["name"])

    # Fetch details for each dependency
    dependency_packages = []
    for dep_name in tqdm(dependency_names, desc=f"Fetching {priority} dependencies"):
        dep_package = apt_utils.get_package_info(dep_name)
        if dep_package:
            dependency_packages.append(dep_package.model_dump())

    return {pkg["package"]: pkg for pkg in dependency_packages}


def main():
    """Main function: Process packages and dependencies for all priorities."""
    priorities = ["required", "important", "standard"]

    # Process all packages and save them
    all_packages = {}
    all_package_names = []
    for priority in priorities:
        package_dict = process_packages(priority)
        all_packages.update(package_dict)
        all_package_names.extend(package_dict.keys())
    save_to_yaml(all_packages, get_data_dir() / "packages.yaml")
    save_to_yaml(all_package_names, get_data_dir() / "package_names.yaml")

    # Process all dependencies and save them
    all_dependencies = {}
    all_dependency_names = []
    for priority in priorities:
        current_dependencies = process_dependencies(priority, all_packages[priority])
        for dep_name, dep_info in current_dependencies.items():
            if dep_name not in all_dependencies:
                dep_info["dependent_priorities"] = [priority]
                all_dependency_names.append(dep_name)
                all_dependencies[dep_name] = dep_info
            else:
                all_dependencies[dep_name]["dependent_priorities"].append(priority)

    save_to_yaml(all_dependencies, get_data_dir() / "dependencies.yaml")
    save_to_yaml(all_dependency_names, get_data_dir() / "dependency_names.yaml")


if __name__ == "__main__":
    main()