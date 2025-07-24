import os
import time
import yaml
from pathlib import Path
from typing import Dict
import logging
import asyncio
from multiprocessing import Pool, Manager, current_process
from tqdm import tqdm

from hsbriskevaluator.collector.github_collector import CollectorSettings
from hsbriskevaluator.utils.file import get_data_dir
from hsbriskevaluator.collector import collect_all
from hsbriskevaluator.utils.progress_manager import get_progress_manager

def setup_process_logging():
    """Setup logging for each child process"""
    # Use progress manager for consistent logging
    progress_manager = get_progress_manager()
    return progress_manager.logger

# Configure main process logging through progress manager
progress_manager = get_progress_manager()
logger = progress_manager.logger

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


def collect_one(package_info: Dict, github_tokens: list, progress_counter):
    """
    Collect repository information for a single package file and update it if needed.

    Args:
        package_info (Dict): Information about the package.
        github_tokens (list): List of GitHub tokens for API access.
        progress_counter: Shared counter for tracking progress
    """
    # Setup process-specific logging and progress manager
    progress_manager = get_progress_manager()
    process_logger = progress_manager.logger
    
    package_name = None
    try:
        # Validate package information
        package_name = package_info.get("parent_package") or package_info.get("package")
        if not package_name:
            raise ValueError("Missing 'package' or 'parent_package' in package info.")
        
        settings = CollectorSettings(github_tokens=github_tokens)
        process_logger.info(f"Processing package {package_name}")
        
        git_url = package_info.get("upstream_git_url")
        if not git_url:
            raise ValueError(f"Missing 'upstream_git_url' for package: {package_name}")

        # Skip non-GitHub URLs
        if not git_url.startswith("https://github"):
            process_logger.info(f"Skipping {package_name} - non-GitHub URL")
            with progress_counter.get_lock():
                progress_counter.value += 1
            return {"status": "skipped", "package": package_name}

        # Define output path for repository info
        repo_info_path = get_data_dir() / "repo_info" / f"{package_name}.yaml"
        if repo_info_path.exists():
            process_logger.info(f"Skipping {package_name} - already exists")
            with progress_counter.get_lock():
                progress_counter.value += 1
            return {"status": "skipped", "package": package_name}

        if not repo_info_path.parent.exists():
            repo_info_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect repository information
        process_logger.info(f"Collecting data for {package_name}")
        repo_info = asyncio.run(collect_all(
            pkt_type="debian",
            pkt_name=package_info["package"],
            repo_name=get_repo_name(git_url),
            settings=settings,
        ))

        # Save the collected repository information
        save_yaml(repo_info.model_dump(), repo_info_path)
        process_logger.info(f"Successfully processed {package_name}")
        
        with progress_counter.get_lock():
            progress_counter.value += 1
        
        return {"status": "completed", "package": package_name}

    except Exception as e:
        error_msg = f"Error processing {package_name or 'unknown'}: {e}"
        process_logger.error(error_msg)
        
        with progress_counter.get_lock():
            progress_counter.value += 1
            
        return {"status": "error", "package": package_name, "error": str(e)}


def collect_repo_info(package_file: Path, max_concurrency: int = 5):
    """
    Collect repository information for multiple packages using multiprocessing.

    Args:
        package_file (Path): Path to the YAML file containing package information.
        max_concurrency (int): Maximum number of concurrent processes allowed.
    """
    package_dict_by_name = load_yaml(package_file)
    total_packages = len(package_dict_by_name)
    
    progress_manager.print_status(f"Starting to process {total_packages} packages from {package_file}")

    # Read GitHub tokens
    github_tokens = []
    with open(Path(__file__).parent.parent / ".github_tokens", "r") as f:
        github_tokens = [line.strip() for line in f if line.strip()]
    
    progress_manager.print_status(f"Loaded {len(github_tokens)} GitHub tokens")
    progress_manager.print_status(f"Starting multiprocessing with {max_concurrency} concurrent processes")
    
    start_time = time.time()
    
    # Use multiprocessing with a simple shared counter for progress
    with Manager() as manager:
        progress_counter = manager.Value('i', 0)
        
        # Prepare arguments for multiprocessing
        package_list = [
            (package_info, github_tokens, progress_counter)
            for package_name, package_info in package_dict_by_name.items()
        ]

        # Use multiprocessing Pool with progress monitoring via progress manager
        with Pool(processes=max_concurrency) as pool:
            # Start the multiprocessing jobs
            async_result = pool.starmap_async(collect_one, package_list)
            
            # Monitor progress with progress manager
            with progress_manager.create_main_progress(total_packages, "Processing packages", "pkg") as main_pbar:
                while not async_result.ready():
                    current_progress = progress_counter.value
                    main_pbar.n = current_progress
                    main_pbar.refresh()
                    time.sleep(0.5)
                
                # Get final results
                results = async_result.get()
                main_pbar.n = total_packages
                main_pbar.refresh()
        
        # Process results and count statistics
        stats = {"completed": 0, "skipped": 0, "error": 0}
        errors = []
        
        for result in results:
            if result and isinstance(result, dict):
                status = result.get("status", "unknown")
                stats[status] = stats.get(status, 0) + 1
                if status == "error":
                    errors.append(result)
        
        elapsed_time = time.time() - start_time
        
        progress_manager.print_status(f"Processing completed in {elapsed_time:.2f} seconds")
        progress_manager.print_status(f"Statistics: {dict(stats)}")
        
        if errors:
            progress_manager.print_status(f"Found {len(errors)} errors during processing (see log file for details)")
            logger.warning(f"Found {len(errors)} errors during processing:")
            for error in errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error.get('package', 'unknown')}: {error.get('error', 'unknown error')}")
            if len(errors) > 5:
                logger.warning(f"  ... and {len(errors) - 5} more errors")
        
        progress_manager.print_status("All tasks completed!")
        progress_manager.print_status(f"📋 Detailed logs: {progress_manager.get_log_file_path()}")


# Main entry point
if __name__ == "__main__":
    package_file = get_data_dir() / "packages.yaml"
    dependency_file = get_data_dir() / "dependencies.yaml"

    for file in [package_file, dependency_file]:
        if not file.exists():
            logger.error(f"File {file} does not exist.")
        else:
            collect_repo_info(file, max_concurrency=5)