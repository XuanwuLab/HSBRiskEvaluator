from multiprocessing.managers import ValueProxy
import argparse
import os
import time
import yaml
from pathlib import Path
from typing import Dict
import logging
import asyncio
from multiprocessing import Pool, Manager, current_process, Lock
from tqdm import tqdm

from hsbriskevaluator.collector.github_collector import CollectorSettings
from hsbriskevaluator.utils.file import get_data_dir
from hsbriskevaluator.collector import collect_all
from hsbriskevaluator.utils.progress_manager import get_progress_manager

yaml.Dumper.ignore_aliases = lambda *args: True  # Disable all aliases
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
        with open(file_path, "w", 0o666) as f:
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

def detect_sibling(pkt_name: str, git_url: str, meta_data_lock):
    with meta_data_lock:
        meta_data_path = get_data_dir() / "meta_data.yaml"
        meta_data_by_url_path = get_data_dir() / "meta_data_by_url.yaml"
        meta_data = load_yaml(meta_data_path) 
        meta_data_by_url = load_yaml(meta_data_by_url_path)
        siblings = meta_data_by_url.get(git_url, [])
        if not siblings:
            meta_data_by_url[git_url] = []

        if pkt_name not in meta_data_by_url[git_url]:
            meta_data_by_url[git_url].append(pkt_name) 

        meta_data[pkt_name] = {
            "url": git_url,
            "sibling": meta_data_by_url[git_url]
        }
        for pkt in meta_data_by_url[git_url]:
            if pkt in meta_data and pkt_name not in meta_data[pkt]["sibling"]: 
                meta_data[pkt]["sibling"].append(pkt_name)
        save_yaml(meta_data, meta_data_path)
        save_yaml(meta_data_by_url, meta_data_by_url_path)

        for sibling in siblings:
            path = get_data_dir() / "repo_info" / f"{sibling}.yaml" 
            if path.exists():
                return path
        return None
    
def collect_one(package_info: Dict, github_tokens: list, progress_counter: ValueProxy, meta_data_lock):
    """
    Collect repository information for a single package file and update it if needed.

    Args:
        package_info (Dict): Information about the package.
        github_tokens (list): List of GitHub tokens for API access.
        progress_counter: Shared counter for tracking progress
        meta_data_lock: Multiprocessing lock for protecting metadata file operations
    """
    # Setup process-specific logging and progress manager
    progress_manager = get_progress_manager()
    process_logger = progress_manager.logger
    
    package_name = None
    try:
        # Validate package information
        package_name = package_info.get("package")
        parent_package_name = package_info.get("parent_debian_package") 
        if not package_name:
            process_logger.error("Missing 'package' or 'parent_package' in package info.")
            progress_counter.value += 1
            return {"status": "skipped", "package": package_name}
        
        settings = CollectorSettings(github_tokens=github_tokens)
        process_logger.info(f"Processing package {package_name}")
        
        git_url = package_info.get("upstream_git_url")
        if not git_url:
            process_logger.error(f"Missing 'upstream_git_url' for package: {package_name}")
            progress_counter.value += 1
            return {"status": "skipped", "package": package_name}

        # Skip non-GitHub URLs
        if not git_url.startswith("https://github"):
            process_logger.info(f"Skipping {package_name} - non-GitHub URL")
            progress_counter.value += 1
            return {"status": "skipped", "package": package_name}

        repo_info_path = get_data_dir() / "repo_info" / f"{package_name}.yaml"

        if parent_package_name: 
            parent_repo_info_path = get_data_dir() / "repo_info" / f"{parent_package_name}.yaml" 
            if parent_repo_info_path.exists() and not repo_info_path.exists():
                rel_path = os.path.relpath(
                    str(parent_repo_info_path.resolve()), 
                    str(repo_info_path.parent.resolve())   
                )
                repo_info_path.symlink_to(rel_path)
                progress_counter.value += 1
                return {"status": "skipped", "package": package_name}

            elif repo_info_path.exists() and not parent_repo_info_path.exists():
                rel_path = os.path.relpath(
                    str(repo_info_path.resolve()),
                    str(parent_repo_info_path.parent.resolve())
                )
                parent_repo_info_path.symlink_to(rel_path)
                progress_counter.value += 1
                return {"status": "skipped", "package": package_name}

        if repo_info_path.exists():
            process_logger.info(f"Skipping {package_name} - already exists")
            progress_counter.value += 1
            return {"status": "skipped", "package": package_name}

        process_logger.info("OK, we will try to find existing package through slow path")
        sibling_repo_info_path = detect_sibling(package_name, git_url, meta_data_lock)
        if sibling_repo_info_path:
            process_logger.info(f"found sibling {sibling_repo_info_path}")
            rel_path = os.path.relpath(
                    str(repo_info_path.resolve()), 
                    str(sibling_repo_info_path.parent.resolve())   
                )
            sibling_repo_info_path.symlink_to(rel_path)
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
        
        progress_counter.value += 1
        
        return {"status": "completed", "package": package_name}

    except Exception as e:
        error_msg = f"Error processing {package_name or 'unknown'}: {e}"
        process_logger.error(error_msg)
        
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
    
    # Sort packages so that for same upstream_git_url, one goes first, others go last
    unique_packages, duplicated_packages = get_unique_packages_by_git_url(package_dict_by_name)
    total_packages = len(unique_packages) + len(duplicated_packages)
    logger.info(f"Sorted packages by upstream_git_url - processing {len(unique_packages)} unique packages and {len(duplicated_packages)} end packages")
    
    # Process unique packages first
    if unique_packages:
        logger.info(f"Processing {len(unique_packages)} unique packages...")
        _process_package_batch(unique_packages, max_concurrency, "unique packages")
    
    # Process duplicated packages second  
    if duplicated_packages:
        logger.info(f"Processing {len(duplicated_packages)} duplicated packages...")
        _process_package_batch(duplicated_packages, max_concurrency, "duplicated packages")

    logger.info(f"Completed processing all {total_packages} packages")


def _process_package_batch(package_dict: Dict, max_concurrency: int, batch_name: str):
    """
    Process a batch of packages with multiprocessing.
    
    Args:
        package_dict (Dict): Dictionary of package_name -> package_info to process
        max_concurrency (int): Maximum number of concurrent processes
        batch_name (str): Name of the batch for logging
    """
    if not package_dict:
        return
        
    total_packages = len(package_dict)
    
    progress_manager.print_status(f"Starting to process {total_packages} packages for {batch_name}")

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
        meta_data_lock = manager.Lock()
        
        # Prepare arguments for multiprocessing
        package_list = [
            (package_info, github_tokens, progress_counter, meta_data_lock)
            for package_name, package_info in package_dict.items()
        ]

        # Use multiprocessing Pool with progress monitoring via progress manager
        with Pool(processes=max_concurrency) as pool:
            # Start the multiprocessing jobs
            async_result = pool.starmap_async(collect_one, package_list)
            
            # Monitor progress with progress manager
            with progress_manager.create_main_progress(total_packages, f"Processing {batch_name}", "pkg") as main_pbar:
                while not async_result.ready():
                    current_progress = progress_counter.value
                    main_pbar.n = current_progress
                    if type(main_pbar) is tqdm:
                        main_pbar.refresh()
                    time.sleep(0.5)
                
                # Get final results
                results = async_result.get()
                main_pbar.n = total_packages
                if type(main_pbar) is tqdm:
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
        
        progress_manager.print_status(f"Processing {batch_name} completed in {elapsed_time:.2f} seconds")
        progress_manager.print_status(f"Statistics for {batch_name}: {dict(stats)}")
        
        if errors:
            progress_manager.print_status(f"Found {len(errors)} errors during {batch_name} processing (see log file for details)")
            logger.warning(f"Found {len(errors)} errors during {batch_name} processing:")
            for error in errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error.get('package', 'unknown')}: {error.get('error', 'unknown error')}")
            if len(errors) > 5:
                logger.warning(f"  ... and {len(errors) - 5} more errors")
        
        progress_manager.print_status(f"{batch_name} processing completed!")
        progress_manager.print_status(f"📋 Detailed logs: {progress_manager.get_log_file_path()}")


def get_unique_packages_by_git_url(package_dict_by_name: Dict) -> tuple[Dict, Dict]:
    """
    Sort packages so that for packages with the same upstream_git_url,
    one goes to the unique and others go to the duplicated.
    
    Args:
        package_dict_by_name: Dictionary of package_name -> package_info
        
    Returns:
        Tuple[Dict, Dict]: (unique_packages, duplicated_packages) dictionaries
    """
    from collections import defaultdict
    
    # Group packages by their upstream_git_url
    url_groups = defaultdict(list)
    packages_without_url = []
    
    for package_name, package_info in package_dict_by_name.items():
        git_url = package_info.get("upstream_git_url")
        if git_url:
            url_groups[git_url].append((package_name, package_info))
        else:
            packages_without_url.append((package_name, package_info))
    
    # Build the sorted result
    unique_packages = []  # First occurrence of each URL group
    duplicated_packages = []    # Additional packages with same URLs
    
    for git_url, packages in url_groups.items():
        if len(packages) == 1:
            # Single package with this URL - goes to unique
            unique_packages.extend(packages)
        else:
            # Multiple packages with same URL - first goes to unique, others to duplicated
            unique_packages.append(packages[0])
            duplicated_packages.extend(packages[1:])
    
    # Add packages without URL to unique packages
    unique_packages.extend(packages_without_url)
    
    # Convert to dictionaries
    unique_dict = {package_name: package_info for package_name, package_info in unique_packages}
    duplicated_dict = {package_name: package_info for package_name, package_info in duplicated_packages}

    return unique_dict, duplicated_dict


def main():
    """Main entry point for the repository information fetcher."""
    parser = argparse.ArgumentParser(
        description="Fetch repository information for Debian packages from GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Process default files (data/packages.yaml, data/dependencies.yaml)
  %(prog)s -d /data/debian    # Process files in specific directory
  %(prog)s -c 5              # Use 5 concurrent processes
"""
    )
    
    parser.add_argument(
        "--directory", "-d",
        type=Path,
        help="Directory containing packages.yaml and dependencies.yaml files",
        metavar="DIR"
    )
    
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Maximum number of concurrent processes (default: 10)",
        metavar="N"
    )
    
    args = parser.parse_args()
    
    # Determine input directory
    if args.directory:
        input_dir = args.directory
        logger.info(f"Using specified directory: {input_dir}")
    else:
        input_dir = get_data_dir()
        logger.info(f"Using default data directory: {input_dir}")
    
    # Define files to process
    package_file = input_dir / "packages.yaml"
    dependency_file = input_dir / "dependencies.yaml"
    files_to_process = [package_file, dependency_file]
    
    # Validate files exist
    for file in files_to_process:
        if not file.exists():
            logger.error(f"File {file} does not exist.")
            continue
        
        logger.info(f"Processing file: {file}")
        collect_repo_info(file, max_concurrency=args.concurrency)


if __name__ == "__main__":
    main()