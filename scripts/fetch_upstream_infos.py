#!/usr/bin/env python3
"""
Upstream Information Fetcher

This module fetches upstream Git repository information for Debian packages
using LLM-based analysis and updates package/dependency YAML files.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import aiohttp
import yaml
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as async_tqdm

from hsbriskevaluator.utils.file import get_data_dir
from hsbriskevaluator.utils.upstream import get_upstream_info_by_llm

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_yaml(file_path: Path) -> Dict:
    """Load a YAML file into a Python dictionary.
    
    Args:
        file_path: Path to the YAML file to load
        
    Returns:
        Dictionary containing the parsed YAML data
        
    Raises:
        SystemExit: If file doesn't exist or parsing fails
    """
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        sys.exit(1)
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            logger.debug(f"Successfully loaded YAML from {file_path}")
            return data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        sys.exit(1)
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        sys.exit(1)


def save_yaml(data: Dict, file_path: Path) -> None:
    """Save a Python dictionary to a YAML file.
    
    Args:
        data: Dictionary to save as YAML
        file_path: Path where the YAML file will be saved
        
    Raises:
        SystemExit: If saving fails
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"Successfully saved updated data to {file_path}")
    except IOError as e:
        logger.error(f"Error saving YAML file {file_path}: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error serializing data to YAML {file_path}: {e}")
        sys.exit(1)


async def check_url(url: str, session: aiohttp.ClientSession) -> bool:
    """Check if a URL is reachable using aiohttp.
    
    Args:
        url: URL to check
        session: aiohttp session to use for the request
        
    Returns:
        True if URL is reachable (status 200), False otherwise
    """
    if not url or url.strip() in ("", "null"):
        return False
        
    try:
        async with session.head(url, allow_redirects=True, timeout=10) as response:
            is_valid = response.status == 200
            logger.debug(f"URL check {url}: {response.status} -> {is_valid}")
            return is_valid
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.debug(f"URL check failed for {url}: {e}")
        return False


async def update_package_info(
    package_file: Path, 
    max_concurrency: int = 10,
    force_update: bool = False,
    skip_validation: bool = False
) -> Dict[str, int]:
    """
    Update package information with upstream details and save the results.
    
    Args:
        package_file: Path to the package YAML file to update
        max_concurrency: Maximum number of concurrent LLM requests
        force_update: If True, update packages even if they have upstream URLs
        skip_validation: If True, skip URL validation for existing upstream URLs
        
    Returns:
        Dictionary with statistics about the update process
    """
    logger.info(f"Processing package file: {package_file}")
    
    # Load package data
    package_dict_by_name = load_yaml(package_file)
    total_packages = len(package_dict_by_name)
    logger.info(f"Loaded {total_packages} packages")

    # Statistics tracking
    stats = {
        "total_packages": total_packages,
        "skipped_valid": 0,
        "skipped_invalid": 0,
        "updated_successful": 0,
        "updated_failed": 0,
        "processed": 0
    }

    # Semaphore for limiting concurrency
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_package(
        package_name: str, package_info: Dict, session: aiohttp.ClientSession
    ) -> None:
        """Process a single package to fetch upstream information."""
        async with semaphore:
            stats["processed"] += 1
            
            # Check existing upstream info
            upstream_url = package_info.get("upstream_git_url")
            if upstream_url and not force_update:
                if skip_validation:
                    logger.debug(f"Skipping {package_name}: has upstream URL (validation skipped)")
                    stats["skipped_valid"] += 1
                    return
                    
                is_valid_url = await check_url(upstream_url, session)
                if is_valid_url:
                    logger.debug(f"Skipping {package_name}: has valid upstream URL")
                    stats["skipped_valid"] += 1
                    return
                else:
                    logger.warning(f"Package {package_name} has invalid upstream URL: {upstream_url}")
                    stats["skipped_invalid"] += 1

            # Fetch upstream information using LLM
            try:
                logger.debug(f"Fetching upstream info for {package_name}")
                response = await get_upstream_info_by_llm(package_name)
                
                if response:
                    # Update package info with LLM response
                    package_info["upstream_git_url"] = response.upstream_git_url
                    package_info["upstream_type"] = response.upstream_type
                    package_info["git_url"] = response.debian_downstream_git_url
                    package_info["parent_debian_package"] = response.parent_debian_package
                    
                    logger.info(f"Updated {package_name}: {response.upstream_git_url}")
                    stats["updated_successful"] += 1
                else:
                    logger.warning(f"No upstream info found for {package_name}")
                    stats["updated_failed"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {package_name}: {e}")
                stats["updated_failed"] += 1

    # Process all packages asynchronously
    logger.info(f"Starting processing with max concurrency: {max_concurrency}")
    
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            process_package(package_name, package_info, session)
            for package_name, package_info in package_dict_by_name.items()
        ]
        
        try:
            await async_tqdm.gather(*tasks, desc=f"Processing {package_file.name}")
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            
    # Save updated package data
    logger.info("Saving updated package data...")
    save_yaml(package_dict_by_name, package_file)
    
    return stats


def validate_input_files(files_to_process: list[Path]) -> None:
    """Validate that input files exist and are readable.
    
    Args:
        files_to_process: List of file paths to validate
        
    Raises:
        SystemExit: If any file is invalid
    """
    for file_path in files_to_process:
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            sys.exit(1)
        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            sys.exit(1)
        if not file_path.suffix.lower() in ('.yaml', '.yml'):
            logger.warning(f"File does not have YAML extension: {file_path}")


def main() -> None:
    """Main entry point for the upstream information fetcher."""
    parser = argparse.ArgumentParser(
        description="Fetch upstream Git repository information for Debian packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Process default files
  %(prog)s -d /data/debian              # Process files in specific directory
  %(prog)s -f packages.yaml deps.yaml  # Process specific files
  %(prog)s --force --concurrency 20    # Force update with high concurrency
"""
    )
    
    parser.add_argument(
        "--directory", "-d",
        type=Path,
        help="Directory containing packages.yaml and dependencies.yaml files",
        metavar="DIR"
    )
    
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        type=Path,
        help="Specific YAML files to process",
        metavar="FILE"
    )
    
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Maximum number of concurrent LLM requests (default: 10)",
        metavar="N"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update packages even if they already have upstream URLs"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation of existing upstream URLs"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Determine files to process
    if args.files:
        files_to_process = args.files
        logger.info(f"Processing specified files: {[str(f) for f in files_to_process]}")
    elif args.directory:
        target_dir = args.directory
        files_to_process = [
            target_dir / "packages.yaml",
            target_dir / "dependencies.yaml"
        ]
        logger.info(f"Processing files in directory: {target_dir}")
    else:
        # Default: use data directory
        data_dir = get_data_dir()
        files_to_process = [
            data_dir / "packages.yaml",
            data_dir / "dependencies.yaml"
        ]
        logger.info(f"Processing default files in: {data_dir}")
    
    # Validate input files
    validate_input_files(files_to_process)
    
    # Validate concurrency parameter
    if args.concurrency < 1 or args.concurrency > 50:
        logger.error("Concurrency must be between 1 and 50")
        sys.exit(1)
    
    try:
        # Process each file
        total_stats = {
            "total_packages": 0,
            "skipped_valid": 0,
            "skipped_invalid": 0,
            "updated_successful": 0,
            "updated_failed": 0,
            "processed": 0
        }
        
        for file_path in files_to_process:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {file_path}")
            
            stats = asyncio.run(update_package_info(
                package_file=file_path,
                max_concurrency=args.concurrency,
                force_update=args.force,
                skip_validation=args.skip_validation
            ))
            
            # Accumulate statistics
            for key, value in stats.items():
                total_stats[key] += value
            
            # Print file statistics
            logger.info(f"File {file_path.name} statistics:")
            logger.info(f"  Total packages: {stats['total_packages']}")
            logger.info(f"  Skipped (valid): {stats['skipped_valid']}")
            logger.info(f"  Skipped (invalid): {stats['skipped_invalid']}")
            logger.info(f"  Updated (success): {stats['updated_successful']}")
            logger.info(f"  Updated (failed): {stats['updated_failed']}")
        
        # Print final summary
        logger.info(f"\n{'='*60}")
        logger.info("Final Summary:")
        logger.info(f"  Total packages processed: {total_stats['total_packages']}")
        logger.info(f"  Packages with valid upstream URLs: {total_stats['skipped_valid']}")
        logger.info(f"  Packages with invalid upstream URLs: {total_stats['skipped_invalid']}")
        logger.info(f"  Successfully updated packages: {total_stats['updated_successful']}")
        logger.info(f"  Failed updates: {total_stats['updated_failed']}")
        
        if total_stats['updated_failed'] > 0:
            logger.warning(f"Some packages failed to update ({total_stats['updated_failed']} failures)")
        
        logger.info("Processing completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
