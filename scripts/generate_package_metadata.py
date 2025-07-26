#!/usr/bin/env python3
"""
Package Metadata Generator

This module generates metadata for Debian packages and their dependencies by analyzing
upstream Git URLs to identify package siblings and relationships.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from tqdm import tqdm

from hsbriskevaluator.utils.file import get_data_dir

# Configure YAML dumper to disable aliases
yaml.Dumper.ignore_aliases = lambda *args: True

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
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
        logger.info(f"Successfully saved metadata to {file_path}")
    except IOError as e:
        logger.error(f"Error saving YAML file {file_path}: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error serializing data to YAML {file_path}: {e}")
        sys.exit(1)


def extract_git_url(package_data: Dict) -> Optional[str]:
    """Extract upstream Git URL from package data for grouping packages.
    
    Args:
        package_data: Dictionary containing package information
        
    Returns:
        Upstream Git URL if valid, None otherwise
    """
    upstream_git_url = package_data.get("upstream_git_url")
    
    # Check if URL is valid (not None, empty string, or "null")
    if upstream_git_url and str(upstream_git_url).strip() not in ("", "null"):
        return str(upstream_git_url).strip()
    
    return None


def process_packages_and_dependencies(target_dir: Path) -> Tuple[Dict, Dict]:
    """Process Debian packages and dependencies to generate metadata.
    
    Args:
        target_dir: Directory containing packages.yaml and dependencies.yaml files
        
    Returns:
        Tuple of (metadata_by_package, metadata_by_url) dictionaries
    """
    packages_file = target_dir / "packages.yaml"
    dependencies_file = target_dir / "dependencies.yaml"
    
    logger.info(f"Loading package data from {target_dir}")
    packages = load_yaml(packages_file)
    dependencies = load_yaml(dependencies_file)
    
    logger.info(f"Loaded {len(packages)} packages and {len(dependencies)} dependencies")
    
    metadata_by_package = {}
    metadata_by_url = {}
    
    # Process main packages
    logger.info("Processing main packages...")
    for package_name, package_data in tqdm(packages.items(), desc="Processing packages"):
        upstream_url = extract_git_url(package_data)
        
        metadata_by_package[package_name] = {
            "url": upstream_url,
            "sibling": []
        }
        
        if upstream_url:
            if upstream_url not in metadata_by_url:
                metadata_by_url[upstream_url] = []
            metadata_by_url[upstream_url].append(package_name)
    
    # Process dependency packages
    logger.info("Processing dependency packages...")
    dependency_count = 0
    for dep_name, dep_data in tqdm(dependencies.items(), desc="Processing dependencies"):
        # Only add dependencies that aren't already in main packages
        if dep_name not in metadata_by_package:
            upstream_url = extract_git_url(dep_data)
            
            metadata_by_package[dep_name] = {
                "url": upstream_url,
                "sibling": []
            }
            
            if upstream_url:
                if upstream_url not in metadata_by_url:
                    metadata_by_url[upstream_url] = []
                metadata_by_url[upstream_url].append(dep_name)
            
            dependency_count += 1
    
    logger.info(f"Added {dependency_count} unique dependencies")
    
    # Update sibling relationships
    logger.info("Computing sibling relationships...")
    for package_name in tqdm(metadata_by_package, desc="Updating siblings"):
        upstream_url = metadata_by_package[package_name]["url"]
        if upstream_url and upstream_url in metadata_by_url:
            # Create sorted list of unique siblings
            siblings = sorted(set(metadata_by_url[upstream_url]))
            metadata_by_url[upstream_url] = siblings
            metadata_by_package[package_name]["sibling"] = siblings
    
    logger.info(f"Found {len(metadata_by_url)} unique upstream repositories")
    return metadata_by_package, metadata_by_url


def validate_input_directory(directory: Path) -> None:
    """Validate that the input directory contains required files.
    
    Args:
        directory: Path to validate
        
    Raises:
        SystemExit: If validation fails
    """
    if not directory.exists():
        logger.error(f"Input directory does not exist: {directory}")
        sys.exit(1)
    
    if not directory.is_dir():
        logger.error(f"Input path is not a directory: {directory}")
        sys.exit(1)
    
    required_files = ["packages.yaml", "dependencies.yaml"]
    for filename in required_files:
        file_path = directory / filename
        if not file_path.exists():
            logger.error(f"Required file not found: {file_path}")
            sys.exit(1)
        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            sys.exit(1)


def main() -> None:
    """Main entry point for the package metadata generator."""
    parser = argparse.ArgumentParser(
        description="Generate package metadata from Debian packages and dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Use default directories
  %(prog)s -d /path/to/debian/data  # Specify input directory
  %(prog)s -d /data -o /output      # Specify both input and output
"""
    )
    
    parser.add_argument(
        "--directory", "-d",
        type=Path,
        help="Directory containing packages.yaml and dependencies.yaml files",
        metavar="DIR"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory for generated metadata files", 
        metavar="DIR"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Determine directories
    input_dir = args.directory if args.directory else get_data_dir() 
    output_dir = args.output_dir if args.output_dir else get_data_dir()
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Validate inputs
    validate_input_directory(input_dir)
    
    try:
        # Process packages and dependencies
        metadata_by_package, metadata_by_url = process_packages_and_dependencies(input_dir)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        output_files = {
            "meta_data.yaml": metadata_by_package,
            "meta_data_by_url.yaml": metadata_by_url
        }
        
        for filename, data in output_files.items():
            save_yaml(data, output_dir / filename)
        
        # Summary
        total_packages = len(metadata_by_package)
        unique_urls = len(metadata_by_url)
        packages_with_urls = sum(1 for pkg in metadata_by_package.values() if pkg["url"])
        
        logger.info("Metadata generation completed successfully")
        logger.info(f"Total packages processed: {total_packages}")
        logger.info(f"Packages with upstream URLs: {packages_with_urls}")
        logger.info(f"Unique upstream repositories: {unique_urls}")
        
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


