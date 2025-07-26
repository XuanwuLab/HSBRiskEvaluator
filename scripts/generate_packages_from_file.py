import argparse
import logging
from dotenv import load_dotenv
from hsbriskevaluator.utils.apt_utils import AptUtils
from hsbriskevaluator.utils.file import get_data_dir
from pathlib import Path
import yaml
from tqdm import tqdm
from typing import Set, Dict, List

# Initialize environment variables and logging
load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize utility classes
apt_utils = AptUtils()


def save_to_yaml(data, file_path: Path) -> None:
    """Save data to a YAML file.
    
    Args:
        data: Data to save as YAML
        file_path: Path where the YAML file will be saved
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"Saved data to {file_path}")
    except IOError as e:
        logger.error(f"Error saving YAML file {file_path}: {e}")
        raise


def read_package_list(file_path: str) -> List[str]:
    """Read package names from a text file.
    
    Args:
        file_path: Path to the text file containing package names
        
    Returns:
        List of package names (empty lines and whitespace stripped)
        
    Raises:
        IOError: If file cannot be read
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        logger.info(f"Read {len(packages)} packages from {file_path}")
        logger.debug(f"First 5 packages: {packages[:5]}")
        return packages
    except IOError as e:
        logger.error(f"Error reading package list file {file_path}: {e}")
        raise


def process_packages_from_list(package_names: List[str]) -> Dict:
    """
    Process packages from a list of package names:
    - Fetch package information for each package.
    - Return a dictionary of package details.
    """
    logger.info(f"Processing {len(package_names)} packages...")
    
    packages_dict = {}
    for package_name in tqdm(package_names, desc="Fetching package info"):
        package_info = apt_utils.get_package_info(package_name)
        if package_info:
            packages_dict[package_name] = package_info.model_dump()
        else:
            logger.warning(f"Package {package_name} not found")
    
    return packages_dict


def process_dependencies(package_dict: Dict) -> Dict:
    """
    Process dependencies for packages:
    - Extract dependency information from the package dictionary.
    - Fetch details of dependency packages.
    - Return a dictionary of dependency details.
    """
    logger.info("Processing package dependencies...")

    # Collect dependency names
    dependency_names: Set[str] = set()
    for package_name, package_info in tqdm(
        package_dict.items(), desc="Analyzing dependencies"
    ):
        if "depends" in package_info:
            for dep in package_info["depends"]:
                dependency_names.add(dep["name"])

    # Fetch details for each dependency
    dependency_packages = []
    for dep_name in tqdm(dependency_names, desc="Fetching dependencies"):
        dep_package = apt_utils.get_package_info(dep_name)
        if dep_package:
            dependency_packages.append(dep_package.model_dump())

    return {pkg["package"]: pkg for pkg in dependency_packages}


def main():
    """Main function: Process packages and dependencies from a file."""
    parser = argparse.ArgumentParser(
        description="Generate packages and dependencies from a package list file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s packages.txt                    # Use default output directory
  %(prog)s packages.txt -o /output        # Specify output directory
  %(prog)s packages.txt -d debian         # Specify target subdirectory
  %(prog)s packages.txt -o /out -d debian # Specify both output and target
"""
    )
    
    parser.add_argument(
        "file_path", 
        help="Path to the file containing package names (one per line)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory (defaults to data directory)",
        metavar="DIR"
    )
    
    parser.add_argument(
        "--directory", "-d",
        help="Target subdirectory name within output directory (e.g., 'debian', 'ubuntu')",
        metavar="NAME"
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
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_data_dir()
    
    # Add target subdirectory if specified
    
    logger.info(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate input file
    input_file = Path(args.file_path)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return 1
    
    if not input_file.is_file():
        logger.error(f"Input path is not a file: {input_file}")
        return 1
    
    try:
        # Read package list from file
        package_names = read_package_list(args.file_path)
        
        if not package_names:
            logger.warning("No package names found in input file")
            return 0
        
        # Process packages
        logger.info("Starting package processing...")
        all_packages = process_packages_from_list(package_names)
        
        if not all_packages:
            logger.warning("No valid packages found")
            return 0
        
        # Save package data
        save_to_yaml(all_packages, output_dir / "packages.yaml")
        save_to_yaml(list(all_packages.keys()), output_dir / "package_names.yaml")
        
        # Process dependencies
        logger.info("Starting dependency processing...")
        all_dependencies = process_dependencies(all_packages)
        
        # Save dependency data
        save_to_yaml(all_dependencies, output_dir / "dependencies.yaml")
        save_to_yaml(list(all_dependencies.keys()), output_dir / "dependency_names.yaml")
        
        # Summary
        logger.info("Processing completed successfully")
        logger.info(f"Generated {len(all_packages)} packages")
        logger.info(f"Generated {len(all_dependencies)} dependencies")
        logger.info(f"Output saved to: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    main()