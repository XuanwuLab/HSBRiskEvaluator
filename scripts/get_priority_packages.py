import logging
from dotenv import load_dotenv
from hsbriskevaluator.utils.apt_utils import AptUtils
from pathlib import Path

# Initialize environment variables and logging
load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize utility classes
apt_utils = AptUtils()


def main():
    """Extract required, important, and standard package names to a text file."""
    priorities = ["required", "important", "standard"]
    
    # Collect all package names
    all_package_names = []
    
    for priority in priorities:
        logger.info(f"Processing {priority} packages...")
        packages = apt_utils.get_package_by_priority(priority)  # type:ignore
        package_names = [pkg.package for pkg in packages]
        all_package_names.extend(package_names)
        logger.info(f"Found {len(package_names)} {priority} packages")
    
    # Remove duplicates while preserving order
    unique_packages = []
    seen = set()
    for pkg in all_package_names:
        if pkg not in seen:
            unique_packages.append(pkg)
            seen.add(pkg)
    
    # Save to text file
    output_file = Path("scripts/debian_priority_packages.txt")
    with open(output_file, "w") as f:
        for package in unique_packages:
            f.write(f"{package}\n")
    
    logger.info(f"Saved {len(unique_packages)} unique packages to {output_file}")


if __name__ == "__main__":
    main()