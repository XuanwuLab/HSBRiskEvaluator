"""
Utility functions for APT package management operations.
"""

import logging
import subprocess
from typing import List, Set, Tuple, Optional, Dict, Literal
from pydantic import BaseModel, ValidationError
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

logger = logging.getLogger(__name__)


class PackageInfo(BaseModel):
    """Information about a package from apt-cache show"""

    package: str
    description: Optional[str] = None
    homepage: Optional[str] = None
    essential: Optional[str] = None
    priority: Optional[str] = None
    section: Optional[str] = None
    depends: List["RecursiveDependency"] = []
    git_url: Optional[str] = None
    dependent_priorities: Optional[List[str]] = None
    upstream_git_url: Optional[str] = None
    upstream_type: Optional[str] = None
    parent_package: Optional[str] = None


class RecursiveDependency(BaseModel):
    """Information about a recursive dependency from apt-rdepends"""

    parent: str
    type: str
    name: str
    version: Optional[str] = None


class AptUtils:
    """Utility class for APT package operations"""

    def __init__(self, max_concurrency: int = 5):
        self.max_concurrency = max_concurrency
        self._package_cache: Dict[str, PackageInfo] = {}

    def run_command(self, command: List[str]) -> str:
        """Run a command and return its output"""
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False
            )
            if result.returncode != 0:
                logger.warning(
                    f"Command {' '.join(command)} failed with return code {result.returncode}"
                )
                logger.warning(f"stderr: {result.stderr}")
                return ""
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error(f"Command {' '.join(command)} timed out")
            return ""
        except Exception as e:
            logger.error(f"Error running command {' '.join(command)}: {str(e)}")
            return ""

    @lru_cache(maxsize=1000)
    def get_package_info(self, package_name: str) -> Optional[PackageInfo]:
        """Get package information using apt-cache show"""
        if package_name in self._package_cache:
            return self._package_cache[package_name]

        try:
            output = self.run_command(["apt-cache", "show", package_name])
            if not output:
                return None

            package_info = self._parse_package_info(package_name, output)
            self._package_cache[package_name] = package_info
            return package_info

        except Exception as e:
            logger.warning(f"Failed to get package info for {package_name}: {str(e)}")
            return None

    def get_package_by_priority(
        self, priority: Literal["required", "important", "standard"]
    ) -> List[PackageInfo]:
        """
        获取指定优先级的 Debian 包信息并解析为 PackageInfo 对象列表。
        """
        # 执行 aptitude search 命令，过滤指定优先级的包
        command = ["aptitude", "search", f"?priority({priority})"]
        output = self.run_command(command)
        packages = []
        for line in output.splitlines():
            # 跳过空行
            if not line.strip():
                continue
            # 解析行内容
            line_split = line.split(" ")
            name = ""
            for word in line_split:
                if len(word) > 1:
                    name = word
                    break
            if not name:
                logger.warning(f"Failed to parse package name from line: {line}")
                continue
            package_info = self.get_package_info(name)
            if package_info:
                packages.append(package_info)
        return packages

    def _parse_package_info(self, package_name: str, output: str) -> PackageInfo:
        """Parse apt-cache show output with elegant field mapping"""

        def _extract_field_value(line: str) -> Tuple[str, str]:
            """Extract field name and value from APT output line"""
            if ":" not in line:
                return "", ""

            field_name, field_value = line.split(":", 1)
            return field_name.strip().lower(), field_value.strip()

        # Initialize all fields with default values
        package_data = {
            "package": package_name,
            "description": None,
            "homepage": None,
            "essential": None,
            "priority": None,
            "section": None,
        }

        # Field mapping for case-insensitive matching
        try:
            for line in output.split("\n"):
                line = line.strip()
                if not line:
                    continue

                field_name, field_value = _extract_field_value(line)
                package_data[field_name] = field_value
            package_data.pop("depends", None)  # Remove depends field if present

            # Fetch dependencies separately to avoid overwriting parsed data
            depends = self.fetch_recursive_dependencies(package_name)

            return PackageInfo(**package_data, depends=depends)

        except Exception as e:
            logger.warning(f"Error parsing package info for {package_name}: {str(e)}")
            # Return minimal package info on parsing error
            return PackageInfo(package=package_name, depends=[])

    def _parse_depends(self, depends_str: str) -> List[str]:
        """Parse dependency string from apt-cache show"""
        depends = []

        # Split by comma and clean up
        for dep in depends_str.split(","):
            dep = dep.strip()
            if not dep or dep == "|":
                continue

            # Remove version constraints and alternatives
            dep = re.sub(r"\s*\([^)]*\)", "", dep)  # Remove version constraints
            dep = re.sub(r"\s*\|.*", "", dep)  # Remove alternatives
            dep = dep.strip()

            if dep and ":" in dep:
                dep = dep.split(":")[0]  # Remove architecture suffix

            if dep:
                depends.append(dep)

        return depends

    def _get_direct_reverse_dependencies(self, package_name: str) -> Set[str]:
        """Get direct reverse dependencies using apt rdepends"""
        try:
            output = self.run_command(
                ["apt", "rdepends", "--no-recommends", "--no-suggests", package_name]
            )
            if not output:
                return set()

            rdepends = set()
            lines = output.split("\n")

            for line in lines[1:]:  # Skip first line which is the package name
                line = line.strip()
                if not line or line.startswith("Reverse Depends:"):
                    continue

                # Remove leading spaces and extract package name
                if line.startswith("  "):
                    pkg = line.strip()
                    if ":" in pkg:
                        pkg = pkg.split(":")[0]  # Remove architecture
                    if pkg and pkg != package_name:
                        rdepends.add(pkg)

            return rdepends

        except Exception as e:
            logger.warning(
                f"Failed to get direct reverse dependencies for {package_name}: {str(e)}"
            )
            return set()

    def is_os_default_package(self, package_name: str) -> bool:
        """Check if package is an OS default package"""
        package_info = self.get_package_info(package_name)
        if not package_info:
            return False

        # Check if package is essential
        if package_info.essential and package_info.essential.lower() == "yes":
            return True

        # Check if package has required or important priority
        if package_info.priority and package_info.priority.lower() in [
            "required",
            "important",
        ]:
            return True

        # Check if package is in base system sections
        if package_info.section:
            base_sections = ["base", "admin", "utils", "shells"]
            if any(
                section in package_info.section.lower() for section in base_sections
            ):
                return True

        return False

    def is_mainstream_os_package(self, package_name: str) -> bool:
        """Check if package is a mainstream OS package"""
        package_info = self.get_package_info(package_name)
        if not package_info:
            return False

        # Check if package has standard priority
        if package_info.priority and package_info.priority.lower() == "standard":
            return True

        # Check if package is in common sections
        if package_info.section:
            common_sections = [
                "devel",
                "editors",
                "net",
                "web",
                "database",
                "mail",
                "text",
                "graphics",
                "sound",
                "video",
                "games",
                "misc",
            ]
            if any(
                section in package_info.section.lower() for section in common_sections
            ):
                return True

        return False

    def get_package_classification(self, package_name: str) -> Tuple[bool, bool]:
        """Get package classification (is_os_default, is_mainstream_os)"""
        is_os_default = self.is_os_default_package(package_name)
        is_mainstream_os = (
            self.is_mainstream_os_package(package_name) if not is_os_default else False
        )

        return is_os_default, is_mainstream_os

    def estimate_dependency_depth(
        self, package_name: str, target_packages: Set[str]
    ) -> int:
        """Estimate dependency depth by checking how many levels to reach target packages"""
        try:
            visited = set()
            queue = [(package_name, 0)]

            while queue:
                current_pkg, depth = queue.pop(0)

                if current_pkg in visited:
                    continue

                visited.add(current_pkg)

                if current_pkg in target_packages:
                    return depth

                if depth >= 5:  # Limit search depth
                    continue

                # Get dependencies of current package
                pkg_info = self.get_package_info(current_pkg)
                if pkg_info and pkg_info.depends:
                    for dep in pkg_info.depends:
                        if dep.name not in visited:  # Use dep.name instead of dep
                            queue.append((dep.name, depth + 1))

            return 5  # Default depth if no path found

        except Exception as e:
            logger.warning(
                f"Failed to estimate dependency depth for {package_name}: {str(e)}"
            )
            return 2  # Default depth

    @lru_cache(maxsize=500)
    def fetch_recursive_dependencies(
        self, package_name: str
    ) -> List[RecursiveDependency]:
        """
        Fetch the dependencies of a Debian package recursively using apt-rdepends.
        This is more efficient than the iterative approach as it gets all dependencies in one command.

        Args:
            package_name: The Debian package name

        Returns:
            List of RecursiveDependency objects with parent-child relationships
        """
        try:
            command = ["apt-rdepends", package_name]
            output = subprocess.check_output(
                command, stderr=subprocess.STDOUT, timeout=60
            ).decode("utf-8")

            return self._parse_recursive_dependencies(output)

        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error fetching recursive dependencies for {package_name}: {e.output.decode()}"
            )
            raise RuntimeError(
                f"Failed to fetch dependencies for {package_name}"
            ) from e
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout fetching recursive dependencies for {package_name}")
            raise RuntimeError(f"Timeout fetching dependencies for {package_name}")
        except Exception as e:
            logger.error(
                f"Unexpected error fetching recursive dependencies for {package_name}: {str(e)}"
            )
            raise RuntimeError(
                f"Failed to fetch dependencies for {package_name}"
            ) from e

    def get_recursive_dependencies(self, package_name: str) -> List["Dependent"]:
        """
        Get recursive dependencies and convert to Dependent objects for compatibility.
        This method is used by apt_collector.py

        Args:
            package_name: The Debian package name

        Returns:
            List of Dependent objects (compatible with repo_info.py)
        """
        from hsbriskevaluator.collector.repo_info import Dependent

        recursive_deps = self.fetch_recursive_dependencies(package_name)

        # Convert RecursiveDependency to Dependent
        dependents = []
        for dep in recursive_deps:
            dependents.append(
                Dependent(
                    parent=dep.parent, type=dep.type, name=dep.name, version=dep.version
                )
            )

        return dependents

    def _parse_recursive_dependencies(
        self, apt_output: str
    ) -> List[RecursiveDependency]:
        """
        Parse the output from apt-rdepends and structure it into a list of RecursiveDependency objects.

        Args:
            apt_output: The output from apt-rdepends

        Returns:
            List of parsed RecursiveDependency objects
        """
        lines = apt_output.strip().split("\n")
        dependencies = []
        current_package = None

        # Regular expression to extract package name and optional version
        dep_regex = re.compile(r"(\S+)(?: \(([^)]+)\))?")

        for line in lines:
            # Remove leading and trailing spaces
            stripped_line = line.strip()
            if not stripped_line or any(
                keyword in stripped_line for keyword in ["Reading", "Building", "Done"]
            ):
                continue

            # Check if the line is a package declaration or a dependency
            if line.startswith("  ") and ":" in line:
                # It's a dependency line
                try:
                    dep_type, deps = line.split(":", 1)
                    deps = deps.strip()
                    if current_package:
                        for dep in deps.split(","):
                            dep_match = dep_regex.search(dep.strip())
                            if dep_match:
                                dep_name = dep_match.group(1)
                                dep_version = (
                                    dep_match.group(2) if dep_match.group(2) else None
                                )
                                dependencies.append(
                                    RecursiveDependency(
                                        parent=current_package,
                                        type=dep_type.strip(),
                                        name=dep_name,
                                        version=dep_version,
                                    )
                                )
                except ValueError:
                    # Skip malformed lines
                    logger.debug(f"Skipping malformed dependency line: {line}")
                    continue
            else:
                # It's a new package name
                current_package = stripped_line

        return dependencies
