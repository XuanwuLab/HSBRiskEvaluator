import logging
import yaml
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from hsbriskevaluator.evaluator.base import (
    BaseEvaluator,
    DependencyEvalResult,
    DependencyDetail,
)
from hsbriskevaluator.collector.repo_info import RepoInfo
from hsbriskevaluator.utils.apt_utils import AptUtils, Dependent

logger = logging.getLogger(__name__)


class DependencyEvaluator(BaseEvaluator):
    """Evaluator for Software Supply Chain Dependency Location metrics"""

    def __init__(self, repo_info: RepoInfo, max_concurrency: int = 5):
        super().__init__(repo_info)
        self.apt_utils = AptUtils(max_concurrency=max_concurrency)
        self.cloud_product_packages, self.cloud_product_dependencies = self._load_yaml(
            "cloud_products.yaml")
        self.os_default_packages, self.os_default_dependencies = self._load_yaml(
            "os_default.yaml")
        self.mainstream_os_packages, self.mainstream_os_dependencies = self._load_yaml(
            "mainstream_os.yaml")

        # Cache for OS default and mainstream packages discovered via apt
        self._os_default_cache: Set[str] = set()
        self._mainstream_os_cache: Set[str] = set()
        self._cache_initialized = False

    def _load_yaml(self, filename: str) -> Tuple[Set[str], Set[str]]:
        try:
            yaml_path = Path(__file__).parent / filename
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                packages = set(data.get("packages", []))
                dependencies = set(data.get("dependencies", []))
                return packages, dependencies
        except Exception as e:
            return set(), set()

    def evaluate(self) -> DependencyEvalResult:
        """Evaluate dependency location metrics"""
        logger.info(
            f"Starting dependency evaluation for repository: {self.repo_info.repo_id}"
        )

        try:
            # Analyze package classification

            # Calculate supply chain risk score

            result = DependencyEvalResult(
                is_os_default_dependency=self.repo_info.pkt_name in self.os_default_dependencies,
                is_mainstream_os_dependency=self.repo_info.pkt_name in self.mainstream_os_dependencies,
                is_cloud_product_dependency=self.repo_info.pkt_name in self.cloud_product_dependencies,
            )

            logger.info(
                f"Dependency evaluation completed for {self.repo_info.repo_id}")
            return result

        except Exception as e:
            logger.error(f"Error during dependency evaluation: {str(e)}")
            raise

    def _analyze_package_dependencies(self) -> DependencyDetail:
        """Analyze what type of dependencies this package represents using apt commands"""
        package_name = self.repo_info.pkt_name

        # Use apt-cache show to determine package classification
        is_os_default, is_mainstream_os = self.apt_utils.get_package_classification(
            package_name
        )

        # Check if it's a cloud product package
        is_cloud_product = package_name in self.cloud_product_packages

        # Identify potential dependents using apt rdepends
        os_default_dependents = []
        mainstream_dependents = []
        cloud_product_dependents = []

        for dependent in self.repo_info.dependent_list:
            is_os_default, is_mainstream_os = self.apt_utils.get_package_classification(
                package_name
            )
            if is_os_default:
                os_default_dependents.append(dependent)
            if is_mainstream_os:
                mainstream_dependents.append(dependent)
            if dependent.name in self.cloud_product_packages:
                cloud_product_dependents.append(dependent)

        result = DependencyDetail(
            os_default_dependent=os_default_dependents,
            mainstream_dependent=mainstream_dependents,
            cloud_product_dependent=cloud_product_dependents,
        )

        logger.debug(
            f"Package dependency analysis: OS_default={is_os_default}, Mainstream_OS={is_mainstream_os}, Cloud={is_cloud_product}"
        )
        return result
