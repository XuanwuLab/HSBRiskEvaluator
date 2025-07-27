import logging
import yaml
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from pydantic import BaseModel
from hsbriskevaluator.evaluator.base import (
    BaseEvaluator,
    DependencyEvalResult,
    DependencyDetail,
)
from hsbriskevaluator.evaluator.settings import EvaluatorSettings
from hsbriskevaluator.collector.repo_info import RepoInfo
from hsbriskevaluator.utils.apt_utils import AptUtils, Dependent

logger = logging.getLogger(__name__)

class Dependent(BaseModel):
    name: str
    parent: Optional[str] = None
    type: str = 'Depends'
class Package(BaseModel):
    package: str
    depends: List[Dependent]
    labels: List[str] = []

class Packages(BaseModel):
    packages: List[Package]

class DependencyEvaluator(BaseEvaluator):
    """Evaluator for Software Supply Chain Dependency Location metrics"""

    def __init__(self, repo_info: RepoInfo, settings: EvaluatorSettings):
        super().__init__(repo_info)
        self.settings = settings

        with open(self.settings.package_list_file) as f:
            self.packages = Packages.model_validate(yaml.safe_load(f))

    def evaluate(self) -> DependencyEvalResult:
        """Evaluate dependency location metrics"""
        logger.info(
            f"Starting dependency evaluation for repository: {self.repo_info.repo_id}"
        )
        try:
            is_important_package = False
            is_important_packages_dependency = False
            details = []
            for pkg in self.packages.packages:
                pkg_name = pkg.package
                if pkg_name == self.repo_info.pkt_name:
                    is_important_package = True
                    details.append(DependencyDetail(name=pkg_name, labels=pkg.labels, type='Self'))
                else:
                    for dependency in pkg.depends:
                        if dependency.name == self.repo_info.pkt_name:
                            is_important_packages_dependency = True
                            details.append(DependencyDetail(name=pkg_name, labels=pkg.labels, type=dependency.type))
                            break
            return DependencyEvalResult(
                is_important_packages_dependency=is_important_packages_dependency,
                is_important_package=is_important_package,
                details=details,
                )
        except Exception as e:
            logger.error(f"Error during dependency evaluation: {str(e)}")
            raise
