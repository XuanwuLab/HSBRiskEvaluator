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

    def __init__(self, repo_info: RepoInfo, settings: Optional[EvaluatorSettings] = None):
        super().__init__(repo_info)
        if settings is None:
            settings = EvaluatorSettings()
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
            self_count = {}
            dependency_count = {}
            if isinstance(self.repo_info.pkt_name, str):
                pkt_name_list = [self.repo_info.pkt_name]
            else:
                pkt_name_list = self.repo_info.pkt_name
            for pkg in self.packages.packages:
                pkg_name = pkg.package
                if pkg_name in pkt_name_list:
                    is_important_package = True
                    details.append(DependencyDetail(
                        name=pkg_name, labels=pkg.labels, type='Self'))
                    for label in pkg.labels:
                        self_count[label] = self_count.get(label, 0) + 1
                for dependency in pkg.depends:
                    if dependency.name in pkt_name_list:
                        is_important_packages_dependency = True
                        details.append(DependencyDetail(
                            name=pkg_name, labels=pkg.labels, type=dependency.type))
                        for label in pkg.labels:
                            dependency_count[label] = dependency_count.get(
                                label, 0) + 1
                        break

            labels = ["priority:required", "priority:important",
                      "priority:standard", "essential"]

            self_priority_required_count = self_count.get(
                'priority:required', 0)
            self_priority_important_count = self_count.get(
                'priority:important', 0)
            self_priority_standard_count = self_count.get(
                'priority:standard', 0)
            self_essential_count = self_count.get('essential', 0)

            dependency_priority_required_count = dependency_count.get(
                'priority:required', 0)
            dependency_priority_important_count = dependency_count.get(
                'priority:important', 0)
            dependency_priority_standard_count = dependency_count.get(
                'priority:standard', 0)
            dependency_essential_count = dependency_count.get('essential', 0)

            return DependencyEvalResult(
                self_priority_required_count=self_priority_required_count,
                self_priority_important_count=self_priority_important_count,
                self_priority_standard_count=self_priority_standard_count,
                self_essential_count=self_essential_count,
                dependency_priority_required_count=dependency_priority_required_count,
                dependency_priority_important_count=dependency_priority_important_count,
                dependency_priority_standard_count=dependency_priority_standard_count,
                dependency_essential_count=dependency_essential_count,
                # details = details
            )
        except Exception as e:
            logger.error(f"Error during dependency evaluation: {str(e)}")
            raise
