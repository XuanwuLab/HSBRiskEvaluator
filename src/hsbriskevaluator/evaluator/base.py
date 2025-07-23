from abc import ABC, abstractmethod
from os import fork
from typing import Optional, Dict
from hsbriskevaluator.collector.repo_info import RepoInfo
from hsbriskevaluator.utils.apt_utils import Dependent
from pydantic import BaseModel, Field


class CommunityEvalResult(BaseModel):
    """Community Quality evaluation results"""

    stargazers_count: int = Field(description="Number of stargazers of the repository")
    watchers_count: int = Field(description="Number of watchers of the repository")
    forks_count: int = Field(description="Number of forks of the repository")
    community_users_count: int = Field(
        description="Number of users actively participating in the community"
    )
    direct_commits: int = Field(description="Number of direct commits to main branch")
    direct_commit_users_count: int = Field(
        description="Number of users that submitted some code to main branch"
    )
    maintainers_count: int = Field(
        description="Number of maintainers with authority to merge pull requests or directly commit code"
    )
    pr_reviewers_count: int = Field(
        description="Number of people with authority to review pull requests"
    )
    required_reviewers_distribution: Dict[int, float] = Field(
        description="Distribution of reviewers required to approve a PR before merge"
    )
    estimated_prs_to_become_maintainer: float = Field(
        description="Estimated number of PRs needed to become a maintainer"
    )
    estimated_prs_to_become_reviewer: float = Field(
        description="Estimated number of PRs needed to become a reviewer"
    )
    prs_merged_without_discussion_count: int = Field(
        description="Number of pull requests merged without discussion"
    )
    prs_with_inconsistent_description_count: int = Field(
        description="Number of PRs where description is inconsistent with implementation"
    )
    avg_participants_per_issue: float = Field(
        description="Average number of participants discussing each issue"
    )
    avg_participants_per_pr: float = Field(
        description="Average number of participants discussing each PR"
    )
    community_activity_score: float = Field(
        description="Overall community activity score (0-1)"
    )


class PayloadHiddenDetail(BaseModel):
    file_path: str
    reason: str
    file_type: str
    is_test_file: bool
    is_documentation: bool


class PayloadHiddenEvalResult(BaseModel):
    """Difficulty of Hiding Malicious Code evaluation results"""

    allows_binary_test_files: bool = Field(
        description="Whether repository allows submitting binary files as test cases"
    )
    allows_binary_document_files: bool = Field(
        description="Whether repository allows submitting binary files as documents"
    )
    binary_files_count: int = Field(
        description="Total number of binary files in repository"
    )
    details: list[PayloadHiddenDetail] = Field(
        default_factory=list,
        description="Detailed analysis of binary files and their classification",
    )


class DependencyDetail(BaseModel):
    """Analysis result for dependency classification"""

    os_default_dependent: list[Dependent]
    mainstream_dependent: list[Dependent]
    cloud_product_dependent: list[Dependent]


class DependencyEvalResult(BaseModel):
    """Software Supply Chain Dependency Location evaluation results"""

    is_os_default_dependency: bool = Field(
        description="Whether repository is direct/indirect dependency of OS default software"
    )
    is_mainstream_os_dependency: bool = Field(
        description="Whether repository is direct/indirect dependency of mainstream OS software"
    )
    is_cloud_product_dependency: bool = Field(
        description="Whether repository is direct/indirect dependency of mainstream cloud products"
    )
    details: Optional[DependencyDetail] = Field(
        default=None, description="Detailed analysis of dependency classification"
    )


class DangerousTriggerAnalysis(BaseModel):
    is_dangerous: bool = Field(
        description="Whether the trigger is considered dangerous"
    )
    danger_level: float = Field(description="Score for dangerous trigger in workflow")
    reason: str = Field(description="Reason for considering the trigger dangerous")


class WorkflowAnalysis(BaseModel):
    name: str
    path: str
    dangerous_token_permission: bool = Field(
        description="Danger level for token permissions in workflow"
    )
    dangerous_action_provider_or_pin: bool = Field(
        description="Danger level for action provider and pinning in workflow"
    )
    dangerous_trigger: DangerousTriggerAnalysis = Field(
        description="Analysis for dangerous triggers in workflow"
    )


class CIEvalResult(BaseModel):
    has_dependabot: bool = Field(
        description="Whether repository has Dependabot enabled"
    )
    workflow_analysis: list[WorkflowAnalysis] = Field(
        description="Detailed analysis of CI workflows"
    )


class EvalResult(BaseModel):
    """Complete evaluation results for all risk categories"""

    community_quality: CommunityEvalResult
    payload_hidden_difficulty: PayloadHiddenEvalResult
    dependency: DependencyEvalResult
    ci: CIEvalResult


class BaseEvaluator(ABC):
    """Base class for all evaluators"""

    def __init__(self, repo_info: RepoInfo):
        self.repo_info = repo_info

    @abstractmethod
    def evaluate(self) -> BaseModel:
        """Perform evaluation and return results"""
        pass
