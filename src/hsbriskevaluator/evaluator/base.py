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
    direct_commits_ratio: float = Field(description="Ratio of direct commits to main branch")
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
    prs_merged_without_discussion_ratio: float = Field(
        description="Ratio of pull requests merged without discussion"
    )
    prs_with_inconsistent_description_ratio: float = Field(
        description="Ratio of PRs where description is inconsistent with implementation"
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
    is_code: bool
    is_asset: bool


class PayloadHiddenEvalResult(BaseModel):
    """Difficulty of Hiding Malicious Code evaluation results"""

    allows_binary_test_files: bool = Field(
        description="Whether repository allows submitting binary files as test cases"
    )
    allows_binary_document_files: bool = Field(
        description="Whether repository allows submitting binary files as documents"
    )

    allows_binary_code_files: bool = Field(
        description="Whether repository allows submitting binary files as the part of codebases"
    )
    allows_binary_assets_files: bool = Field(
        description="Whether repository allows submitting binary files as assets"
    )
    allows_other_binary_files: bool = Field(
        description="Whether repository allows submitting binary files as other files"
    )
    binary_files_count: int = Field(
        description="Total number of binary files in repository"
    )
    details: list[PayloadHiddenDetail] = Field(
        default_factory=list,
        description="Detailed analysis of binary files and their classification",
    )

class DependencyDetail(BaseModel):
    name: str = Field(description="Name of the package")
    labels: list[str] = Field(description="importance labels associated with the package")
    type: str = Field(description="Type of dependency (e.g., 'Depends', 'PreDepends', 'Self')")

class DependencyEvalResult(BaseModel):
    """Software Supply Chain Dependency Location evaluation results"""

    is_important_packages_dependency: bool = Field(
        description="Whether repository is direct/indirect dependency of some important packages"
    )
    is_important_package: bool = Field(
        description="Whether repository is considered important package itself"
    )
    details: list[DependencyDetail] = Field(
        description="Detailed analysis of dependency classification"
    )


class DangerousTriggerAnalysis(BaseModel):
    is_dangerous: bool = Field(
        description="Whether the trigger is considered dangerous"
    )
    danger_level: float = Field(
        ge=0.0,
        le=1.0,
        description="A score from 0.0 (safe) to 1.0 (very dangerous)."
    )
    reason: str = Field(description="Reason for considering the trigger dangerous")


class WorkflowAnalysis(BaseModel):
    name: str
    path: str
    dangerous_token_permission: bool = Field(
        description="Danger level for token permissions in workflow"
    )
    dangerous_action_provider_or_pin: float = Field(
        ge=0.0,
        le=1.0,
        description="Danger level for action provider and pinning in workflow, from 0.0(safe) to 1.0(very dangerous)"

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
