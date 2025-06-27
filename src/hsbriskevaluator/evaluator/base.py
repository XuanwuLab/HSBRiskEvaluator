from abc import ABC, abstractmethod
from typing import Optional
from hsbriskevaluator.collector.repo_info import RepoInfo
from hsbriskevaluator.utils.apt_utils import Dependent
from pydantic import BaseModel, Field


class CommunityEvalResult(BaseModel):
    """Community Quality evaluation results"""

    direct_commit_users_count: int = Field(
        description="Number of people allowed to directly submit code to main branch"
    )
    pr_reviewers_count: int = Field(
        description="Number of people with authority to review pull requests"
    )
    required_reviewers_count: int = Field(
        description="Number of reviewers required to approve a PR before merge"
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


class PayloadHiddenEvalResult(BaseModel):
    """Difficulty of Hiding Malicious Code evaluation results"""

    allows_binary_test_files: bool = Field(
        description="Whether repository allows submitting binary files as test cases"
    )
    allows_binary_document_files: bool = Field(
        description="Whether repository allows submitting binary files as documents"
    )
    has_payload_trigger_features: bool = Field(
        description="Whether repository enables features that could trigger malicious payloads"
    )
    binary_files_count: int = Field(
        description="Total number of binary files in repository"
    )
    risk_score: float = Field(description="Overall risk score for payload hiding (0-1)")


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


class EvalResult(BaseModel):
    """Complete evaluation results for all risk categories"""

    community_quality: CommunityEvalResult
    payload_hidden_difficulty: PayloadHiddenEvalResult
    dependency: DependencyEvalResult
    overall_risk_score: float = Field(
        description="Overall risk score combining all categories (0-1)"
    )


class BaseEvaluator(ABC):
    """Base class for all evaluators"""

    def __init__(self, repo_info: RepoInfo):
        self.repo_info = repo_info

    @abstractmethod
    def evaluate(self) -> BaseModel:
        """Perform evaluation and return results"""
        pass
