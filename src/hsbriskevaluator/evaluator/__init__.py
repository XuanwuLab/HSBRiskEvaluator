"""
HSB Risk Evaluator Module

This module provides comprehensive risk evaluation for software repositories
based on three main categories:
1. Community Quality
2. Difficulty of Hiding Malicious Code (Payload)
3. Software Supply Chain Dependency Location

Usage:
    from hsbriskevaluator.evaluator import HSBRiskEvaluator
    from hsbriskevaluator.collector.repo_info import RepoInfo
    
    # Create evaluator
    evaluator = HSBRiskEvaluator(repo_info)
    
    # Run evaluation
    result = evaluator.evaluate()
    
    # Get risk summary
    summary = evaluator.get_risk_summary(result)
"""

from .main_evaluator import HSBRiskEvaluator
from .community_evaluator import CommunityEvaluator
from .payload_evaluator import PayloadEvaluator
from .dependency_evaluator import DependencyEvaluator
from .CI_evaluator import CIEvaluator
from .base import (
    BaseEvaluator,
    EvalResult,
    CommunityEvalResult,
    PayloadHiddenEvalResult,
    DependencyEvalResult,
    CIEvalResult,
)

__all__ = [
    "HSBRiskEvaluator",
    "CommunityEvaluator",
    "PayloadEvaluator",
    "DependencyEvaluator",
    "BaseEvaluator",
    "EvalResult",
    "CommunityEvalResult",
    "PayloadHiddenEvalResult",
    "DependencyEvalResult",
    "CIEvalResult",
]
