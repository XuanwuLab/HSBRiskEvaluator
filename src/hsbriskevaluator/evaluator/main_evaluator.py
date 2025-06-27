import logging
from typing import Optional
from hsbriskevaluator.evaluator.base import BaseEvaluator, EvalResult
from hsbriskevaluator.evaluator.community_evaluator import CommunityEvaluator
from hsbriskevaluator.evaluator.payload_evaluator import PayloadEvaluator
from hsbriskevaluator.evaluator.dependency_evaluator import DependencyEvaluator
from hsbriskevaluator.collector.repo_info import RepoInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import partial

logger = logging.getLogger(__name__)


class HSBRiskEvaluator(BaseEvaluator):
    """Main evaluator that combines all HSB risk assessment categories"""

    def __init__(
        self,
        repo_info: RepoInfo,
        llm_model_name: str = "anthropic/claude-3.5-sonnet",
        max_concurrency: int = 3,
    ):
        super().__init__(repo_info)
        self.llm_model_name = llm_model_name
        self.max_concurrency = max_concurrency

        # Initialize individual evaluators
        self.community_evaluator = CommunityEvaluator(repo_info, llm_model_name)
        self.payload_evaluator = PayloadEvaluator(repo_info, llm_model_name)
        self.dependency_evaluator = DependencyEvaluator(repo_info, max_concurrency)

    def evaluate(self) -> EvalResult:
        """Perform comprehensive HSB risk evaluation"""
        logger.info(
            f"Starting comprehensive HSB risk evaluation for repository: {self.repo_info.repo_id}"
        )

        try:
            # Run evaluations concurrently for better performance
            (
                community_result,
                payload_result,
                dependency_result,
            ) = self._run_evaluations_concurrently()

            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(
                community_result, payload_result, dependency_result
            )

            # Create comprehensive result
            result = EvalResult(
                community_quality=community_result,
                payload_hidden_difficulty=payload_result,
                dependency=dependency_result,
                overall_risk_score=overall_risk_score,
            )

            logger.info(f"HSB risk evaluation completed for {self.repo_info.repo_id}")
            logger.info(f"Overall risk score: {overall_risk_score:.3f}")

            return result

        except Exception as e:
            logger.error(f"Error during HSB risk evaluation: {str(e)}")
            raise

    def _run_evaluations_concurrently(self):
        """Run all evaluations concurrently using ThreadPoolExecutor"""
        logger.debug("Running evaluations concurrently")

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            # Submit all evaluation tasks
            community_future = executor.submit(self.community_evaluator.evaluate)
            payload_future = executor.submit(self.payload_evaluator.evaluate)
            dependency_future = executor.submit(self.dependency_evaluator.evaluate)

            # Wait for all results
            try:
                community_result = community_future.result()
                logger.debug("Community evaluation completed")

                payload_result = payload_future.result()
                logger.debug("Payload evaluation completed")

                dependency_result = dependency_future.result()
                logger.debug("Dependency evaluation completed")

                return community_result, payload_result, dependency_result

            except Exception as e:
                logger.error(f"Evaluation failed: {str(e)}")
                raise

    def _calculate_overall_risk_score(
        self, community_result, payload_result, dependency_result
    ) -> float:
        """Calculate overall risk score combining all categories"""

        # Define weights for each category based on security impact
        weights = {
            "community": 0.3,  # Community quality affects long-term security
            "payload": 0.4,  # Payload hiding is direct security threat
            "dependency": 0.3,  # Supply chain position affects impact scope
        }

        # Extract individual risk scores
        community_risk = self._calculate_community_risk_score(community_result)
        payload_risk = payload_result.risk_score
        dependency_risk = dependency_result.supply_chain_risk_score

        # Calculate weighted average
        overall_risk = (
            community_risk * weights["community"]
            + payload_risk * weights["payload"]
            + dependency_risk * weights["dependency"]
        )

        # Apply risk amplification for high-risk combinations
        overall_risk = self._apply_risk_amplification(
            overall_risk, community_risk, payload_risk, dependency_risk
        )

        # Ensure score is within bounds
        overall_risk = max(0.0, min(1.0, overall_risk))

        logger.debug(
            f"Risk scores - Community: {community_risk:.3f}, "
            f"Payload: {payload_risk:.3f}, Dependency: {dependency_risk:.3f}"
        )
        logger.debug(f"Overall risk score: {overall_risk:.3f}")

        return round(overall_risk, 3)

    def _calculate_community_risk_score(self, community_result) -> float:
        """Convert community quality metrics to risk score"""
        risk_factors = []

        # Low community activity increases risk
        activity_risk = 1.0 - community_result.community_activity_score
        risk_factors.append(activity_risk * 0.4)

        # Few reviewers increases risk
        if community_result.pr_reviewers_count < 3:
            risk_factors.append(0.3)
        elif community_result.pr_reviewers_count < 5:
            risk_factors.append(0.2)

        # PRs merged without discussion increases risk
        if community_result.prs_merged_without_discussion_count > 0:
            discussion_risk = min(
                community_result.prs_merged_without_discussion_count * 0.1, 0.4
            )
            risk_factors.append(discussion_risk)

        # Inconsistent PR descriptions increase risk
        if community_result.prs_with_inconsistent_description_count > 0:
            inconsistency_risk = min(
                community_result.prs_with_inconsistent_description_count * 0.15, 0.5
            )
            risk_factors.append(inconsistency_risk)

        # Calculate average risk
        community_risk = sum(risk_factors) / len(risk_factors) if risk_factors else 0.1
        return min(community_risk, 1.0)

    def _apply_risk_amplification(
        self,
        base_risk: float,
        community_risk: float,
        payload_risk: float,
        dependency_risk: float,
    ) -> float:
        """Apply risk amplification for dangerous combinations"""

        # High payload risk + high dependency risk = amplified threat
        if payload_risk > 0.7 and dependency_risk > 0.7:
            amplification = 0.2
            base_risk = min(1.0, base_risk + amplification)
            logger.debug("Applied high payload + high dependency amplification")

        # High payload risk + poor community = amplified threat
        elif payload_risk > 0.6 and community_risk > 0.6:
            amplification = 0.15
            base_risk = min(1.0, base_risk + amplification)
            logger.debug("Applied high payload + poor community amplification")

        # Critical dependency + poor community = amplified threat
        elif dependency_risk > 0.8 and community_risk > 0.5:
            amplification = 0.1
            base_risk = min(1.0, base_risk + amplification)
            logger.debug("Applied critical dependency + poor community amplification")

        return base_risk

    def get_risk_summary(self, result: EvalResult) -> dict:
        """Generate a human-readable risk summary"""

        def risk_level(score: float) -> str:
            if score >= 0.8:
                return "CRITICAL"
            elif score >= 0.6:
                return "HIGH"
            elif score >= 0.4:
                return "MEDIUM"
            elif score >= 0.2:
                return "LOW"
            else:
                return "MINIMAL"

        summary = {
            "overall_risk": {
                "score": result.overall_risk_score,
                "level": risk_level(result.overall_risk_score),
            },
            "community_quality": {
                "score": self._calculate_community_risk_score(result.community_quality),
                "level": risk_level(
                    self._calculate_community_risk_score(result.community_quality)
                ),
                "key_concerns": self._get_community_concerns(result.community_quality),
            },
            "payload_hiding": {
                "score": result.payload_hidden_difficulty.risk_score,
                "level": risk_level(result.payload_hidden_difficulty.risk_score),
                "key_concerns": self._get_payload_concerns(
                    result.payload_hidden_difficulty
                ),
            },
            "dependency_risk": {
                "score": result.dependency.supply_chain_risk_score,
                "level": risk_level(result.dependency.supply_chain_risk_score),
                "key_concerns": self._get_dependency_concerns(result.dependency),
            },
        }

        return summary

    def _get_community_concerns(self, community_result) -> list:
        """Extract key community concerns"""
        concerns = []

        if community_result.pr_reviewers_count < 3:
            concerns.append("Very few PR reviewers")

        if community_result.prs_merged_without_discussion_count > 5:
            concerns.append("Many PRs merged without discussion")

        if community_result.prs_with_inconsistent_description_count > 0:
            concerns.append("PRs with inconsistent descriptions detected")

        if community_result.community_activity_score < 0.3:
            concerns.append("Low community engagement")

        return concerns

    def _get_payload_concerns(self, payload_result) -> list:
        """Extract key payload hiding concerns"""
        concerns = []

        if payload_result.allows_binary_test_files:
            concerns.append("Binary files allowed in test directories")

        if payload_result.has_payload_trigger_features:
            concerns.append("Features that could trigger malicious payloads")

        if len(payload_result.suspicious_binary_files) > 0:
            concerns.append(
                f"{len(payload_result.suspicious_binary_files)} suspicious binary files detected"
            )

        if payload_result.binary_files_count > 20:
            concerns.append("High number of binary files")

        return concerns

    def _get_dependency_concerns(self, dependency_result) -> list:
        """Extract key dependency concerns"""
        concerns = []

        if dependency_result.is_os_default_dependency:
            concerns.append("Critical OS default dependency")

        if dependency_result.is_cloud_product_dependency:
            concerns.append("Critical cloud product dependency")

        if dependency_result.dependency_depth == 0:
            concerns.append("Direct system dependency")

        if len(dependency_result.critical_dependents) > 3:
            concerns.append("Many critical systems depend on this package")

        return concerns
