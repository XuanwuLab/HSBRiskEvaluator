import logging
from typing import Optional
from hsbriskevaluator.evaluator.base import BaseEvaluator, EvalResult
from hsbriskevaluator.evaluator.community_evaluator import CommunityEvaluator
from hsbriskevaluator.evaluator.payload_evaluator import PayloadEvaluator
from hsbriskevaluator.evaluator.dependency_evaluator import DependencyEvaluator
from hsbriskevaluator.evaluator.CI_evaluator import CIEvaluator
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
        self.community_evaluator = CommunityEvaluator(
            repo_info, llm_model_name)
        self.payload_evaluator = PayloadEvaluator(repo_info, llm_model_name)
        self.dependency_evaluator = DependencyEvaluator(
            repo_info, max_concurrency)
        self.CI_evaluator = CIEvaluator(repo_info, llm_model_name)

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
                CI_result,
            ) = self._run_evaluations_concurrently()

            # Calculate overall risk score

            # Create comprehensive result
            result = EvalResult(
                community_quality=community_result,
                payload_hidden_difficulty=payload_result,
                dependency=dependency_result,
                ci=CI_result,
            )

            logger.info(
                f"HSB risk evaluation completed for {self.repo_info.repo_id}")

            return result

        except Exception as e:
            logger.error(f"Error during HSB risk evaluation: {str(e)}")
            raise

    def _run_evaluations_concurrently(self):
        """Run all evaluations concurrently using ThreadPoolExecutor"""
        logger.debug("Running evaluations concurrently")

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            # Submit all evaluation tasks
            community_future = executor.submit(
                self.community_evaluator.evaluate)
            payload_future = executor.submit(self.payload_evaluator.evaluate)
            dependency_future = executor.submit(
                self.dependency_evaluator.evaluate)
            CI_future = executor.submit(self.CI_evaluator.evaluate)

            # Wait for all results
            try:
                community_result = community_future.result()
                logger.debug("Community evaluation completed")

                payload_result = payload_future.result()
                logger.debug("Payload evaluation completed")

                dependency_result = dependency_future.result()
                logger.debug("Dependency evaluation completed")
                
                CI_result = CI_future.result()
                logger.debug("CI evaluation completed")
                return community_result, payload_result, dependency_result, CI_result

            except Exception as e:
                logger.error(f"Evaluation failed: {str(e)}")
                raise
