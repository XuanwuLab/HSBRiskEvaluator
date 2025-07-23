import logging
import asyncio
from typing import Optional
from hsbriskevaluator.evaluator.base import BaseEvaluator, EvalResult
from hsbriskevaluator.evaluator.community_evaluator import CommunityEvaluator
from hsbriskevaluator.evaluator.payload_evaluator import PayloadEvaluator
from hsbriskevaluator.evaluator.dependency_evaluator import DependencyEvaluator
from hsbriskevaluator.evaluator.CI_evaluator import CIEvaluator
from hsbriskevaluator.collector.repo_info import RepoInfo

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
            repo_info, llm_model_name, max_concurrency
        )
        self.payload_evaluator = PayloadEvaluator(
            repo_info, llm_model_name, max_concurrency
        )
        self.dependency_evaluator = DependencyEvaluator(repo_info, max_concurrency)
        self.CI_evaluator = CIEvaluator(repo_info, llm_model_name, max_concurrency)

    async def evaluate(self) -> EvalResult:
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
            ) = await self._run_evaluations_concurrently()

            # Create comprehensive result
            result = EvalResult(
                community_quality=community_result,
                payload_hidden_difficulty=payload_result,
                dependency=dependency_result,
                ci=CI_result,
            )

            logger.info(f"HSB risk evaluation completed for {self.repo_info.repo_id}")

            return result

        except Exception as e:
            logger.error(f"Error during HSB risk evaluation: {str(e)}")
            raise

    async def _run_evaluations_concurrently(self):
        """Run all evaluations concurrently using asyncio"""
        logger.debug("Running evaluations concurrently")

        try:
            # Create tasks for async evaluations
            community_task = asyncio.create_task(
                self.community_evaluator.evaluate(), name="community_evaluation"
            )
            payload_task = asyncio.create_task(
                self.payload_evaluator.evaluate(), name="payload_evaluation"
            )
            CI_task = asyncio.create_task(
                self.CI_evaluator.evaluate(), name="CI_evaluation"
            )

            # Dependency evaluator might not be async yet, so handle it separately
            dependency_task = asyncio.create_task(
                asyncio.to_thread(self.dependency_evaluator.evaluate),
                name="dependency_evaluation",
            )

            # Wait for all results - don't use return_exceptions to let exceptions propagate
            community_result = await community_task
            logger.debug("Community evaluation completed")

            payload_result = await payload_task
            logger.debug("Payload evaluation completed")

            dependency_result = await dependency_task
            logger.debug("Dependency evaluation completed")

            CI_result = await CI_task
            logger.debug("CI evaluation completed")

            return community_result, payload_result, dependency_result, CI_result

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    # Provide a sync wrapper for backward compatibility
    def evaluate_sync(self) -> EvalResult:
        """Synchronous wrapper for the async evaluate method"""
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we are, we need to run in a new thread to avoid blocking
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.evaluate())
                return future.result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run directly
            return asyncio.run(self.evaluate())
