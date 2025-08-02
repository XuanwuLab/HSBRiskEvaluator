import logging
import asyncio
from typing import Optional
from hsbriskevaluator.evaluator.base import BaseEvaluator, EvalResult
from hsbriskevaluator.evaluator.community_evaluator import CommunityEvaluator
from hsbriskevaluator.evaluator.payload_evaluator import PayloadEvaluator
from hsbriskevaluator.evaluator.dependency_evaluator import DependencyEvaluator
from hsbriskevaluator.evaluator.CI_evaluator import CIEvaluator
from hsbriskevaluator.evaluator.settings import EvaluatorSettings
from hsbriskevaluator.collector.repo_info import RepoInfo

logger = logging.getLogger(__name__)


class HSBRiskEvaluator(BaseEvaluator):
    """Main evaluator that combines all HSB risk assessment categories"""

    def __init__(
        self,
        repo_info: RepoInfo,
        settings: Optional[EvaluatorSettings] = None,
    ):
        super().__init__(repo_info)
        if settings is None:
            settings = EvaluatorSettings()
        self.settings = settings

        # Initialize individual evaluators
        self.community_evaluator = CommunityEvaluator(repo_info, self.settings)
        self.payload_evaluator = PayloadEvaluator(repo_info, self.settings)
        self.dependency_evaluator = DependencyEvaluator(repo_info, self.settings)
        self.CI_evaluator = CIEvaluator(repo_info, self.settings)

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
                url = self.repo_info.basic_info.url,
                pkt_name = self.repo_info.pkt_name,
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
