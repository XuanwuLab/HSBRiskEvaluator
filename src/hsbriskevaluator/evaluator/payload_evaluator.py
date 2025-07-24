import logging
import asyncio
from typing import List, Set
from pathlib import Path
from hsbriskevaluator.evaluator.base import (
    BaseEvaluator,
    PayloadHiddenEvalResult,
    PayloadHiddenDetail,
)
from hsbriskevaluator.evaluator.settings import EvaluatorSettings
from hsbriskevaluator.collector.repo_info import RepoInfo
from hsbriskevaluator.utils.llm import get_async_instructor_client, call_llm_with_client
from hsbriskevaluator.utils.file import get_data_dir, is_binary
from hsbriskevaluator.utils.prompt import (
    PAYLOAD_FILE_ANALYSIS_PROMPT,
    PAYLOAD_FILE_ANALYSIS_MODEL_ID,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PayloadEvaluator(BaseEvaluator):
    """Evaluator for Difficulty of Hiding Malicious Code metrics"""

    def __init__(
        self,
        repo_info: RepoInfo,
        settings: EvaluatorSettings,
    ):
        super().__init__(repo_info)
        self.settings = settings
        self.llm_model_name = settings.payload_file_analysis_model_id
        self.max_concurrency = settings.payload_max_concurrency
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self.client = get_async_instructor_client()

    async def evaluate(self) -> PayloadHiddenEvalResult:
        """Evaluate payload hiding difficulty metrics"""
        logger.info(
            f"Starting payload evaluation for repository: {self.repo_info.repo_id}"
        )

        try:
            # Analyze binary files in the repository (already verified by collector)
            (
                allows_binary_test_files,
                test_files_details,
            ) = await self._check_binary_test_files()
            (
                allows_binary_document_files,
                document_files_details,
            ) = await self._check_binary_document_files()

            # Count and analyze binary files
            binary_files_count = len(self.repo_info.binary_file_list)

            # Combine details from both checks, removing duplicates
            all_details = {}
            for detail in test_files_details + document_files_details:
                all_details[detail.file_path] = detail

            result = PayloadHiddenEvalResult(
                allows_binary_test_files=allows_binary_test_files,
                allows_binary_document_files=allows_binary_document_files,
                binary_files_count=binary_files_count,
                details=list(all_details.values()),
            )

            logger.info(f"Payload evaluation completed for {self.repo_info.repo_id}")
            return result

        except Exception as e:
            logger.error(f"Error during payload evaluation: {str(e)}")
            raise

    async def _check_binary_test_files(self) -> tuple[bool, List[PayloadHiddenDetail]]:
        """Check if repository allows binary files in test directories using LLM analysis"""
        test_binary_files = []

        # Analyze all binary files concurrently
        details = await self._analyze_files_with_llm(self.repo_info.binary_file_list)

        for detail in details:
            # Consider files that LLM identifies as test files with reasonable confidence
            if detail.is_test_file:
                test_binary_files.append(detail.file_path)
                logger.debug(
                    f"LLM identified test binary file: {detail.file_path} - {detail.reason}"
                )

        test_binaries = len(test_binary_files) > 0

        if test_binaries:
            logger.warning(
                f"Found {len(test_binary_files)} binary files identified as test files by LLM"
            )

        return test_binaries, details

    async def _check_binary_document_files(
        self,
    ) -> tuple[bool, List[PayloadHiddenDetail]]:
        """Check if repository allows binary document/media files using LLM analysis"""
        document_files = []

        # Analyze all binary files concurrently
        details = await self._analyze_files_with_llm(self.repo_info.binary_file_list)

        for detail in details:
            # Consider files that LLM identifies as documentation or media files with reasonable confidence
            if detail.is_documentation:
                document_files.append(detail.file_path)
                logger.debug(
                    f"LLM identified document/media binary file: {detail.file_path} - {detail.reason}"
                )

        document_binaries = len(document_files) > 0

        if document_binaries:
            logger.info(
                f"Found {len(document_files)} binary files identified as documentation/media by LLM"
            )

        return document_binaries, details

    def _check_payload_trigger_features(self) -> bool:
        """Check if repository has features that could trigger malicious payloads"""
        # todo: implement specific checks for features that could trigger payloads
        has_triggers = False
        return has_triggers

    async def _analyze_files_with_llm(
        self, file_paths: List[str]
    ) -> List[PayloadHiddenDetail]:
        """Analyze multiple files concurrently with rate limiting"""
        tasks = []
        for file_path in file_paths:
            task = self._analyze_file_with_llm(file_path)
            tasks.append(task)

        # Execute all tasks concurrently with rate limiting
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions and return valid results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to analyze file {file_paths[i]}: {str(result)}")
                # Create a fallback result for failed analysis
                valid_results.append(
                    PayloadHiddenDetail(
                        file_path=file_paths[i],
                        reason=f"Analysis failed: {str(result)}",
                        file_type="unknown",
                        is_test_file=False,
                        is_documentation=False,
                    )
                )
            else:
                valid_results.append(result)

        return valid_results

    async def _analyze_file_with_llm(self, file_path: str) -> PayloadHiddenDetail:
        """Use LLM to analyze file path and categorize the binary file"""
        async with self._semaphore:  # Rate limiting
            messages = [
                {
                    "role": "user",
                    "content": PAYLOAD_FILE_ANALYSIS_PROMPT.format(file_path=file_path),
                }
            ]

            try:
                analysis = await call_llm_with_client(
                    client=self.client,
                    model_id=self.llm_model_name,
                    messages=messages,
                    response_model=PayloadHiddenDetail,
                )
                return analysis

            except Exception as e:
                logger.warning(f"LLM analysis failed for file {file_path}: {str(e)}")
                return PayloadHiddenDetail(
                    file_path=file_path,
                    reason=f"Analysis failed: {str(e)}",
                    file_type="unknown",
                    is_test_file=False,
                    is_documentation=False,
                )
