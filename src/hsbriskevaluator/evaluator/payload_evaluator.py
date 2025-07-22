import logging
import os
from typing import List, Set
from pathlib import Path
from hsbriskevaluator.evaluator.base import BaseEvaluator, PayloadHiddenEvalResult, PayloadHiddenDetail
from hsbriskevaluator.collector.repo_info import RepoInfo
from hsbriskevaluator.utils.llm import get_model
from hsbriskevaluator.utils.file import get_data_dir, is_binary
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PayloadEvaluator(BaseEvaluator):
    """Evaluator for Difficulty of Hiding Malicious Code metrics"""

    # Common binary file extensions that could be used to hide payloads

    def __init__(
        self, repo_info: RepoInfo, llm_model_name: str = "anthropic/claude-3.5-sonnet"
    ):
        super().__init__(repo_info)
        self.llm = get_model(llm_model_name)

    def evaluate(self) -> PayloadHiddenEvalResult:
        """Evaluate payload hiding difficulty metrics"""
        logger.info(
            f"Starting payload evaluation for repository: {self.repo_info.repo_id}"
        )

        try:
            # Analyze binary files in the repository (already verified by collector)
            allows_binary_test_files, test_files_details = self._check_binary_test_files()
            allows_binary_document_files, document_files_details = self._check_binary_document_files()

            # Count and analyze binary files
            binary_files_count = len(self.repo_info.binary_file_list)

            # Calculate overall risk score

            result = PayloadHiddenEvalResult(
                allows_binary_test_files=allows_binary_test_files,
                allows_binary_document_files=allows_binary_document_files,
                binary_files_count=binary_files_count,
                details=test_files_details + document_files_details
            )

            logger.info(
                f"Payload evaluation completed for {self.repo_info.repo_id}")
            return result

        except Exception as e:
            logger.error(f"Error during payload evaluation: {str(e)}")
            raise

    def _check_binary_test_files(self) -> tuple[bool, List[PayloadHiddenDetail]]:
        """Check if repository allows binary files in test directories using LLM analysis"""
        test_binary_files = []

        details = []
        for file_path in self.repo_info.binary_file_list:
            detail = self._analyze_file_with_llm(file_path)

            # Consider files that LLM identifies as test files with reasonable confidence
            if detail.is_test_file:
                test_binary_files.append(file_path)
                logger.debug(
                    f"LLM identified test binary file: {file_path} - {detail.reason}"
                )
            details.append(detail)

        test_binaries = len(test_binary_files) > 0

        if test_binaries:
            logger.warning(
                f"Found {len(test_binary_files)} binary files identified as test files by LLM"
            )

        return test_binaries, details

    def _check_binary_document_files(self) -> tuple[bool, List[PayloadHiddenDetail]]:
        """Check if repository allows binary document/media files using LLM analysis"""
        document_files = []

        details = []
        for file_path in self.repo_info.binary_file_list:
            detail = self._analyze_file_with_llm(file_path)

            # Consider files that LLM identifies as documentation or media files with reasonable confidence
            if detail.is_documentation:
                document_files.append(file_path)
                logger.debug(
                    f"LLM identified document/media binary file: {file_path} - {detail.reason}"
                )
            details.append(detail)
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

    def _analyze_file_with_llm(self, file_path: str) -> PayloadHiddenDetail:
        """Use LLM to analyze file path and categorize the binary file"""
        prompt = ChatPromptTemplate.from_template(
            """
        Analyze this binary file path from a software repository to categorize its purpose and assess risk.
        
        File Path: {file_path}
        
        Please analyze:
        1. Is this file likely a test file, test fixture, or test resource?
        2. Is this file documentation-related (images in docs, example files, etc.)?
        3. Is this file an image, media file, or similar content file?
        
        Consider the context:
        - Path structure and directory names
        - File naming conventions
        - Common patterns in software repositories
        - Whether the file type makes sense in its location
        
        Respond with a JSON object containing:
        - reason: string (brief explanation of your assessment)
        - file_type: string (inferred file type/purpose, e.g., "test_resource", "documentation_image", "executable", etc.)
        - is_test_file: boolean (true if this appears to be a test file or test resource)
        - is_documentation: boolean (true if this appears to be documentation-related)
        - file_path: string (the original file path for reference)

        Don't include any other text, just the JSON response.
        """
        )

        try:
            response = self.llm.invoke(prompt.format(file_path=file_path))

            response_content = response.content
            if isinstance(response_content, str):
                analysis = PayloadHiddenDetail.model_validate_json(
                    response_content)
                return analysis
            else:
                logger.warning(
                    f"Unexpected response type from LLM for file {file_path}"
                )
                return PayloadHiddenDetail(
                    file_path=file_path,
                    reason="Failed to parse LLM response",
                    file_type="unknown",
                    is_test_file=False,
                    is_documentation=False,
                )

        except Exception as e:
            logger.warning(
                f"LLM analysis failed for file {file_path}: {str(e)}")
            return PayloadHiddenDetail(
                file_path=file_path,
                reason=f"Analysis failed: {str(e)}",
                file_type="unknown",
                is_test_file=False,
                is_documentation=False,
            )
