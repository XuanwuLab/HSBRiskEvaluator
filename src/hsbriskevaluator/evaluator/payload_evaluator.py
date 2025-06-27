import logging
import os
from typing import List, Set
from pathlib import Path
from hsbriskevaluator.evaluator.base import BaseEvaluator, PayloadHiddenEvalResult
from hsbriskevaluator.collector.repo_info import RepoInfo
from hsbriskevaluator.utils.llm import get_model
from hsbriskevaluator.utils.file import get_data_dir, is_binary
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BinaryFileAnalysis(BaseModel):
    """Analysis result for binary file risk assessment"""

    is_suspicious: bool
    confidence: float
    reason: str
    file_type: str
    is_test_file: bool
    is_documentation: bool
    is_image_or_media: bool


class PayloadEvaluator(BaseEvaluator):
    """Evaluator for Difficulty of Hiding Malicious Code metrics"""

    # Common binary file extensions that could be used to hide payloads
    BINARY_EXTENSIONS = {
        # Executable files
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".app",
        # Archive files
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        # Image files (can contain steganography)
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        # Document files (can contain macros/embedded content)
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        # Media files
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".flac",
        # Other binary formats
        ".jar",
        ".war",
        ".ear",
        ".deb",
        ".rpm",
        ".msi",
        ".dmg",
    }

    # Extensions commonly found in test directories
    TEST_BINARY_EXTENSIONS = {
        ".jar",
        ".war",
        ".exe",
        ".dll",
        ".so",
        ".bin",
        ".zip",
        ".tar.gz",
    }

    # Document-like extensions that could hide payloads
    DOCUMENT_EXTENSIONS = {
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
    }

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
            allows_binary_test_files = self._check_binary_test_files()
            allows_binary_document_files = self._check_binary_document_files()
            has_payload_trigger_features = self._check_payload_trigger_features()

            # Count and analyze binary files
            binary_files_count = len(self.repo_info.binary_file_list)

            # Calculate overall risk score
            risk_score = self._calculate_risk_score(
                allows_binary_test_files,
                allows_binary_document_files,
                has_payload_trigger_features,
                binary_files_count,
            )

            result = PayloadHiddenEvalResult(
                allows_binary_test_files=allows_binary_test_files,
                allows_binary_document_files=allows_binary_document_files,
                has_payload_trigger_features=has_payload_trigger_features,
                binary_files_count=binary_files_count,
                risk_score=risk_score,
            )

            logger.info(f"Payload evaluation completed for {self.repo_info.repo_id}")
            return result

        except Exception as e:
            logger.error(f"Error during payload evaluation: {str(e)}")
            raise

    def _check_binary_test_files(self) -> bool:
        """Check if repository allows binary files in test directories using LLM analysis"""
        test_binary_files = []

        for file_path in self.repo_info.binary_file_list:
            analysis = self._analyze_file_with_llm(file_path)

            # Consider files that LLM identifies as test files with reasonable confidence
            if analysis.is_test_file and analysis.confidence > 0.6:
                test_binary_files.append(file_path)
                logger.debug(
                    f"LLM identified test binary file: {file_path} - {analysis.reason}"
                )

        allows_test_binaries = len(test_binary_files) > 0

        if allows_test_binaries:
            logger.warning(
                f"Found {len(test_binary_files)} binary files identified as test files by LLM"
            )

        return allows_test_binaries

    def _check_binary_document_files(self) -> bool:
        """Check if repository allows binary document/media files using LLM analysis"""
        document_files = []

        for file_path in self.repo_info.binary_file_list:
            analysis = self._analyze_file_with_llm(file_path)

            # Consider files that LLM identifies as documentation or media files with reasonable confidence
            if (
                analysis.is_documentation or analysis.is_image_or_media
            ) and analysis.confidence > 0.6:
                document_files.append(file_path)
                logger.debug(
                    f"LLM identified document/media binary file: {file_path} - {analysis.reason}"
                )

        allows_document_binaries = len(document_files) > 0

        if allows_document_binaries:
            logger.info(
                f"Found {len(document_files)} binary files identified as documentation/media by LLM"
            )

        return allows_document_binaries

    def _check_payload_trigger_features(self) -> bool:
        """Check if repository has features that could trigger malicious payloads"""
        # todo: implement specific checks for features that could trigger payloads
        has_triggers = False
        return has_triggers

    def _analyze_file_with_llm(self, file_path: str) -> BinaryFileAnalysis:
        """Use LLM to analyze file path and categorize the binary file"""
        prompt = ChatPromptTemplate.from_template(
            """
        Analyze this binary file path from a software repository to categorize its purpose and assess risk.
        
        File Path: {file_path}
        
        Please analyze:
        1. Is this file likely a test file, test fixture, or test resource?
        2. Is this file documentation-related (images in docs, example files, etc.)?
        3. Is this file an image, media file, or similar content file?
        4. Does the file location/name seem suspicious for hiding malicious code?
        
        Consider the context:
        - Path structure and directory names
        - File naming conventions
        - Common patterns in software repositories
        - Whether the file type makes sense in its location
        - If this could be a legitimate project asset vs potential payload
        
        Respond with a JSON object containing:
        - is_suspicious: boolean (true if potentially suspicious for hiding malicious code)
        - confidence: float (0.0 to 1.0, confidence in your assessment)
        - reason: string (brief explanation of your assessment)
        - file_type: string (inferred file type/purpose, e.g., "test_resource", "documentation_image", "executable", etc.)
        - is_test_file: boolean (true if this appears to be a test file or test resource)
        - is_documentation: boolean (true if this appears to be documentation-related)
        - is_image_or_media: boolean (true if this appears to be an image, video, audio, or similar media file)
        """
        )

        try:
            response = self.llm.invoke(prompt.format(file_path=file_path))

            response_content = response.content
            if isinstance(response_content, str):
                analysis = BinaryFileAnalysis.model_validate_json(response_content)
                return analysis
            else:
                logger.warning(
                    f"Unexpected response type from LLM for file {file_path}"
                )
                return BinaryFileAnalysis(
                    is_suspicious=False,
                    confidence=0.0,
                    reason="Failed to parse LLM response",
                    file_type="unknown",
                    is_test_file=False,
                    is_documentation=False,
                    is_image_or_media=False,
                )

        except Exception as e:
            logger.warning(f"LLM analysis failed for file {file_path}: {str(e)}")
            return BinaryFileAnalysis(
                is_suspicious=False,
                confidence=0.0,
                reason=f"Analysis failed: {str(e)}",
                file_type="unknown",
                is_test_file=False,
                is_documentation=False,
                is_image_or_media=False,
            )

    def _calculate_risk_score(
        self,
        allows_binary_test_files: bool,
        allows_binary_document_files: bool,
        has_payload_trigger_features: bool,
        binary_files_count: int,
    ) -> float:
        """Calculate overall risk score for payload hiding (0-1)"""
        risk_factors = []

        # Binary files in tests (high risk)
        if allows_binary_test_files:
            risk_factors.append(0.4)

        # Binary document files (medium risk)
        if allows_binary_document_files:
            risk_factors.append(0.3)

        # Payload trigger features (high risk)
        if has_payload_trigger_features:
            risk_factors.append(0.4)

        # Number of binary files (scaled risk)
        if binary_files_count > 0:
            # Normalize binary file count (assuming 50+ files is very high risk)
            binary_risk = min(binary_files_count / 50.0, 1.0) * 0.3
            risk_factors.append(binary_risk)

        # Calculate weighted average risk
        if risk_factors:
            risk_score = min(sum(risk_factors), 1.0)
        else:
            risk_score = 0.0

        logger.debug(f"Payload hiding risk score: {risk_score:.3f}")
        return round(risk_score, 3)
