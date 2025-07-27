import logging
import asyncio
from typing import List, Set, Optional
from pydantic import ValidationError, BaseModel, Field
from pathlib import Path
from hsbriskevaluator.evaluator.base import (
    BaseEvaluator,
    PayloadHiddenEvalResult,
    PayloadHiddenDetail,
)
from hsbriskevaluator.utils.llm import get_async_instructor_client
from hsbriskevaluator.evaluator.settings import EvaluatorSettings
from hsbriskevaluator.collector.repo_info import RepoInfo
from hsbriskevaluator.utils.llm import get_async_instructor_client, call_llm_with_client
from hsbriskevaluator.utils.file import get_data_dir, is_binary
from hsbriskevaluator.utils.prompt import (
    PAYLOAD_FILES_ANALYSIS_PROMPT,
    PAYLOAD_FILES_ANALYSIS_MODEL_ID,
)

llm_client = get_async_instructor_client()
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PayloadEvaluator(BaseEvaluator):
    """Evaluator for Difficulty of Hiding Malicious Code metrics"""

    def __init__(
        self,
        repo_info: RepoInfo,
        settings: Optional[EvaluatorSettings] = None,
    ):
        super().__init__(repo_info)
        if settings is None:
            settings = EvaluatorSettings()
        self.settings = settings

    async def evaluate(self) -> PayloadHiddenEvalResult:
        files_detail = await self._analyze_binary_files()
        binary_files_count=len(self.repo_info.binary_file_list)
        allows_binary_test_files = any(detail.is_test_file for detail in files_detail)
        allows_binary_document_files = any(detail.is_documentation for detail in files_detail)
        allows_binary_code_files = any(detail.is_code for detail in files_detail)
        allows_binary_assets_files = any(detail.is_asset for detail in files_detail)
        allows_other_binary_files = any(not(detail.is_test_file or detail.is_documentation or detail.is_code or detail.is_asset) for detail in files_detail)
        return PayloadHiddenEvalResult(allows_binary_test_files=allows_binary_test_files,
            allows_binary_document_files=allows_binary_document_files,
            allows_binary_code_files=allows_binary_code_files,
            allows_binary_assets_files=allows_binary_assets_files,
            allows_other_binary_files=allows_other_binary_files,
            binary_files_count=binary_files_count,
            details=files_detail
        )
    async def _analyze_binary_files(self) -> List[PayloadHiddenDetail]:
        logger.info(
            f"Starting binary files {self.repo_info.binary_file_list} evaluation for repository: {self.repo_info.repo_id}"
        )
        tasks = []
        batch_size = self.settings.binary_file_batch_size
        file_list = self.repo_info.binary_file_list[:self.settings.binary_file_to_analyze_limit]

        if(len(file_list)) ==0: 
            return []
        page_num = (len(file_list)-1)//batch_size+1
        for page in range(page_num):
            task = self._check_binary_files(file_list[batch_size*page:batch_size*(page+1)])
            tasks.append(task)
        if not tasks:
            return []

        # Execute all tasks concurrently with rate limiting

        results = await asyncio.gather(*tasks, return_exceptions=True)
        details = []
        # Count inconsistent PRs
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Failed to analyze {file_list[i*batch_size:(i+1)*batch_size]}: {str(result)}"
                )
            else:
                details.extend(result)
        return details

    async def _check_binary_files(self, file_list) -> List[PayloadHiddenDetail]:
        """Check if repository allows binary files in test directories using LLM analysis"""
        logger.info(
            f"Starting binary files {file_list} evaluation for repository: {self.repo_info.repo_id}"
        )


        try:
            class AnalysisResult(BaseModel):
                results: List[PayloadHiddenDetail] = Field(
                    description="List of detected hidden payload details.",
                    max_length=len(file_list),
                    min_length=len(file_list)
                )
            response = await llm_client.chat.completions.create(
                model=PAYLOAD_FILES_ANALYSIS_MODEL_ID,
                messages=[
                    {"role": "system", "content": PAYLOAD_FILES_ANALYSIS_PROMPT},
                    {"role": "user", "content": f"Binary files: {file_list}"},
                ],
                response_model=AnalysisResult, 
                extra_body={"provider": {"require_parameters": True}},
            )
            return response.results
        except ValidationError as e:
            logger.error(
                f"Validation error while evaluating binary files for {self.repo_info.binary_file_list}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error while evaluating binary files for {self.repo_info.binary_file_list}: {e}"
                )
        return []
