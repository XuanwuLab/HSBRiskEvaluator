import logging
import asyncio
import re
import requests
from hsbriskevaluator.evaluator.base import (
    BaseEvaluator,
    CIEvalResult,
    WorkflowAnalysis,
    DangerousTriggerAnalysis,
)
from typing import Optional
from pydantic import ValidationError
from hsbriskevaluator.collector.repo_info import RepoInfo, Workflow
from hsbriskevaluator.evaluator.settings import EvaluatorSettings
from hsbriskevaluator.utils.llm import get_async_instructor_client, call_llm_with_client
from hsbriskevaluator.utils.file import get_data_dir, is_binary
from hsbriskevaluator.utils.prompt import (
    CI_WORKFLOW_ANALYSIS_PROMPT,
    CI_WORKFLOW_ANALYSIS_MODEL_ID,
)
import yaml

llm_client = get_async_instructor_client()
logger = logging.getLogger(__name__)


class CIEvaluator(BaseEvaluator):
    """Evaluator for CI/CD Security metrics"""

    def __init__(
        self,
        repo_info: RepoInfo,
        settings: Optional[EvaluatorSettings] = None,
    ):
        super().__init__(repo_info)
        if settings is None:
            settings = EvaluatorSettings()
        self.settings = settings
        self.llm_model_name = settings.ci_workflow_analysis_model_id
        self.max_concurrency = settings.ci_max_concurrency
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self.client = get_async_instructor_client()

    async def evaluate(self) -> CIEvalResult:
        """Evaluate CI/CD security metrics"""
        logger.info(f"Starting CI evaluation for repository: {self.repo_info.repo_id}")

        try:
            has_dependabot = False
            workflow_analysis = []

            # Analyze workflows concurrently
            github_workflows = []
            for workflow in self.repo_info.workflow_list:
                if workflow.path.startswith(".github"):
                    github_workflows.append(workflow)
                elif workflow.path == "dynamic/dependabot/dependabot-updates":
                    has_dependabot = True

            if github_workflows:
                workflow_analysis = await self._analyze_workflows_concurrently(
                    github_workflows
                )

            result = CIEvalResult(
                has_dependabot=has_dependabot, workflow_analysis=workflow_analysis
            )

            logger.info(f"CI evaluation completed for {self.repo_info.repo_id}")
            return result

        except Exception as e:
            logger.error(f"Error during CI evaluation: {str(e)}")
            raise

    async def _analyze_workflows_concurrently(
        self, workflows: list[Workflow]
    ) -> list[WorkflowAnalysis]:
        """Analyze multiple workflows concurrently with rate limiting"""
        tasks = []
        for workflow in workflows:
            task = self._analyze_all(workflow)
            tasks.append(task)

        # Execute all tasks concurrently with rate limiting
        results = await asyncio.gather(*tasks, return_exceptions=True)


        # Handle any exceptions and return valid results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Failed to analyze workflow {workflows[i].name}: {str(result)}"
                )
                # Create a fallback result for failed analysis
                valid_results.append(
                    WorkflowAnalysis(
                        name=workflows[i].name,
                        path=workflows[i].path,
                        dangerous_token_permission=False,
                        dangerous_action_provider_or_pin=False,
                        dangerous_trigger=DangerousTriggerAnalysis(
                            is_dangerous=False,
                            danger_level=0.0,
                            reason=f"Analysis failed: {str(result)}",
                        ),
                    )
                )
            else:
                valid_results.append(result)

        return valid_results

    async def _analyze_all(self, workflow: Workflow) -> WorkflowAnalysis:
        """Analyze a single workflow for security issues"""
        try:
            workflow_dict = yaml.safe_load(workflow.content)

            # Run dangerous trigger analysis asynchronously
            dangerous_trigger = await self._check_dangerous_trigger(workflow)

            return WorkflowAnalysis(
                name=workflow.name,
                path=workflow.path,
                dangerous_token_permission=self._check_token_permissions(workflow_dict),
                dangerous_action_provider_or_pin=self._check_action_provider_and_pin(
                    workflow_dict
                ),
                dangerous_trigger=dangerous_trigger,
            )
        except Exception as e:
            logger.warning(f"Failed to parse workflow {workflow.name}: {str(e)}")
            return WorkflowAnalysis(
                name=workflow.name,
                path=workflow.path,
                dangerous_token_permission=False,
                dangerous_action_provider_or_pin=False,
                dangerous_trigger=DangerousTriggerAnalysis(
                    is_dangerous=False,
                    danger_level=0.0,
                    reason=f"Failed to parse workflow: {str(e)}",
                ),
            )

    def _check_token_permissions(self, workflow: dict) -> bool:
        """Check if the workflow has dangerous token permissions"""
        missing_permissions = False
        if "permissions" not in workflow:
            missing_permissions = True
        if workflow.get("permissions") == "write-all":
            return True
        for jobs in workflow.get("jobs", {}).values():
            if "permissions" in jobs:
                if jobs["permissions"] == "write-all":
                    return True
            else:
                if missing_permissions:
                    return True
        return False

    def _check_action_provider_and_pin(self, workflow: dict) -> bool:
        """Check if the workflow uses actions from untrusted providers or didn't pin the actions to a specific SHA"""
        if "jobs" not in workflow:
            return False
        for job in workflow["jobs"].values():
            if "steps" not in job:
                continue
            for step in job["steps"]:
                if "uses" in step:
                    uses = step["uses"]
                    if "@" not in uses:
                        return True  # Unpinned action
                    action, version = uses.split("@", 1)
                    if not re.match("[a-f0-9]{40}", version):
                        return True  # Not a SHA
                    if action.startswith("actions/") or action.startswith("github/"):
                        continue
                    try:
                        response = requests.get(
                            f"https://github.com/marketplace/actions/{action.split('/')[1]}",
                            timeout=self.settings.http_request_timeout,
                        )
                        if "About badges in GitHub Marketplace" not in response.text:
                            return True  # Untrusted action provider
                    except Exception as e:
                        logger.warning(
                            f"Failed to check action provider for {action}: {str(e)}"
                        )
                        return True  # Assume dangerous if we can't verify
        return False

    async def _check_dangerous_trigger(
        self, workflow: Workflow
    ) -> DangerousTriggerAnalysis:
        """Use LLM to analyze if workflow has potential command injection vulnerabilities"""
        async with self._semaphore:  # Rate limiting
            try:
                response = await llm_client.chat.completions.create(
                    model=CI_WORKFLOW_ANALYSIS_MODEL_ID,
                    messages=[
                        {"role": "system", "content": CI_WORKFLOW_ANALYSIS_PROMPT},
                        {"role": "user", "content": f"Workflow file:{workflow.content}"}
                    ],
                    response_model=DangerousTriggerAnalysis,
                    extra_body={"provider": {"require_parameters": True}},
                )
                return response
            except ValidationError as e:
                logger.error(
                    f"Validation error while checking {workflow.path}: {e}")
                return DangerousTriggerAnalysis(
                    is_dangerous=False,
                    danger_level=0.0,
                    reason=f"LLM analysis validation error: {e}",
                )

            except Exception as e:
                logger.error(
                    f"Unexpected error while checking {workflow.path}: {e}")
                return DangerousTriggerAnalysis(
                    is_dangerous=False,
                    danger_level=0.0,
                    reason=f"LLM analysis failed: {e}",
                )

