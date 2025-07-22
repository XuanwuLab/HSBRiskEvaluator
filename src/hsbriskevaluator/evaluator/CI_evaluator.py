import logging
from hsbriskevaluator.evaluator.base import BaseEvaluator, CIEvalResult, WorkflowAnalysis, DangerousTriggerAnalysis
from hsbriskevaluator.collector.repo_info import RepoInfo, Workflow
from hsbriskevaluator.utils.llm import get_model
from langchain_core.prompts import ChatPromptTemplate
import yaml
import re
import requests

logger = logging.getLogger(__name__)


class CIEvaluator(BaseEvaluator):
    """Evaluator for Community Quality metrics"""

    def __init__(self, repo_info: RepoInfo, llm_model_name: str = "anthropic/claude-3.5-sonnet"):
        super().__init__(repo_info)
        self.llm = get_model(llm_model_name)

    def evaluate(self) -> CIEvalResult:
        """Evaluate community quality metrics"""
        logger.info(
            f"Starting community evaluation for repository: {self.repo_info.repo_id}")

        try:
            has_dependabot = False
            workflow_analysis = []
            for workflow in self.repo_info.workflow_list:
                if workflow.path.startswith(".github"):
                    workflow_analysis.append(self._analyze_all(workflow))
                elif workflow.path == "dynamic/dependabot/dependabot-updates":
                    has_dependabot = True

            result = CIEvalResult(
                has_dependabot=has_dependabot,
                workflow_analysis=workflow_analysis
            )

            logger.info(
                f"Community evaluation completed for {self.repo_info.repo_id}")
            return result

        except Exception as e:
            logger.error(f"Error during community evaluation: {str(e)}")
            raise

    def _analyze_all(self, workflow: Workflow) -> WorkflowAnalysis:
        workflow_dict = yaml.safe_load(workflow.content)
        return WorkflowAnalysis(name=workflow.name, path=workflow.path, dangerous_token_permission=self._check_token_permissions(workflow_dict), dangerous_action_provider_or_pin=self._check_action_provider_and_pin(workflow_dict), dangerous_trigger=self._check_dangerous_trigger(workflow))

    def _check_token_permissions(self, workflow: dict) -> bool:
        # Check if the workflow has dangerous token permissions or not
        missing_permissions = False
        if 'permissions' not in workflow:
            missing_permissions = True
        if workflow.get('permissions') == 'write-all':
            return True
        for jobs in workflow.get('jobs', {}).values():
            if 'permissions' in jobs:
                if jobs['permissions'] == 'write-all':
                    return True
            else:
                if missing_permissions:
                    return True
        return False

    def _check_action_provider_and_pin(self, workflow: dict) -> bool:
        # Check if the workflow uses actions from untrusted providers or didn't pin the actions to a specific SHA
        if 'jobs' not in workflow:
            return False
        for job in workflow['jobs'].values():
            if 'steps' not in job:
                continue
            for step in job['steps']:
                if 'uses' in step:
                    uses = step['uses']
                    if '@' not in uses:
                        return True  # Unpinned action
                    action, version = uses.split('@', 1)
                    if not re.match('[a-f0-9]{40}', version):
                        return True  # Not a SHA
                    if action.startswith('actions/') or action.startswith('github/'):
                        continue
                    if "About badges in GitHub Marketplace" not in requests.get(f"https://github.com/marketplace/actions/{action.split('/')[1]}").text:
                        return True  # Untrusted action provider
        return False

    def _check_dangerous_trigger(self, workflow: Workflow) -> DangerousTriggerAnalysis:
        """Use LLM to analyze if PR description matches its likely implementation"""
        prompt = ChatPromptTemplate.from_template("""
        Analyze if this github workflow seems have potential command injection vulnerabilities:
        {workflow_yml}

        Consider factors like:
        - Use dangerious triggers, like pull_request_target, workflow_run
        - Use input from sources like issue titles, pull request bodies, or commit messages without proper sanitization.
        - Use inline shell commands or scripts that execute user input directly.
        - Use sensitive values like tokens or credentials, without masking via GitHub Secrets.
        
        Respond with a JSON object containing:
        - is_dangerous: boolean
        - danger_level: float (0.0 to 1.0, where 0.0 is safe and 1.0 is very dangerous)
        - reason: string (brief explanation)

        Don't include any other text, just the JSON response.
    """
                                                  )

        try:
            response = self.llm.invoke(
                prompt.format(
                    workflow_yml=workflow.content
                )
            )
            # Parse LLM response - ensure it's a string
            response_content = response.content
            if isinstance(response_content, str):
                analysis = DangerousTriggerAnalysis.model_validate_json(
                    response_content)
                return analysis
            else:

                logger.warning(
                    f"Unexpected response type from LLM for {workflow.name}")
                return DangerousTriggerAnalysis(is_dangerous=False, danger_level=0, reason="Unexpected response type from LLM")
                return False

        except Exception as e:
            logger.warning(
                f"LLM analysis failed for {workflow.name}: {str(e)}")
            return DangerousTriggerAnalysis(is_dangerous=False, danger_level=0, reason=f"LLM analysis failed for {str(e)}")
            return False
