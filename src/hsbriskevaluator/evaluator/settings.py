from pydantic_settings import BaseSettings
from pathlib import Path
from hsbriskevaluator.utils.file import get_data_dir

class EvaluatorSettings(BaseSettings):
    # LLM Configuration
    default_llm_model_name: str = "openai/gpt-4.1-mini"
    ci_workflow_analysis_model_id: str = "openai/gpt-4.1-mini"
    pr_consistency_analysis_model_id: str = "openai/gpt-4.1-mini"
    payload_file_analysis_model_id: str = "openai/gpt-4.1-mini"

    # Concurrency Settings
    max_concurrency: int = 3
    ci_max_concurrency: int = 3
    community_max_concurrency: int = 3
    payload_max_concurrency: int = 5
    dependency_max_concurrency: int = 5
    
    # Community Evaluation Settings
    prs_to_analyze_limit: int = 300
    pr_consistency_confidence_threshold: float = 0.7
    pr_batch_size: int = 30

    # Community Activity Scoring
    max_issue_participants_for_normalization: float = 5.0
    max_pr_participants_for_normalization: float = 3.0
    issue_activity_weight: float = 0.3
    pr_activity_weight: float = 0.7
   
    package_list_file: Path = get_data_dir() / "debian.yaml"
    
    # HTTP Timeouts
    http_request_timeout: int = 10
    
    class Config:
        env_prefix = "HSB_EVALUATOR_"
