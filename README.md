# HSBRiskEvaulator
# HSB Risk Evaluator Documentation

The Highly Stealthy Backdoor(HSB) Risk Evaluator provides comprehensive security risk assessment for software repositories based on three main categories defined in the framework.

## Overview

The evaluator analyzes repositories across three critical dimensions:

1. **Software Supply Chain Dependency Location** - Assesses the repository's position in the software supply chain
2. **Difficulty of Hiding Malicious Code** - Evaluates how easily malicious payloads could be hidden
3. **Community Quality** - Analyzes the health and security practices of the repository's community

## Installation

Ensure you have the required dependencies installed:

```bash
uv sync
uv venv
source .venv/bin/activate
```

Set up your environment variables in a `.env` file:

```bash
OPENAI_API_KEY = your_openrouter_api_key_here
GITHUB_TOKEN = your_github_token_here
```

## Quick Start

```python
from hsbriskevaluator.evaluator import HSBRiskEvaluator
from hsbriskevaluator.collector.repo_info import RepoInfo

# Load repository information (from collector or create manually)
repo_info = RepoInfo(...)

# Create evaluator
evaluator = HSBRiskEvaluator(repo_info)

# Run evaluation
result = evaluator.evaluate()

# Get human-readable summary
summary = evaluator.get_risk_summary(result)

print(f"Overall Risk: {summary['overall_risk']['level']}")
```

## Evaluation Categories

### 1. Community Quality Evaluation

Analyzes the repository's community health and security practices:

**Metrics:**
- Number of users with direct commit access
- Number of PR reviewers
- Required reviewers for PR approval
- PRs merged without discussion
- PRs with inconsistent descriptions (detected via LLM)
- Average participants per issue/PR
- Overall community activity score

**Risk Factors:**
- Few reviewers increase risk
- PRs merged without discussion indicate poor oversight
- Low community engagement suggests limited security review
- Inconsistent PR descriptions may hide malicious changes

### 2. Payload Hiding Difficulty

Evaluates how easily malicious code could be hidden in the repository:

**Metrics:**
- Binary files allowed in test directories
- Binary document files (images, PDFs, etc.)
- Features that could trigger malicious payloads
- Total binary file count
- Suspicious binary files (detected via heuristics and LLM)

**Risk Factors:**
- Binary files in tests can hide malicious executables
- Document files can contain embedded malware
- CI/CD configurations can be exploited to execute payloads
- Suspicious file names or locations indicate potential threats

### 3. Dependency Risk Assessment

Analyzes the repository's position in the software supply chain:

**Metrics:**
- OS default dependency status
- Mainstream OS software dependency
- Cloud product dependency
- Dependency depth in the chain
- Critical systems that depend on this package

**Risk Factors:**
- OS default dependencies have maximum impact
- Mainstream software dependencies affect many users
- Cloud product dependencies impact infrastructure
- Lower dependency depth means higher impact potential

## API Reference

### HSBRiskEvaluator

Main evaluator class that orchestrates all risk assessments.

```python
class HSBRiskEvaluator(BaseEvaluator):
    def __init__(
        self, 
        repo_info: RepoInfo, 
        llm_model_name: str = "anthropic/claude-3.5-sonnet",
        max_concurrency: int = 3
    )
    
    def evaluate(self) -> EvalResult
    def get_risk_summary(self, result: EvalResult) -> dict
```

**Parameters:**
- `repo_info`: Repository information collected via the collector module
- `llm_model_name`: LLM model to use for semantic analysis
- `max_concurrency`: Maximum concurrent evaluations (for performance)

### Individual Evaluators

You can also use individual evaluators for specific assessments:

```python
from hsbriskevaluator.evaluator import (
    CommunityEvaluator,
    PayloadEvaluator, 
    DependencyEvaluator
)

# Community evaluation only
community_eval = CommunityEvaluator(repo_info)
community_result = community_eval.evaluate()

# Payload evaluation only  
payload_eval = PayloadEvaluator(repo_info)
payload_result = payload_eval.evaluate()

# Dependency evaluation only
dependency_eval = DependencyEvaluator(repo_info)
dependency_result = dependency_eval.evaluate()
```

### Result Models

#### EvalResult
```python
class EvalResult(BaseModel):
    community_quality: CommunityEvalResult
    payload_hidden_difficulty: PayloadHiddenEvalResult
    dependency: DependencyEvalResult
    overall_risk_score: float  # 0.0 to 1.0
```

#### CommunityEvalResult
```python
class CommunityEvalResult(BaseModel):
    direct_commit_users_count: int
    pr_reviewers_count: int
    required_reviewers_count: int
    prs_merged_without_discussion_count: int
    prs_with_inconsistent_description_count: int
    avg_participants_per_issue: float
    avg_participants_per_pr: float
    community_activity_score: float  # 0.0 to 1.0
```

#### PayloadHiddenEvalResult
```python
class PayloadHiddenEvalResult(BaseModel):
    allows_binary_test_files: bool
    allows_binary_document_files: bool
    has_payload_trigger_features: bool
    binary_files_count: int
    suspicious_binary_files: list[str]
    risk_score: float  # 0.0 to 1.0
```

#### DependencyEvalResult
```python
class DependencyEvalResult(BaseModel):
    is_os_default_dependency: bool
    is_mainstream_os_dependency: bool
    is_cloud_product_dependency: bool
    dependency_depth: int
    critical_dependents: list[str]
    supply_chain_risk_score: float  # 0.0 to 1.0
```

## Risk Scoring

### Overall Risk Score Calculation

The overall risk score is calculated as a weighted average:

- **Community Quality**: 30% weight
- **Payload Hiding**: 40% weight (highest impact)
- **Dependency Risk**: 30% weight

### Risk Amplification

The system applies risk amplification for dangerous combinations:

- High payload risk + high dependency risk → +0.2 amplification
- High payload risk + poor community → +0.15 amplification  
- Critical dependency + poor community → +0.1 amplification

### Risk Levels

Risk scores are categorized into levels:

- **CRITICAL**: 0.8 - 1.0
- **HIGH**: 0.6 - 0.8
- **MEDIUM**: 0.4 - 0.6
- **LOW**: 0.2 - 0.4
- **MINIMAL**: 0.0 - 0.2

## Configuration

### LLM Models

The evaluator supports various LLM models via OpenRouter:

```python
# Use different models for different use cases
evaluator = HSBRiskEvaluator(
    repo_info,
    llm_model_name="anthropic/claude-3.5-sonnet"  # High accuracy
    # llm_model_name="openai/gpt-4"              # Alternative
    # llm_model_name="anthropic/claude-3-haiku"  # Faster/cheaper
)
```

### Performance Tuning

```python
# Adjust concurrency for performance
evaluator = HSBRiskEvaluator(
    repo_info,
    max_concurrency=5  # Higher for faster evaluation
)
```

### Logging

Configure logging to monitor evaluation progress:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('hsbriskevaluator.evaluator')
```

## Examples

### Basic Evaluation

```python
from hsbriskevaluator.evaluator import HSBRiskEvaluator

# Run evaluation
evaluator = HSBRiskEvaluator(repo_info)
result = evaluator.evaluate()

# Check overall risk
if result.overall_risk_score > 0.7:
    print("HIGH RISK repository detected!")
    
# Get detailed summary
summary = evaluator.get_risk_summary(result)
for category, details in summary.items():
    print(f"{category}: {details['level']} ({details['score']:.3f})")
```

### Filtering High-Risk Repositories

```python
def is_high_risk_repository(repo_info: RepoInfo) -> bool:
    evaluator = HSBRiskEvaluator(repo_info)
    result = evaluator.evaluate()
    
    # Define high-risk criteria
    return (
        result.overall_risk_score > 0.6 or
        result.payload_hidden_difficulty.risk_score > 0.7 or
        result.dependency.supply_chain_risk_score > 0.8
    )

# Filter repositories
high_risk_repos = [
    repo for repo in repository_list 
    if is_high_risk_repository(repo)
]
```

### Custom Risk Analysis

```python
def analyze_community_risks(repo_info: RepoInfo):
    evaluator = CommunityEvaluator(repo_info)
    result = evaluator.evaluate()
    
    risks = []
    if result.pr_reviewers_count < 2:
        risks.append("Insufficient PR reviewers")
    if result.prs_merged_without_discussion_count > 10:
        risks.append("Too many PRs merged without discussion")
    if result.community_activity_score < 0.3:
        risks.append("Low community engagement")
        
    return risks
```

## Best Practices

1. **Batch Processing**: For multiple repositories, process them in batches to manage API rate limits
2. **Caching**: Cache LLM results to avoid redundant API calls
3. **Error Handling**: Implement robust error handling for network issues and API failures
4. **Monitoring**: Log evaluation progress and results for audit trails
5. **Thresholds**: Define organization-specific risk thresholds based on your security requirements

## Limitations

1. **LLM Dependency**: Some evaluations require LLM API access and may incur costs
2. **Heuristic Analysis**: Some metrics use heuristics that may not capture all edge cases
3. **GitHub Focus**: Currently optimized for GitHub repositories
4. **Package Manager**: Currently supports Debian packages only

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your OpenRouter API key is correctly set in the `.env` file
2. **Rate Limiting**: Reduce `max_concurrency` if hitting API rate limits
3. **Memory Usage**: For large repositories, consider processing in smaller chunks
4. **Network Timeouts**: Implement retry logic for network-related failures

### Debug Mode

Enable debug logging for detailed evaluation information:

```python
import logging
logging.getLogger('hsbriskevaluator.evaluator').setLevel(logging.DEBUG)
```

## Contributing

To extend the evaluator:

1. **New Metrics**: Add new evaluation metrics by extending the result models
2. **Custom Evaluators**: Create custom evaluators by inheriting from `BaseEvaluator`
3. **Risk Algorithms**: Modify risk calculation algorithms in the main evaluator
4. **LLM Prompts**: Improve LLM prompts for better semantic analysis

See the source code for implementation details and extension points.
