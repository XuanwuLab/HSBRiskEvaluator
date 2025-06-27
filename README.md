Here's the updated README focused on single evaluation analysis in a markdown container:

```markdown
# HSB Risk Evaluator

The Highly Stealthy Backdoor (HSB) Risk Evaluator provides security risk assessment for software repositories by analyzing individual evaluation entries across three key dimensions.


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
from hsbriskevaluator.collector import collect_all

# Load repository information (from collector)
repo_info = await collect_all(
    pkt_type='debian',
    pkt_name='xz-utils',
    repo_name='tukaani-project/xz', 
)
# Create evaluator
evaluator = HSBRiskEvaluator(repo_info)

# Run evaluation
result = evaluator.evaluate()

print(result    )

```
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


## Evaluation Analysis

Each repository evaluation produces detailed analysis across these core aspects:

### 1. Community Quality Analysis

```python
class CommunityEvalResult(BaseModel):
    direct_commit_users_count: int          # Number of users with direct commit access
    pr_reviewers_count: int                # Number of active PR reviewers
    required_reviewers_count: int          # Minimum reviewers required for PR approval
    prs_merged_without_discussion_count: int  # PRs merged without discussion
    prs_with_inconsistent_description_count: int  # PRs with mismatched descriptions
    avg_participants_per_issue: float      # Average participants in issues
    avg_participants_per_pr: float         # Average participants in PRs
    community_activity_score: float         # Overall community engagement score (0.0-1.0)
```

**Key Indicators:**
- Few direct committers increases centralization risk
- Low PR reviewer count suggests limited oversight
- PRs merged without discussion may bypass scrutiny
- Inconsistent PR descriptions could hide malicious changes

### 2. Payload Concealment Analysis

```python
class PayloadHiddenEvalResult(BaseModel):
    allows_binary_test_files: bool         # Whether binary files are allowed in tests
    allows_binary_document_files: bool     # Whether document binaries are permitted
    has_payload_trigger_features: bool     # Existence of potential trigger mechanisms
    binary_files_count: int                # Total binary files detected
```

**Key Indicators:**
- Binary files in tests can hide malicious executables
- Document binaries may contain embedded malware
- Trigger features could enable payload execution
- Suspicious file locations warrant investigation

### 3. Dependency Impact Analysis

```python
class DependencyEvalResult(BaseModel):
    is_os_default_dependency: bool         # Included in OS default installations
    is_mainstream_os_dependency: bool      # Required by popular OS components
    is_cloud_product_dependency: bool      # Used by cloud infrastructure
```

**Key Indicators:**
- OS-level dependencies have widespread impact
- Mainstream dependencies increase attack surface
- Cloud dependencies affect infrastructure security

## Example Analysis Output

```python
{
    "community_quality": {
        "direct_commit_users_count": 2,
        "pr_reviewers_count": 3,
        "required_reviewers_count": 1,
        "prs_merged_without_discussion_count": 8,
        "prs_with_inconsistent_description_count": 2,
        "avg_participants_per_issue": 1.2,
        "avg_participants_per_pr": 2.1,
        "community_activity_score": 0.45
    },
    "payload_hidden_difficulty": {
        "allows_binary_test_files": True,
        "allows_binary_document_files": False,
        "binary_files_count": 12,
    },
    "dependency": {
        "is_os_default_dependency": False,
        "is_mainstream_os_dependency": True,
        "is_cloud_product_dependency": False,
    }
}
```
## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your OpenRouter API key is correctly set in the `.env` file
2. **Rate Limiting**: Reduce `max_concurrency` if hitting API rate limits

### Debug Mode

Enable debug logging for detailed evaluation information:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
)
```

## Contributing

To extend the evaluator:

1. **New Metrics**: Add new evaluation metrics by extending the result models
2. **Custom Evaluators**: Create custom evaluators by inheriting from `BaseEvaluator`
3. **Risk Algorithms**: Modify risk calculation algorithms in the main evaluator
4. **LLM Prompts**: Improve LLM prompts for better semantic analysis

See the source code for implementation details and extension points.