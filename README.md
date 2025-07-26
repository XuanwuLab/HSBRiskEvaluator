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

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Required for LLM-based upstream repository discovery  
- `GITHUB_TOKEN` - Primary GitHub token for API access

### Collector Settings

Collector configuration options are available through `CollectorSettings`. See [`src/hsbriskevaluator/collector/settings.py`](src/hsbriskevaluator/collector/settings.py) for detailed configuration parameters.

```python
from hsbriskevaluator.collector.settings import CollectorSettings

settings = CollectorSettings(github_tokens=["token1", "token2"])
repo_info = await collect_all(settings=settings, ...)
```

### Evaluator Settings  

Evaluator configuration parameters are defined in [`src/hsbriskevaluator/evaluator/settings.py`](src/hsbriskevaluator/evaluator/settings.py). Configure risk analysis thresholds and weights as needed.

```python
from hsbriskevaluator.evaluator.settings import EvaluatorSettings

evaluator_settings = EvaluatorSettings()
evaluator = HSBRiskEvaluator(repo_info, settings=evaluator_settings)
```


## Quick Start

```python
from hsbriskevaluator.evaluator import HSBRiskEvaluator
from hsbriskevaluator.collector.repo_info import RepoInfo
from hsbriskevaluator.collector import collect_all
from datetime import timedelta
import asyncio

# Load repository information (from collector)
repo_info = await collect_all(
    pkt_type='debian',
    pkt_name='xz-utils',
    repo_name='tukaani-project/xz',
)
# Create evaluator
evaluator = HSBRiskEvaluator(repo_info)

# Run evaluation
result = asyncio.run(evaluator.evaluate())

print(result)

```

### Individual Evaluators

You can also use individual evaluators for specific assessments:

```python
from hsbriskevaluator.evaluator import (
    CommunityEvaluator,
    PayloadEvaluator,
    DependencyEvaluator,
    CIEvaluator
)

# Community evaluation only
community_eval = CommunityEvaluator(repo_info)
community_result = asyncio.run(community_eval.evaluate())

# Payload evaluation only
payload_eval = PayloadEvaluator(repo_info)
payload_result = asyncio.run(payload_eval.evaluate())

# Dependency evaluation only
dependency_eval = DependencyEvaluator(repo_info)
dependency_result = asyncio.run(dependency_eval.evaluate())

CI_eval = CIEvaluator(repo_info)
CI_result = asyncio.run(CI_eval.evaluate())
```

## Scripts for Batch Repository Information Collection

The `scripts/` directory contains utilities for collecting repository information in batch processing workflows. These scripts work together in a specific order to gather comprehensive data about Debian packages and their upstream repositories:

### 1. Package List Generation

**[`scripts/get_priority_packages.py`](scripts/get_priority_packages.py)**  
Extracts Debian packages with "required", "important", and "standard" priorities into a text file for further processing.

**[`scripts/get_cloud_packages.py`](scripts/get_cloud_packages.py)**  
Generates lists of cloud-related packages from various sources for specialized analysis workflows.

### 2. Package Information Collection

**[`scripts/generate_packages_from_file.py`](scripts/generate_packages_from_file.py)**  
Reads a package list file and generates detailed package and dependency information using APT utilities. Creates `packages.yaml` and `dependencies.yaml` files with comprehensive package metadata.

### 3. Upstream Repository Discovery

**[`scripts/fetch_upstream_infos.py`](scripts/fetch_upstream_infos.py)**  
Uses LLM-based analysis to discover and validate upstream Git repository URLs for packages. Updates package files with upstream repository information, which is essential for the next steps.

### 4. Metadata Generation

**[`scripts/generate_package_metadata.py`](scripts/generate_package_metadata.py)**  
Processes packages and dependencies to generate metadata about package relationships and sibling packages sharing the same upstream repository. Creates `meta_data.yaml` files for efficient package grouping.

### 5. Repository Data Collection

**[`scripts/fetch_repo_infos.py`](scripts/fetch_repo_infos.py)**  
Fetches comprehensive repository information from GitHub for packages with upstream URLs. Collects commit history, contributor data, security information, and other repository metrics for risk analysis.

#### GitHub Token Management

The [`scripts/fetch_repo_infos.py`](scripts/fetch_repo_infos.py)t  script requires GitHub API access and supports multiple tokens for improved rate limiting:

1. **Create `.github_tokens` file** in the project root with one token per line
2. **Generate GitHub tokens** with repository read permissions
3. **The script automatically distributes** API requests across available tokens
4. **Rate limiting is handled automatically** to maximize throughput

Example `.github_tokens` file:
```text
ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ghp_yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
ghp_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
```

**Benefits of multiple tokens:**
- Higher API rate limits (5,000 requests/hour per token)
- Reduced processing time for large package sets
- Automatic failover if a token becomes rate limited

### Typical Workflow

```bash
# 1. Generate package list
python scripts/get_priority_packages.py

# 2. Generate package information 
python scripts/generate_packages_from_file.py debian_priority_packages.txt -d data/debian

# 3. Fetch upstream repository URLs
python scripts/fetch_upstream_infos.py -d data/debian

# 4. Generate package metadata
python scripts/generate_package_metadata.py -d data/debian

# 5. Collect repository information
python scripts/fetch_repo_infos.py -d data/debian
```

All scripts support the `--directory` parameter to specify custom input/output directories and include comprehensive help documentation accessible via `--help`.


## Evaluation Analysis

Each repository evaluation produces detailed analysis across these core aspects:

### 1. Community Quality Analysis

```python
class CommunityEvalResult(BaseModel):
    stargazers_count: int                   # Number of stargazers of the repository
    watchers_count: int                     # Number of watchers of the repository
    forks_count: int                        # Number of forks of the repository
    community_users_count: int              # Number of users actively participating in the community
    direct_commits: int                     # Number of direct commits to the main branch
    direct_commit_users_count: int          # Number of users with direct commit access
    maintainers_count: int                  # Number of maintainers with authority to merge pull requests or directly commit code
    pr_reviewers_count: int                # Number of active PR reviewers
    required_reviewers_distribution: Dict[int, float] # Distribution of reviewers required to approve a PR before merge, -1 means direct commit
    estimated_prs_to_become_maintainer: float # Estimated number of PRs needed to become a maintainer
    estimated_prs_to_become_reviewer: float # Estimated number of PRs needed to become a reviewer
    prs_merged_without_discussion_count: int  # PRs merged without discussion
    prs_with_inconsistent_description_count: int  # PRs with mismatched descriptions
    avg_participants_per_issue: float      # Average participants in issues
    avg_participants_per_pr: float         # Average participants in PRs
    community_activity_score: float         # Overall community engagement score (0.0-1.0) (just ignore it, do not have a good formula yet)
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
    binary_files_count: int                # Total binary files detected
    details: list[PayloadHiddenDetail]     # Details of the binary files
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

### 4. CI Analysis

```python
class WorkflowAnalysis(BaseModel):
    name: str
    path: str
    dangerous_token_permission: bool # Whether this workflow have dangerous token permissions
    dangerous_action_provider_or_pin: bool # Whether this workflow have dangerous action provider or pinning
    dangerous_trigger: DangerousTriggerAnalysis # Analysis for dangerous triggers in workflow

class CIEvalResult(BaseModel):
    has_dependabot: bool # Whether repository has Dependabot enabled
    workflow_analysis: list[WorkflowAnalysis] = # Analysis for each CI workflow
```

### Diff github repo with debian repo

```python
from hsbriskevaluator.collector import GitHubRepoCollector
from hsbriskevaluator.utils.diff import Comparator
from hsbriskevaluator.utils.apt_utils import AptUtils
from hsbriskevaluator.utils.file import get_data_dir

github_collector = GitHubRepoCollector()
apt_utils = AptUtils()
comparator = Comparator(github_collector, apt_utils)

diff_result=comparator.clone_and_compare("xz-utils")
```

## Example Analysis Output

```json
{
  "community_quality": {
    "stargazers_count": 792,
    "watchers_count": 792,
    "forks_count": 162,
    "community_users_count": 60,
    "direct_commits": 271,
    "direct_commit_users_count": 8,
    "maintainers_count": 8,
    "pr_reviewers_count": 11,
    "required_reviewers_distribution": {},
    "estimated_prs_to_become_maintainer": 1.0,
    "estimated_prs_to_become_reviewer": 0.7272727272727273,
    "prs_merged_without_discussion_count": 0,
    "prs_with_inconsistent_description_count": 0,
    "avg_participants_per_issue": 2.2413793103448274,
    "avg_participants_per_pr": 2.111111111111111,
    "community_activity_score": 0.627
  },
  "payload_hidden_difficulty": {
    "allows_binary_test_files": true,
    "allows_binary_document_files": false,
    "binary_files_count": 95
  },
  "dependency": {
    "is_os_default_dependency": true,
    "is_mainstream_os_dependency": true,
    "is_cloud_product_dependency": false,
    "details": null
  },
  "ci": {
    "has_dependabot": false,
    "workflow_analysis": [
      {
        "name": "CI",
        "path": ".github/workflows/ci.yml",
        "dangerous_token_permission": false,
        "dangerous_action_provider_or_pin": true,
        "dangerous_trigger": {
          "is_dangerous": false,
          "danger_level": 0.2,
          "reason": "The workflow uses safe triggers (push, pull_request, workflow_dispatch), has restricted permissions ({}), and doesn't execute arbitrary user input. The only slight risk comes from running apt-get and brew commands, but these use fixed package names rather than variables. Build parameters are passed through a controlled script rather than directly executing user input."
        }
      }
    ]
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
