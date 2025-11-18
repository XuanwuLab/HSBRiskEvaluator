from hsbriskevaluator.evaluator import HSBRiskEvaluator
from hsbriskevaluator.collector.repo_info import RepoInfo
from hsbriskevaluator.collector import collect_all
from datetime import timedelta
import asyncio

# Load repository information (from collector)
repo_info = asyncio.run(collect_all(
    pkt_type='debian',
    pkt_name='xz-utils',
    repo_name='tukaani-project/xz',
))
# Create evaluator
evaluator = HSBRiskEvaluator(repo_info)

# Run evaluation
result = asyncio.run(evaluator.evaluate())

print(result)
