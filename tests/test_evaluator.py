#! /usr/bin/env python3

from hsbriskevaluator.utils.file import get_data_dir
from hsbriskevaluator.evaluator import HSBRiskEvaluator
from hsbriskevaluator.collector.repo_info import RepoInfo
if __name__ == "__main__":
    with open(get_data_dir() / 'test_collector_output.json', 'r') as f:
        repo_info = RepoInfo.model_validate_json(f.read())

    evaluator = HSBRiskEvaluator(repo_info)
    result = evaluator.evaluate()
    with open(get_data_dir() / 'test_evaluator_output.json', 'w') as f:
        f.write(result.model_dump_json(indent=2))
    print(result)
