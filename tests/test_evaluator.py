#! /usr/bin/env python3

import logging
from hsbriskevaluator.utils.file import get_data_dir
from hsbriskevaluator.evaluator import HSBRiskEvaluator
import asyncio
from hsbriskevaluator.collector.repo_info import RepoInfo
import yaml

if __name__ == "__main__":
    with open(get_data_dir() / 'repo_info'/'isc-dhcp-common.yaml', 'r') as f:
        repo_info_dict = yaml.safe_load(f)
        repo_info = RepoInfo.model_validate(repo_info_dict)

    evaluator = HSBRiskEvaluator(repo_info)
    result = asyncio.run(evaluator.evaluate())
    with open(get_data_dir() / 'test_evaluator_output.json', 'w') as f:
        f.write(result.model_dump_json(indent=2))
    print(result)
