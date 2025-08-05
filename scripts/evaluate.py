#! /usr/bin/env python3

import logging
from hsbriskevaluator.utils.file import get_data_dir
from hsbriskevaluator.evaluator import HSBRiskEvaluator, DependencyEvaluator
import asyncio
from hsbriskevaluator.collector.repo_info import RepoInfo
import yaml
import os
from typing import Dict
from pathlib import Path
logger = logging.getLogger(__name__)
yaml.Dumper.ignore_aliases = lambda *args: True  # Disable all aliases

def save_yaml(data: Dict, file_path: Path):
    """Save a Python dictionary to a YAML file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(file_path, "w", 0o666) as f:
            yaml.dump(data, f, allow_unicode=True)
        logger.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving YAML file {file_path}: {e}")

if __name__ == "__main__":
    (get_data_dir() / "results"). mkdir(parents=True, exist_ok=True)
    for name in os.listdir(get_data_dir() / "repo_info"):
        logger.info(f"Processing {name}:")
        with open(get_data_dir() / 'repo_info' / name, "r") as f:
            repo_info_dict = yaml.safe_load(f)
            repo_info = RepoInfo.model_validate(repo_info_dict)
        evaluator = HSBRiskEvaluator(repo_info)
        result = asyncio.run(evaluator.evaluate())
        save_yaml(result.model_dump(), get_data_dir() / 'results' / name)
