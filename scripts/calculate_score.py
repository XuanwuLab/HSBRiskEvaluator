#! /usr/bin/env python3

import logging
import pandas as pd
from hsbriskevaluator.utils.file import get_data_dir
from hsbriskevaluator.evaluator import EvalResult
from hsbriskevaluator.score_calculator import calculate_all_score 
import yaml
import os
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    repo_evals=[]
    for name in os.listdir(get_data_dir() / "results"):
        logger.info(f"Processing {name}:")
        with open(get_data_dir() / 'results' / name, "r") as f:
            repo_eval_dict = yaml.safe_load(f)
            repo_eval = EvalResult.model_validate(repo_eval_dict)
            repo_evals.append(repo_eval)
    scores = calculate_all_score(repo_evals)
    df = pd.DataFrame(list(map(lambda score: score.model_dump(), scores)))
    df.to_csv(get_data_dir() / "data.csv", index=False, encoding='utf-8')
