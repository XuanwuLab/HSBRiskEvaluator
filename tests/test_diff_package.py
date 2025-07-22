#! /usr/bin/env python3

from hsbriskevaluator.collector import GitHubRepoCollector
from hsbriskevaluator.utils.diff import Comparator
from hsbriskevaluator.utils.apt_utils import AptUtils
from hsbriskevaluator.utils.file import get_data_dir

github_collector = GitHubRepoCollector()
apt_utils = AptUtils()
comparator = Comparator(github_collector, apt_utils)

diff_result = comparator.clone_and_compare("xz-utils")
with open(get_data_dir() / 'test_diff.json', 'w') as f:
    f.write(diff_result.model_dump_json(indent=2))
