#! /usr/bin/env python3

from hsbriskevaluator.collector import GitHubRepoCollector
from hsbriskevaluator.utils.diff import Comparator
from hsbriskevaluator.utils.apt_utils import AptUtils

github_collector = GitHubRepoCollector()
apt_utils = AptUtils()
comparator = Comparator(github_collector, apt_utils)
print(comparator.clone_and_compare("xz-utils"))
