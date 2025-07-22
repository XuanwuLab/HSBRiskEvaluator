from dotenv import load_dotenv
load_dotenv()
from hsbriskevaluator.collector import GitHubRepoCollector
from hsbriskevaluator.utils.diff import Comparator
from hsbriskevaluator.utils.apt_utils import AptUtils
from hsbriskevaluator.utils.file import get_data_dir

github_collector = GitHubRepoCollector()
apt_utils = AptUtils()
comparator = Comparator(github_collector, apt_utils)

diff_result=comparator.clone_and_compare("xz-utils")
