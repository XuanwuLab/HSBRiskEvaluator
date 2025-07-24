from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List
from pydantic import BaseModel, Field
from copydetect import CopyDetector
from .file import is_binary, detect_language, get_data_dir
import subprocess
from .apt_utils import AptUtils
from ..collector.github_collector.data_collectors import GitHubDataCollector

logger = logging.getLogger(__name__)


class DiffFile(BaseModel):
    name: str = Field(description="Name of the file")
    similarity: float = Field(description="Similarity rate of the file (0-1)")


class DiffResult(BaseModel):
    same_project: bool = Field(
        description="Whether the files are from the same project"
    )
    dir1: str
    dir2: str
    url1: str
    url2: str
    files: List[DiffFile] = Field(
        description="List of files with their similarity rates"
    )


class Comparator:
    def __init__(
        self,
        github_collector: GitHubDataCollector = GitHubDataCollector(ThreadPoolExecutor(max_workers=1)),
        apt_utils: AptUtils = AptUtils(),
    ):
        self.github_collector = github_collector
        self.apt_utils = apt_utils
        pass

    def compare_dirs(self, dir1: str, dir2: str, url1: str, url2: str) -> DiffResult:
        detector = CopyDetector(test_dirs=[dir1], ref_dirs=[dir2])
        detector.run()
        similar_list = detector.get_copied_code_list()
        files = []
        for file in detector.test_files:
            if file.startswith(f"{dir1}/debian"):
                continue
            if file.startswith(f"{dir1}/.git"):
                continue
            if is_binary(file):
                continue
            if detect_language(file) in ("Groff", "Text only", "Unknown"):
                continue
            similarity = next(
                (x for x in similar_list if x[2] == file),
                [
                    0,
                ],
            )[0]
            files.append(DiffFile(name=file, similarity=similarity))
        files = sorted(files, key=lambda file: file.similarity, reverse=True)
        return DiffResult(
            same_project=(files[len(files) * 4 // 5].similarity >= 0.8),
            files=files,
            url1=url1,
            url2=url2,
            dir1=dir1,
            dir2=dir2,
        )

    def _clone_repo(self, repo_url: str, dest_dir: str) -> str:
        data_dir = get_data_dir() / "diff" / dest_dir
        if data_dir.exists():
            import shutil

            shutil.rmtree(data_dir)
        subprocess.run(["git", "clone", repo_url, str(data_dir)], check=True)
        return str(data_dir)

    def clone_and_compare(self, package: str) -> DiffResult:
        debian_url = self.apt_utils.get_package_git(package)
        #github_path = self._clone_repo(github_url, f"{package}/github")
        #debian_path = self._clone_repo(debian_url, f"{package}/debian")
        try:
            github_url = self.github_collector.search_repo(package).clone_url
            return DiffResult(
                same_project=True,
                files=[],
                url1=debian_url,
                url2=github_url,
                dir1="",
                dir2="",
            )
        except:
            return DiffResult(
                same_project=False,
                files=[],
                url1=debian_url,
                url2="",
                dir1="",
                dir2="",
                )
