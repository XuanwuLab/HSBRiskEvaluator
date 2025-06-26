from hsbriskevaluator.collector.repo_info import RepoInfo
from pydantic import BaseModel, Field
class CommunityEvalResult(BaseModel):
    pass
class PayloadHiddenEvalResult(BaseModel):
    pass
class DependencyEvalResult(BaseModel):
    pass

class EvalResult(BaseModel):
    community_qualtiy: CommunityEvalResult
    payload_hidden_difficulty: PayloadHiddenEvalResult
    dependency: DependencyEvalResult

class BaseEvaulator():
    def __init__(self, repo_info:RepoInfo):
        self.repo_info = repo_info
    def perform():
        pass