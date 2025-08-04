from hsbriskevaluator.evaluator import EvalResult, PayloadHiddenEvalResult
from pydantic import BaseModel, Field
import numpy as np
import math 

class Score(BaseModel):
    url: str = Field(description="URL of the repository")
    pkt_name: str | list[str] = Field(
        description="package name list in package management system such as apt."
    )
    ci_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Continuous Integration score, from 0.0 (safe) to 1.0 (very dangerous)."
    )
    di_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Dependency Impact score, from 0.0 (safe) to 1.0 (very dangerous)."
    )
    cq_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Community Quality score, from 0.0 (safe) to 1.0 (very dangerous)."
    )
    pc_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Payload Concealment score, from 0.0 (safe) to 1.0 (very dangerous)."
    )
    hsb_score: float=Field(
        ge=0.0,
        le=1.0,
        description="Total score of the repository, from 0.0 (safe) to 1.0 (very dangerous)."
    )
def log(x):
    return math.log(x) if x>0 else 0
def apply(f: list[dict], name, values: list[float]):
    for fi, value in zip(f,values):
        fi[name]=value
def normalize(values: list[float], reverse=False) -> list[float]:
    p95 = np.percentile(values, 95).item()
    p5 = np.percentile(values, 5).item()
    if reverse:
        return list(map(lambda value : 1-min(max((value-p5)/(p95-p5),0),1),values))
    else:
        return list(map(lambda value : min(max((value-p5)/(p95-p5),0),1),values))
def calculate_DI_score(eval_results: list[EvalResult]) -> list[float]:
    self_priority = normalize([er.dependency.self_priority_required_count * 0.5+er.dependency.self_priority_important_count*0.3+er.dependency.self_priority_standard_count*0.2 for er in eval_results])
    self_essential = normalize([er.dependency.self_essential_count for er in eval_results])
    dependency_priority = normalize([er.dependency.dependency_priority_required_count * 0.5+er.dependency.dependency_priority_important_count*0.3+er.dependency.dependency_priority_standard_count*0.2 for er in eval_results])
    dependency_essential = normalize([er.dependency.dependency_essential_count for er in eval_results])
    scores = [sp*0.36 + se*0.24 + dp*0.24 + de*0.16 for sp, se, dp, de in zip(self_priority, self_essential, dependency_priority, dependency_essential)]
    return scores
def calculate_PC_score(eval_results: list[EvalResult]) -> list[float]:
    normalized_count = normalize([log(er.payload_hidden_difficulty.binary_files_count) for er in eval_results])
    def calc_score(count: float, result: PayloadHiddenEvalResult) -> float:
        return ((result.allows_binary_test_files+result.allows_binary_code_files+count)*3+(result.allows_binary_document_files+result.allows_binary_assets_files)+result.allows_other_binary_files*0.5)/11.5
    scores=[calc_score(count, er.payload_hidden_difficulty) for count, er in zip(normalized_count, eval_results)]
    return scores
def calculate_CI_score(eval_results: list[EvalResult]) -> list[float]:
    return [(1-int(er.ci.has_dependabot))*0.4+er.ci.dangerous_action_pin_ratio*0.3+er.ci.dangerous_action_provider_ratio*0.3 if er.ci.dangerous_action_pin_ratio >=-0.5 else 1 for er in eval_results]
def calculate_CQ_score(eval_results: list[EvalResult]) -> list[float]:
    cq_results = [er.community_quality for er in eval_results]
    stars = normalize([log(cq.stargazers_count) for cq in cq_results])
    forks = normalize([log(cq.forks_count) for cq in cq_results])
    watchers = normalize([log(cq.watchers_count) for cq in cq_results])
    users_count = normalize([log(cq.community_users_count) for cq in cq_results])
    avg_per_issue = normalize([cq.avg_participants_per_issue for cq in cq_results])
    avg_per_pr = normalize([cq.avg_participants_per_pr for cq in cq_results])
    popularity = [((stars+forks+watchers)+2*(users_count+avg_per_issue+avg_per_pr))/9 for stars, forks, watchers, users_count, avg_per_issue, avg_per_pr in zip(stars, forks, watchers,users_count, avg_per_issue, avg_per_pr)]

    def getavg(value: dict) -> float:
        sum=0
        for k, v in value.items():
            sum+=k*v
        return sum

    direct_commit_users = normalize([cq.direct_commit_users_count for cq in cq_results])
    required_approvals = normalize([getavg(cq.required_approvals_distribution) for cq in cq_results], reverse=True)
    review=[0.25*cq.direct_commits_ratio+0.2*dc+0.25*ra+0.15*cq.prs_merged_without_discussion_ratio+0.15*cq.prs_with_inconsistent_description_ratio for (dc, ra, cq) in zip(direct_commit_users, required_approvals, cq_results)]

    maintainers_count = normalize([cq.maintainers_count for cq in cq_results])
    approvers_count = normalize([cq.approvers_count for cq in cq_results])
    prs_needed_to_become_maintainer = normalize([getavg(cq.prs_needed_to_become_maintainer) for cq in cq_results],reverse=True)
    prs_needed_to_become_approver = normalize([getavg(cq.prs_needed_to_become_approver) for cq in cq_results],reverse=True)

    privelege=[0.2*mc+0.2*ac+0.3*prm+0.3*pra for (mc, ac, prm, pra) in zip(maintainers_count, approvers_count, prs_needed_to_become_maintainer, prs_needed_to_become_approver)]
  
    scores = [0.2*po+0.4*r+0.4*pri for po, r, pri in zip(popularity, review, privelege)]
    return scores

def calculate_all_score(eval_results: list[EvalResult]) -> list[Score]:
    DI_scores=calculate_DI_score(eval_results)
    PC_scores=calculate_PC_score(eval_results)
    CI_scores=calculate_CI_score(eval_results)
    CQ_scores=calculate_CQ_score(eval_results)
    scores = [Score(di_score=di, pc_score=pc, ci_score=ci, cq_score=cq, hsb_score=di*0.3+pc*0.2+ci*0.3+cq*0.2, url=er.url, pkt_name=er.pkt_name) for di, pc, ci, cq, er in zip(DI_scores, PC_scores, CI_scores, CQ_scores, eval_results)]
    return scores
