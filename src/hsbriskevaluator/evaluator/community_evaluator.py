import logging
import asyncio
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Set, List, Optional
from statistics import mean, geometric_mean
from hsbriskevaluator.evaluator.base import BaseEvaluator, CommunityEvalResult
from hsbriskevaluator.collector.repo_info import (
    RepoInfo,
    PullRequest,
    Issue,
    Commit,
    User,
)
from hsbriskevaluator.evaluator.settings import EvaluatorSettings
from hsbriskevaluator.utils.llm import get_async_instructor_client, call_llm_with_client
from hsbriskevaluator.utils.prompt import (
    PR_CONSISTENCY_ANALYSIS_PROMPT,
    PR_CONSISTENCY_ANALYSIS_MODEL_ID,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PRInconsistencyAnalysis(BaseModel):
    """Analysis result for PR description vs implementation consistency"""

    is_inconsistent: bool
    confidence: float
    reason: str


class CommunityEvaluator(BaseEvaluator):
    """Evaluator for Community Quality metrics"""

    def __init__(
        self,
        repo_info: RepoInfo,
        settings: Optional[EvaluatorSettings]=None,
    ):
        super().__init__(repo_info)
        if settings is None:
            settings = EvaluatorSettings()
        self.settings = settings
        self.llm_model_name = settings.pr_consistency_analysis_model_id
        self.max_concurrency = settings.community_max_concurrency
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self.client = get_async_instructor_client()

    async def evaluate(self) -> CommunityEvalResult:
        """Evaluate community quality metrics"""
        logger.info(
            f"Starting community evaluation for repository: {self.repo_info.repo_id}"
        )

        try:
            # Analyze contributors and their roles
            direct_commits_ratio = self._count_direct_commits_ratio()
            direct_commit_users = self._count_direct_commit_users()
            community_users = self._count_community_users()
            maintainers = self._count_maintainers()
            pr_reviewers = self._count_pr_reviewers()
            required_reviewers = self._analyze_required_reviewers()
            prs_needed_to_become_maintainer = self._prs_needed_to_become_maintainer()
            prs_needed_to_become_reviewer = self._prs_needed_to_become_reviewer()

            # Analyze PR and issue discussions
            prs_without_discussion_ratio = self._count_prs_without_discussion_ratio()
            prs_with_inconsistent_desc_ratio = (
                await self._count_prs_with_inconsistent_description_ratio()
            )

            # Calculate community activity metrics
            avg_participants_per_issue = self._calculate_avg_participants_per_issue()
            avg_participants_per_pr = self._calculate_avg_participants_per_pr()

            # Calculate overall community activity score
            community_activity_score = self._calculate_community_activity_score(
                avg_participants_per_issue, avg_participants_per_pr
            )

            result = CommunityEvalResult(
                stargazers_count=self.repo_info.basic_info.stargazers_count,
                watchers_count=self.repo_info.basic_info.watchers_count,
                forks_count=self.repo_info.basic_info.forks_count,
                community_users_count=community_users,
                direct_commits_ratio=direct_commits_ratio,
                direct_commit_users_count=direct_commit_users,
                maintainers_count=maintainers,
                pr_reviewers_count=pr_reviewers,
                required_reviewers_distribution=required_reviewers,
                prs_needed_to_become_maintainer=prs_needed_to_become_maintainer,
                prs_needed_to_become_reviewer=prs_needed_to_become_reviewer,
                prs_merged_without_discussion_ratio=prs_without_discussion_ratio,
                prs_with_inconsistent_description_ratio=prs_with_inconsistent_desc_ratio,
                avg_participants_per_issue=avg_participants_per_issue,
                avg_participants_per_pr=avg_participants_per_pr,
                community_activity_score=community_activity_score,
            )

            logger.info(f"Community evaluation completed for {self.repo_info.repo_id}")
            return result

        except Exception as e:
            logger.error(f"Error during community evaluation: {str(e)}")
            raise

    def _list_community_users(self) -> list[User]:
        unique_users = set()

        def add_issue_users(issue: Issue):
            unique_users.add(issue.author)
            for comment in issue.comments:
                unique_users.add(comment.user)

        def add_pr_users(pr: PullRequest):
            unique_users.add(pr.author)
            if pr.status == "merged":
                unique_users.add(pr.merger)
            for comment in pr.comments:
                unique_users.add(comment.user)
            for user in pr.approvers:
                unique_users.add(user)
            for user in pr.reviewers:
                unique_users.add(user)

        for issue in self.repo_info.issue_list:
            add_issue_users(issue)
        for pr in self.repo_info.pr_list:
            add_pr_users(pr)
        logger.debug(f"Unique users in community: {unique_users}")
        return list(unique_users)

    def _count_community_users(self) -> int:
        # not counting bots here
        return len(
            list(filter(lambda user: user.type == "User", self._list_community_users()))
        )

    def _list_direct_commits(self) -> list[Commit]:
        def is_direct_commit(commit: Commit) -> bool:
            if commit.pull_numbers:
                return False
            if f"{self.repo_info.basic_info.url}/pull/" in commit.message:
                return False
            pattern = r'.*?(close|merge).*?#\d+.*'
            for line in commit.message.splitlines():
                if re.search(pattern, line, re.IGNORECASE):
                    return False 
            return True
        return list(
            filter(
                is_direct_commit, self.repo_info.commit_list
            )
        )

    def _count_direct_commits_ratio(self) -> float:
        return len(self._list_direct_commits())/len(self.repo_info.commit_list) if self.repo_info.commit_list else 0.0

    def _list_direct_commit_users(self) -> list[User]:
        unique_contributors = set()
        for commit in self._list_direct_commits():
            unique_contributors.add(commit.author)
        return list(unique_contributors)

    def _list_maintainers(self) -> list[User]:
        """Count users who can directly commit to main branch"""
        # In GitHub, this typically includes repository owners, admins, and maintainers
        # We approximate this by counting the commits that are not part of any PR
        unique_contributors = set(self._list_direct_commit_users())
        for pr in self.repo_info.pr_without_comment_list:
            if pr.status == "merged":
                unique_contributors.add(pr.merger)

        logger.debug(f"Maintainers: {unique_contributors}")
        return list(unique_contributors)


    def _count_direct_commit_users(self) -> int:
        return len(
            list(
                filter(
                    lambda user: user.type == "User", self._list_direct_commit_users()
                )
            )
        )

    def _count_maintainers(self) -> int:
        return len(
            list(filter(lambda user: user.type == "User", self._list_maintainers()))
        )

    def _list_reviewers(self) -> Set[User]:
        reviewers = set(self._list_maintainers())
        for pr in self.repo_info.pr_without_comment_list:
            if pr.reviewers:
                for reviewer in pr.reviewers:
                    reviewers.add(reviewer)
        logger.debug(f"PR reviewers: {reviewers}")
        return reviewers

    def _count_pr_reviewers(self) -> int:
        """Count users with PR review authority"""
        return len(
            list(filter(lambda user: user.type == "User", self._list_reviewers()))
        )

    def _prs_needed_to_become_maintainer(self) -> Dict[int, int]:
        activities = {}
        maintainers = self._list_maintainers()
        direct_commits = self._list_direct_commits()

        for user in maintainers:
            if user.type != "User":
                continue
            granted_to_maintainer_date = datetime.now(timezone.utc)

            direct_commits = list(
                filter(lambda commit: commit.author == user, direct_commits)
            )
            if len(direct_commits):
                granted_to_maintainer_date = datetime.fromisoformat(
                    direct_commits[-1].timestamp
                )

            in_merger_list = list(
                filter(lambda pr: pr.merger == user, self.repo_info.pr_without_comment_list)
            )
            if in_merger_list != []:
                granted_to_maintainer_date = min(
                    granted_to_maintainer_date,
                    datetime.fromisoformat(in_merger_list[-1].created_at),
                )

            def in_pr(pr: PullRequest) -> bool:
                return (
                    pr.author == user or pr.merger == user or user in pr.reviewers
                ) and datetime.fromisoformat(pr.created_at) < granted_to_maintainer_date

            early_prs = list(filter(in_pr, self.repo_info.pr_without_comment_list))
            if early_prs == []:
                logger.debug(
                    f"User {user.username} has no PRs before granted to maintainer"
                )
            else:
                logger.debug(
                    f"User {user.username} contributed to {len(early_prs)} PRs in {granted_to_maintainer_date - datetime.fromisoformat(early_prs[-1].created_at)} before becoming a maintainer"
                )
            if len(early_prs) not in activities:
                activities[len(early_prs)]=0
            activities[len(early_prs)]+=1
        logger.debug(
            f"Average time to become maintainer: {mean(activities) if len(activities) else -1.0} PRs"
        )
        return activities

    def _prs_needed_to_become_reviewer(self) -> Dict[int,int]:
        activities = {}
        reviewers = self._list_reviewers()
        direct_commits = self._list_direct_commits()
        for user in reviewers:
            if user.type != "User":
                continue
            granted_to_reviewer_date = datetime.now(timezone.utc)
            direct_commits = list(
                filter(lambda commit: commit.author == user, direct_commits)
            )
            if len(direct_commits):
                granted_to_reviewer_date = datetime.fromisoformat(
                    direct_commits[-1].timestamp
                )

            in_reviewer_list = list(
                filter(
                    lambda pr: user in pr.reviewers or pr.merger == user,
                    self.repo_info.pr_without_comment_list,
                )
            )
            if in_reviewer_list != []:
                granted_to_reviewer_date = min(
                    granted_to_reviewer_date,
                    datetime.fromisoformat(in_reviewer_list[-1].created_at),
                )

            def in_pr(pr: PullRequest) -> bool:
                return (
                    pr.author == user or pr.merger == user or user in pr.reviewers
                ) and datetime.fromisoformat(pr.created_at) < granted_to_reviewer_date

            early_prs = list(filter(in_pr, self.repo_info.pr_without_comment_list))
            if early_prs == []:
                logger.debug(
                    f"User {user.username} has no PRs before granted to reviewer"
                )
            else:
                logger.debug(
                    f"User {user.username} contributed to {len(early_prs)} PRs in {granted_to_reviewer_date - datetime.fromisoformat(early_prs[-1].created_at)} before becoming a reviewer"
                )
            if len(early_prs) not in activities:
                activities[len(early_prs)]=0
            activities[len(early_prs)]+=1
        logger.debug(
            f"Average time to become reviewer: {mean(activities) if len(activities) else 0.0} PRs"
        )
        return activities

    def _analyze_required_reviewers(self) -> dict[int, float]:
        """Estimate minimum required reviewers for PR approval"""
        # Analyze merged PRs to estimate review requirements
        # -1 means direct commits to main branch

        if not self.repo_info.pr_without_comment_list:
            return {}

        prs = [pr for pr in self.repo_info.pr_without_comment_list if pr.status == "merged" or len(pr.approvers)>0]

        if not prs:
            return {}

        prs_with_approvers = {}
        for pr in prs:
            approvers_count = len(set(pr.approvers))
            if approvers_count not in prs_with_approvers.keys():
                prs_with_approvers[approvers_count] = 0
            prs_with_approvers[approvers_count] += 1
        for key in prs_with_approvers.keys():
            prs_with_approvers[key] /= len(prs)
        return prs_with_approvers

    def _count_prs_without_discussion_ratio(self) -> float:
        """Count PRs merged without any discussion"""
        prs_without_discussion = 0
        total_prs=0

        for pr in self.repo_info.pr_list:
            if pr.status == "merged":
                # Check if there are any comments or if it was immediately merged
                # This is a simplified check - in practice, we'd need PR comments data
                if len(pr.comments) == 0 and not pr.approvers:
                    prs_without_discussion += 1
                total_prs+=1

        logger.debug(f"PRs merged without discussion: {prs_without_discussion}")
        return prs_without_discussion/total_prs if total_prs>0 else 0.0

    async def _count_prs_with_inconsistent_description_ratio(self) -> float:
        """Count PRs where description is inconsistent with implementation using LLM"""
        inconsistent_count = 0

        # For demonstration, we'll analyze a subset of PRs due to LLM cost considerations
        prs_to_analyze = list(
            filter(lambda pr: pr.status == "merged", self.repo_info.pr_list)
        )[
            :self.settings.prs_to_analyze_limit
        ]  # Analyze first N PRs
        
        if not prs_to_analyze:
            return 0

        # Analyze PRs concurrently
        tasks = []
        batch_size = self.settings.pr_batch_size
        page_num = (len(prs_to_analyze)-1)//batch_size+1
        for page in range(page_num):
            task = self._analyze_pr_consistency(prs_to_analyze[batch_size*page:batch_size*(page+1)])
            tasks.append(task)


        if not tasks:
            return 0

        # Execute all tasks concurrently with rate limiting

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count inconsistent PRs
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Failed to analyze PR {list(map(lambda pr: pr.number,prs_to_analyze[i*batch_size:(i+1)*batch_size]))} consistency: {str(result)}"
                )
            else:
                inconsistent_count += 0
                for analysis in result:
                    if analysis.is_inconsistent and analysis.confidence > self.settings.pr_consistency_confidence_threshold:
                        inconsistent_count+=1
                

        logger.debug(f"PRs with inconsistent descriptions: {inconsistent_count}")
        return inconsistent_count / len(prs_to_analyze) if prs_to_analyze else 0.0

    async def _analyze_pr_consistency(self, prs: List[PullRequest]) -> List[PRInconsistencyAnalysis]:
        """Use LLM to analyze if PR description matches its likely implementation"""

        logger.info(f"Start analysis for {len(prs)} PRs")
        logger.debug(f"PR list: {list(map(lambda pr: pr.number, prs))}")
        async with self._semaphore:  # Rate limiting
            class AnalysisResult(BaseModel):
                results: List[PRInconsistencyAnalysis] = Field(
                    description="List of detected consistency details.",
                    max_length=len(prs),
                    min_length=len(prs)
                )
            def simplify_pr(pr:PullRequest)->Dict:
                return {'title': pr.title, 'author':pr.author, 'status': pr.status, 'created_at':pr.created_at, 'merged_at':pr.merged_at or 'Not merged', 'changed_files':pr.changed_files,'body':pr.body}

            simplified_prs = list(map(simplify_pr, prs))
            messages = [
                {"role": "system", "content": PR_CONSISTENCY_ANALYSIS_PROMPT},
                {"role": "user", "content": f"Pull requests: {simplified_prs}"},
            ]
            try:
                analysis = await call_llm_with_client(
                    client=self.client,
                    model_id=self.llm_model_name,
                    messages=messages,
                    response_model=AnalysisResult,
                )
                return analysis.results
            except Exception as e:
                logger.warning(f"LLM analysis failed for PR {list(map(lambda pr: pr.number, prs))}: {str(e)}")
                return [PRInconsistencyAnalysis(in_consistent=False, confidence=0.0, reason="LLM analysis failed") for _ in prs]

    def _calculate_avg_participants_per_issue(self) -> float:
        """Calculate average number of participants per issue"""
        if not self.repo_info.issue_list:
            return 0.0

        participant_counts = []

        for issue in self.repo_info.issue_list:
            participants = set()
            participants.add(issue.author)

            for comment in issue.comments:
                participants.add(comment.user)

            participant_counts.append(len(participants))

        avg_participants = mean(participant_counts) if participant_counts else 0.0
        logger.debug(f"Average participants per issue: {avg_participants:.2f}")
        return avg_participants

    def _calculate_avg_participants_per_pr(self) -> float:
        """Calculate average number of participants per PR"""
        if not self.repo_info.pr_list:
            return 0.0

        participant_counts = []

        for pr in self.repo_info.pr_list:
            participants = set()
            participants.add(pr.author)

            if pr.approvers:
                for approver in pr.approvers:
                    participants.add(approver)
            if pr.merger:
                participants.add(pr.merger)

            participant_counts.append(len(participants))

        avg_participants = mean(participant_counts) if participant_counts else 0.0
        logger.debug(f"Average participants per PR: {avg_participants:.2f}")
        return avg_participants

    def _calculate_community_activity_score(
        self, avg_issue_participants: float, avg_pr_participants: float
    ) -> float:
        """Calculate overall community activity score (0-1)"""
        # Normalize and combine metrics
        # Higher participation indicates better community engagement

        # Normalize participants (assuming 5+ participants is very active)
        normalized_issue_activity = min(avg_issue_participants / self.settings.max_issue_participants_for_normalization, 1.0)
        normalized_pr_activity = min(avg_pr_participants / self.settings.max_pr_participants_for_normalization, 1.0)

        # Weight PR activity higher as it's more critical for security
        activity_score = (normalized_issue_activity * self.settings.issue_activity_weight) + (
            normalized_pr_activity * self.settings.pr_activity_weight
        )

        logger.debug(f"Community activity score: {activity_score:.3f}")
        return round(activity_score, 3)
