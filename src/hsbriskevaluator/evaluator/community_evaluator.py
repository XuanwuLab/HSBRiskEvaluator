import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Set
from statistics import mean, geometric_mean
from hsbriskevaluator.evaluator.base import BaseEvaluator, CommunityEvalResult
from hsbriskevaluator.collector.repo_info import RepoInfo, PullRequest, Issue, Commit, User
from hsbriskevaluator.utils.llm import get_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PRInconsistencyAnalysis(BaseModel):
    """Analysis result for PR description vs implementation consistency"""
    is_inconsistent: bool
    confidence: float
    reason: str


class CommunityEvaluator(BaseEvaluator):
    """Evaluator for Community Quality metrics"""

    def __init__(self, repo_info: RepoInfo, llm_model_name: str = "anthropic/claude-3.5-sonnet"):
        super().__init__(repo_info)
        self.llm = get_model(llm_model_name)

    def evaluate(self) -> CommunityEvalResult:
        """Evaluate community quality metrics"""
        logger.info(
            f"Starting community evaluation for repository: {self.repo_info.repo_id}")

        try:
            # Analyze contributors and their roles

            direct_commits = self._count_direct_commits()
            direct_commit_users = self._count_direct_commit_users()
            community_users = self._count_community_users()
            maintainers = self._count_maintainers()
            pr_reviewers = self._count_pr_reviewers()
            required_reviewers = self._analyze_required_reviewers()
            estimated_prs_to_become_maintainer = self._estimated_prs_to_become_maintainer()
            estimated_prs_to_become_reviewer = self._estimated_prs_to_become_reviewer()
            # Analyze PR and issue discussions
            prs_without_discussion = self._count_prs_without_discussion()
            prs_with_inconsistent_desc = self._count_prs_with_inconsistent_description()

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
                direct_commits=direct_commits,
                direct_commit_users_count=direct_commit_users,
                maintainers_count=maintainers,
                pr_reviewers_count=pr_reviewers,
                required_reviewers_distribution=required_reviewers,
                estimated_prs_to_become_maintainer=estimated_prs_to_become_maintainer,
                estimated_prs_to_become_reviewer=estimated_prs_to_become_reviewer,
                prs_merged_without_discussion_count=prs_without_discussion,
                prs_with_inconsistent_description_count=prs_with_inconsistent_desc,
                avg_participants_per_issue=avg_participants_per_issue,
                avg_participants_per_pr=avg_participants_per_pr,
                community_activity_score=community_activity_score,

            )

            logger.info(
                f"Community evaluation completed for {self.repo_info.repo_id}")
            return result

        except Exception as e:
            logger.error(f"Error during community evaluation: {str(e)}")
            raise

    def _list_community_users(self) -> list[User]:
        unique_users = set()
        # map(lambda commit: unique_users.add(commit.author), self.repo_info.commit_list)
        for commit in self.repo_info.commit_list:
            unique_users.add(commit.author)

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
        logger.info(f"Unique users in community: {unique_users}")
        return list(unique_users)

    def _count_community_users(self) -> int:
        # not counting bots here
        return len(list(filter(lambda user: user.type == "User", self._list_community_users())))

    def _list_direct_commits(self) -> list[Commit]:
        return list(filter(lambda commit: len(commit.pull_numbers) == 0, self.repo_info.commit_list))

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
        for pr in self.repo_info.pr_list:
            if pr.status == "merged":
                unique_contributors.add(pr.merger)

        logger.debug(f"Maintainers: {unique_contributors}")
        return list(unique_contributors)

    def _count_direct_commits(self) -> int:
        return len(self._list_direct_commits())

    def _count_direct_commit_users(self) -> int:
        return len(list(filter(lambda user: user.type == "User", self._list_direct_commit_users())))

    def _count_maintainers(self) -> int:
        return len(list(filter(lambda user: user.type == "User", self._list_maintainers())))

    def _list_reviewers(self) -> Set[User]:
        reviewers = set(self._list_maintainers())
        for pr in self.repo_info.pr_list:
            if pr.reviewers:
                for reviewer in pr.reviewers:
                    reviewers.add(reviewer)
        logger.debug(f"PR reviewers: {reviewers}")
        return reviewers

    def _count_pr_reviewers(self) -> int:
        """Count users with PR review authority"""
        return len(list(filter(lambda user: user.type == "User", self._list_reviewers())))

    def _estimated_prs_to_become_maintainer(self) -> float:
        activities = []
        maintainers = self._list_maintainers()
        direct_commits = self._list_direct_commits()

        for user in maintainers:
            if user.type != "User":
                continue
            granted_to_maintainer_date = datetime.now(timezone.utc)

            direct_commits = list(
                filter(lambda commit: commit.author == user, direct_commits))
            if len(direct_commits):
                granted_to_maintainer_date = datetime.fromisoformat(
                    direct_commits[-1].timestamp)

            in_merger_list = list(filter(
                lambda pr: pr.merger == user, self.repo_info.pr_list))
            if in_merger_list != []:
                granted_to_maintainer_date = min(
                    granted_to_maintainer_date, datetime.fromisoformat(in_merger_list[-1].created_at))

            def in_pr(pr: PullRequest) -> bool:
                return (pr.author == user or pr.merger == user or user in pr.reviewers) and datetime.fromisoformat(pr.created_at) < granted_to_maintainer_date
            early_prs = list(filter(in_pr, self.repo_info.pr_list))
            if early_prs == []:
                logger.debug(
                    f"User {user.username} has no PRs before granted to maintainer")
            else:
                logger.debug(
                    f"User {user.username} contributed to {len(early_prs)} PRs in {granted_to_maintainer_date - datetime.fromisoformat(early_prs[-1].created_at)} before becoming a maintainer")
            activities.append(len(early_prs))

        logger.debug(
            f"Average time to become maintainer: {mean(activities) if len(activities) else -1.0} PRs")
        return mean(activities) if len(activities) else -1.0

    def _estimated_prs_to_become_reviewer(self) -> float:
        activities = []
        reviewers = self._list_reviewers()
        direct_commits = self._list_direct_commits()
        for user in reviewers:
            if user.type != "User":
                continue
            granted_to_reviewer_date = datetime.now(timezone.utc)
            direct_commits = list(
                filter(lambda commit: commit.author == user, direct_commits))
            if len(direct_commits):
                granted_to_reviewer_date = datetime.fromisoformat(
                    direct_commits[-1].timestamp)

            in_reviewer_list = list(filter(
                lambda pr: user in pr.reviewers or pr.merger == user, self.repo_info.pr_list))
            if in_reviewer_list != []:
                granted_to_reviewer_date = min(
                    granted_to_reviewer_date, datetime.fromisoformat(in_reviewer_list[-1].created_at))

            def in_pr(pr: PullRequest) -> bool:
                return (pr.author == user or pr.merger == user or user in pr.reviewers) and datetime.fromisoformat(pr.created_at) < granted_to_reviewer_date
            early_prs = list(filter(in_pr, self.repo_info.pr_list))
            if early_prs == []:
                logger.debug(
                    f"User {user.username} has no PRs before granted to reviewer")
            else:
                logger.debug(
                    f"User {user.username} contributed to {len(early_prs)} PRs in {granted_to_reviewer_date - datetime.fromisoformat(early_prs[-1].created_at)} before becoming a reviewer")
            activities.append(len(early_prs))
        logger.debug(
            f"Average time to become reviewer: {mean(activities) if len(activities) else 0.0} PRs")
        return mean(activities) if len(activities) else 0.0

    def _analyze_required_reviewers(self) -> dict[int, float]:
        """Estimate minimum required reviewers for PR approval"""
        # Analyze merged PRs to estimate review requirements
        # -1 means direct commits to main branch

        if not self.repo_info.pr_list:
            return {}

        merged_prs = [
            pr for pr in self.repo_info.pr_list if pr.status == "merged"]

        if not merged_prs:
            return {}

        prs_with_approvers = {}
        prs_with_approvers[-1] = self._count_direct_commits()
        for pr in merged_prs:
            approvers_count = len(pr.approvers)
            if approvers_count not in prs_with_approvers.keys():
                prs_with_approvers[approvers_count] = 0
            prs_with_approvers[approvers_count] += 1
        pr_count = sum(prs_with_approvers.values())
        for key in prs_with_approvers.keys():
            prs_with_approvers[key] /= pr_count
            len(merged_prs)
        return prs_with_approvers

    def _count_prs_without_discussion(self) -> int:
        """Count PRs merged without any discussion"""
        prs_without_discussion = 0

        for pr in self.repo_info.pr_list:
            if pr.status == "merged":
                # Check if there are any comments or if it was immediately merged
                # This is a simplified check - in practice, we'd need PR comments data
                if len(pr.comments) == 0 and not pr.approvers:
                    prs_without_discussion += 1

        logger.debug(
            f"PRs merged without discussion: {prs_without_discussion}")
        return prs_without_discussion

    def _count_prs_with_inconsistent_description(self) -> int:
        """Count PRs where description is inconsistent with implementation using LLM"""
        inconsistent_count = 0

        # For demonstration, we'll analyze a subset of PRs due to LLM cost considerations
        prs_to_analyze = list(filter(lambda pr: pr.status == "merged", self.repo_info.pr_list))[
            :10]  # Analyze first 10 PRs

        for pr in prs_to_analyze:
            if pr.status == "merged" and pr.title:
                try:
                    is_inconsistent = self._analyze_pr_consistency(pr)
                    if is_inconsistent:
                        inconsistent_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to analyze PR {pr.number} consistency: {str(e)}")

        logger.debug(
            f"PRs with inconsistent descriptions: {inconsistent_count}")
        return inconsistent_count

    def _analyze_pr_consistency(self, pr: PullRequest) -> bool:
        """Use LLM to analyze if PR description matches its likely implementation"""
        prompt = ChatPromptTemplate.from_template("""
        Analyze if this pull request title and description seem consistent with what the implementation likely does.
        
        PR Title: {title}
        PR Author: {author}
        PR Status: {status}
        Created: {created_at}
        Merged: {merged_at}
        Changed files: {changed_files}
        PR Body: {body}
        
        Based on the title and timing, does this seem like a legitimate change or potentially suspicious?
        Consider factors like:
        - Vague or misleading titles
        - Very quick merge times (potential bypass of review)
        - Generic titles that could hide malicious changes
        
        Respond with a JSON object containing:
        - is_inconsistent: boolean (true if potentially suspicious/inconsistent)
        - confidence: float (0.0 to 1.0)
        - reason: string (brief explanation)

        Don't include any other text, just the JSON response.
        """)

        try:
            response = self.llm.invoke(
                prompt.format(
                    title=pr.title,
                    author=pr.author,
                    status=pr.status,
                    created_at=pr.created_at,
                    merged_at=pr.merged_at or "Not merged",
                    changed_files=pr.changed_files,
                    body=pr.body
                )
            )
            # Parse LLM response - ensure it's a string
            response_content = response.content
            if isinstance(response_content, str):
                analysis = PRInconsistencyAnalysis.model_validate_json(
                    response_content)
                return analysis.is_inconsistent and analysis.confidence > 0.7
            else:
                logger.warning(
                    f"Unexpected response type from LLM for PR {pr.number}")
                return False

        except Exception as e:
            logger.warning(f"LLM analysis failed for PR {pr.number}: {str(e)}")
            return False

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

        avg_participants = mean(
            participant_counts) if participant_counts else 0.0
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

        avg_participants = mean(
            participant_counts) if participant_counts else 0.0
        logger.debug(f"Average participants per PR: {avg_participants:.2f}")
        return avg_participants

    def _calculate_community_activity_score(self, avg_issue_participants: float, avg_pr_participants: float) -> float:
        """Calculate overall community activity score (0-1)"""
        # Normalize and combine metrics
        # Higher participation indicates better community engagement

        # Normalize participants (assuming 5+ participants is very active)
        normalized_issue_activity = min(avg_issue_participants / 5.0, 1.0)
        normalized_pr_activity = min(avg_pr_participants / 3.0, 1.0)

        # Weight PR activity higher as it's more critical for security
        activity_score = (normalized_issue_activity * 0.3) + \
            (normalized_pr_activity * 0.7)

        logger.debug(f"Community activity score: {activity_score:.3f}")
        return round(activity_score, 3)
