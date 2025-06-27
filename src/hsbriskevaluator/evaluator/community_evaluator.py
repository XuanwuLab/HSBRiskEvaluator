import logging
from typing import Dict, Set
from statistics import mean
from hsbriskevaluator.evaluator.base import BaseEvaluator, CommunityEvalResult
from hsbriskevaluator.collector.repo_info import RepoInfo, PullRequest, Issue
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
        logger.info(f"Starting community evaluation for repository: {self.repo_info.repo_id}")
        
        try:
            # Analyze contributors and their roles
            direct_commit_users = self._count_direct_commit_users()
            pr_reviewers = self._count_pr_reviewers()
            required_reviewers = self._estimate_required_reviewers()
            
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
                direct_commit_users_count=direct_commit_users,
                pr_reviewers_count=pr_reviewers,
                required_reviewers_count=required_reviewers,
                prs_merged_without_discussion_count=prs_without_discussion,
                prs_with_inconsistent_description_count=prs_with_inconsistent_desc,
                avg_participants_per_issue=avg_participants_per_issue,
                avg_participants_per_pr=avg_participants_per_pr,
                community_activity_score=community_activity_score
            )
            
            logger.info(f"Community evaluation completed for {self.repo_info.repo_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error during community evaluation: {str(e)}")
            raise
    
    def _count_direct_commit_users(self) -> int:
        """Count users who can directly commit to main branch"""
        # In GitHub, this typically includes repository owners, admins, and maintainers
        # We approximate this by counting unique contributors
        # In a real implementation, this would require GitHub API calls to get repository permissions
        unique_contributors = set()
        
        # Add contributors from the contributor list
        for contributor in self.repo_info.contributor_list:
            unique_contributors.add(contributor.username)
        
        # Add PR authors and mergers (they likely have commit access)
        for pr in self.repo_info.pr_list:
            if pr.status == "merged" and pr.merger:
                unique_contributors.add(pr.merger)
        
        logger.debug(f"Estimated direct commit users: {len(unique_contributors)}")
        return len(unique_contributors)
    
    def _count_pr_reviewers(self) -> int:
        """Count users with PR review authority"""
        reviewers =  []
        
        for pr in self.repo_info.pr_list:
            if pr.reviewers:
                reviewers += pr.reviewers
        
        reviewers = list(set(reviewers))  # Remove duplicates
        logger.debug(f"PR reviewers count: {len(reviewers)}")
        return len(reviewers)
    
    def _estimate_required_reviewers(self) -> int:
        """Estimate minimum required reviewers for PR approval"""
        # Analyze merged PRs to estimate review requirements
        # This is a heuristic since we don't have direct access to branch protection rules
        
        if not self.repo_info.pr_list:
            return 0
        
        merged_prs = [pr for pr in self.repo_info.pr_list if pr.status == "merged"]
        
        if not merged_prs:
            return 0
        
        # Simple heuristic: if most PRs have approvers, assume 1 reviewer required
        # In practice, this would need more sophisticated analysis
        prs_with_approvers = sum(1 for pr in merged_prs if pr.approvers)
        approval_ratio = prs_with_approvers / len(merged_prs)
        
        if approval_ratio > 0.8:
            return 1
        elif approval_ratio > 0.5:
            return 1
        else:
            return 0
    
    def _count_prs_without_discussion(self) -> int:
        """Count PRs merged without any discussion"""
        prs_without_discussion = 0
        
        for pr in self.repo_info.pr_list:
            if pr.status == "merged":
                # Check if there are any comments or if it was immediately merged
                # This is a simplified check - in practice, we'd need PR comments data
                if pr.created_at == pr.merged_at or not pr.approvers:
                    prs_without_discussion += 1
        
        logger.debug(f"PRs merged without discussion: {prs_without_discussion}")
        return prs_without_discussion
    
    def _count_prs_with_inconsistent_description(self) -> int:
        """Count PRs where description is inconsistent with implementation using LLM"""
        inconsistent_count = 0
        
        # For demonstration, we'll analyze a subset of PRs due to LLM cost considerations
        prs_to_analyze = self.repo_info.pr_list[:10]  # Analyze first 10 PRs
        
        for pr in prs_to_analyze:
            if pr.status == "merged" and pr.title:
                try:
                    is_inconsistent = self._analyze_pr_consistency(pr)
                    if is_inconsistent:
                        inconsistent_count += 1
                except Exception as e:
                    logger.warning(f"Failed to analyze PR {pr.number} consistency: {str(e)}")
        
        logger.debug(f"PRs with inconsistent descriptions: {inconsistent_count}")
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
        
        Based on the title and timing, does this seem like a legitimate change or potentially suspicious?
        Consider factors like:
        - Vague or misleading titles
        - Very quick merge times (potential bypass of review)
        - Generic titles that could hide malicious changes
        
        Respond with a JSON object containing:
        - is_inconsistent: boolean (true if potentially suspicious/inconsistent)
        - confidence: float (0.0 to 1.0)
        - reason: string (brief explanation)
        """)
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    title=pr.title,
                    author=pr.author,
                    status=pr.status,
                    created_at=pr.created_at,
                    merged_at=pr.merged_at or "Not merged"
                )
            )
            
            # Parse LLM response - ensure it's a string
            response_content = response.content
            if isinstance(response_content, str):
                analysis = PRInconsistencyAnalysis.model_validate_json(response_content)
                return analysis.is_inconsistent and analysis.confidence > 0.7
            else:
                logger.warning(f"Unexpected response type from LLM for PR {pr.number}")
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
                participants.add(comment.username)
            
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
                participants.add(pr.approvers)
            if pr.merger:
                participants.add(pr.merger)
            
            participant_counts.append(len(participants))
        
        avg_participants = mean(participant_counts) if participant_counts else 0.0
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
        activity_score = (normalized_issue_activity * 0.3) + (normalized_pr_activity * 0.7)
        
        logger.debug(f"Community activity score: {activity_score:.3f}")
        return round(activity_score, 3)
