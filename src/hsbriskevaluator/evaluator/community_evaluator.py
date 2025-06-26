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


class IssueSemanticAnalysis(BaseModel):
    """Analysis result for issue semantic content"""
    is_legitimate: bool
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
            # Statistical analysis on repo_info data
            pr_reviewers = self._count_pr_reviewers()
            required_reviewers = self._estimate_required_reviewers()
            avg_participants_per_issue = self._calculate_avg_participants_per_issue()
            avg_participants_per_pr = self._calculate_avg_participants_per_pr()
            prs_without_discussion = self._count_prs_without_discussion()
            
            # Methods that need additional data (TODO comments added)
            direct_commit_users = self._count_direct_commit_users()
            
            # LLM-based semantic analysis
            prs_with_inconsistent_desc = self._count_prs_with_inconsistent_description()
            
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
    
    # ========== METHODS REQUIRING ADDITIONAL DATA (TODO) ==========
    
    def _count_direct_commit_users(self) -> int:
        """Count users who can directly commit to main branch"""
        # TODO: This evaluation requires GitHub API access to get repository permissions
        # Need: Repository collaborators with push/admin permissions from GitHub API
        # Current implementation is a rough approximation based on available data
        
        unique_contributors = set()
        
        # Add contributors from the contributor list
        for contributor in self.repo_info.contributor_list:
            unique_contributors.add(contributor.username)
        
        # Add PR mergers (they likely have commit access)
        for pr in self.repo_info.pr_list:
            if pr.status == "merged" and pr.merger:
                unique_contributors.add(pr.merger)
        
        logger.debug(f"Estimated direct commit users (approximation): {len(unique_contributors)}")
        return len(unique_contributors)
    
    # ========== STATISTICAL ANALYSIS ON REPO_INFO DATA ==========
    
    def _count_pr_reviewers(self) -> int:
        """Count users with PR review authority - Statistical analysis"""
        reviewers = []
        
        for pr in self.repo_info.pr_list:
            if pr.reviewers:
                reviewers.extend(pr.reviewers)
        
        unique_reviewers = list(set(reviewers))  # Remove duplicates
        logger.debug(f"PR reviewers count: {len(unique_reviewers)}")
        return len(unique_reviewers)
    
    def _estimate_required_reviewers(self) -> int:
        """Estimate minimum required reviewers for PR approval - Statistical analysis"""
        if not self.repo_info.pr_list:
            return 0
        
        merged_prs = [pr for pr in self.repo_info.pr_list if pr.status == "merged"]
        
        if not merged_prs:
            return 0
        
        # Statistical heuristic: analyze approval patterns
        prs_with_approvers = sum(1 for pr in merged_prs if pr.approvers)
        approval_ratio = prs_with_approvers / len(merged_prs)
        
        # Simple heuristic based on approval patterns
        if approval_ratio > 0.8:
            return 1
        elif approval_ratio > 0.5:
            return 1
        else:
            return 0
    
    def _count_prs_without_discussion(self) -> int:
        """Count PRs merged without any discussion - Statistical analysis"""
        prs_without_discussion = 0
        
        for pr in self.repo_info.pr_list:
            if pr.status == "merged":
                # Statistical check: no approvers and same creation/merge time suggests no discussion
                has_no_approvers = not pr.approvers
                same_time_merge = pr.created_at == pr.merged_at
                
                if has_no_approvers or same_time_merge:
                    prs_without_discussion += 1
        
        logger.debug(f"PRs merged without discussion: {prs_without_discussion}")
        return prs_without_discussion
    
    def _calculate_avg_participants_per_issue(self) -> float:
        """Calculate average number of participants per issue - Statistical analysis"""
        if not self.repo_info.issue_list:
            return 0.0
        
        participant_counts = []
        
        for issue in self.repo_info.issue_list:
            participants = set()
            participants.add(issue.author)
            
            # Count unique commenters
            for comment in issue.comments:
                participants.add(comment.username)
            
            participant_counts.append(len(participants))
        
        avg_participants = mean(participant_counts) if participant_counts else 0.0
        logger.debug(f"Average participants per issue: {avg_participants:.2f}")
        return avg_participants
    
    def _calculate_avg_participants_per_pr(self) -> float:
        """Calculate average number of participants per PR - Statistical analysis"""
        if not self.repo_info.pr_list:
            return 0.0
        
        participant_counts = []
        
        for pr in self.repo_info.pr_list:
            participants = set()
            participants.add(pr.author)
            
            # Add reviewers and approvers
            if pr.reviewers:
                participants.update(pr.reviewers)
            if pr.approvers:
                participants.update(pr.approvers)
            if pr.merger:
                participants.add(pr.merger)
            
            participant_counts.append(len(participants))
        
        avg_participants = mean(participant_counts) if participant_counts else 0.0
        logger.debug(f"Average participants per PR: {avg_participants:.2f}")
        return avg_participants
    
    def _calculate_community_activity_score(self, avg_issue_participants: float, avg_pr_participants: float) -> float:
        """Calculate overall community activity score (0-1) - Statistical calculation"""
        # Normalize and combine metrics
        # Higher participation indicates better community engagement
        
        # Normalize participants (assuming 5+ participants is very active for issues, 3+ for PRs)
        normalized_issue_activity = min(avg_issue_participants / 5.0, 1.0)
        normalized_pr_activity = min(avg_pr_participants / 3.0, 1.0)
        
        # Weight PR activity higher as it's more critical for security
        activity_score = (normalized_issue_activity * 0.3) + (normalized_pr_activity * 0.7)
        
        logger.debug(f"Community activity score: {activity_score:.3f}")
        return round(activity_score, 3)
    
    # ========== LLM-BASED SEMANTIC ANALYSIS ==========
    
    def _count_prs_with_inconsistent_description(self) -> int:
        """Count PRs where description is inconsistent with implementation - LLM semantic analysis"""
        inconsistent_count = 0
        
        # Analyze a subset of PRs due to LLM cost considerations
        prs_to_analyze = self.repo_info.pr_list[:10]  # Analyze first 10 PRs
        
        for pr in prs_to_analyze:
            if pr.status == "merged" and pr.title:
                try:
                    is_inconsistent = self._analyze_pr_consistency_with_llm(pr)
                    if is_inconsistent:
                        inconsistent_count += 1
                except Exception as e:
                    logger.warning(f"Failed to analyze PR {pr.number} consistency: {str(e)}")
        
        logger.debug(f"PRs with inconsistent descriptions: {inconsistent_count}")
        return inconsistent_count
    
    def _analyze_pr_consistency_with_llm(self, pr: PullRequest) -> bool:
        """Use LLM to semantically analyze if PR description matches its likely implementation"""
        prompt = ChatPromptTemplate.from_template("""
        Analyze if this pull request title and description seem consistent and legitimate.
        
        PR Title: {title}
        PR Body: {body}
        PR Author: {author}
        PR Status: {status}
        Created: {created_at}
        Merged: {merged_at}
        
        Perform semantic analysis to determine if this PR seems suspicious:
        Consider factors like:
        - Vague or misleading titles that could hide malicious changes
        - Very quick merge times (potential bypass of review process)
        - Generic titles that don't describe actual changes
        - Inconsistency between title and body content
        - Suspicious timing patterns
        
        Respond with a JSON object containing:
        - is_inconsistent: boolean (true if potentially suspicious/inconsistent)
        - confidence: float (0.0 to 1.0)
        - reason: string (brief explanation of your analysis)
        """)
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    title=pr.title,
                    body=pr.body[:500] if pr.body else "No description",  # Limit body length
                    author=pr.author,
                    status=pr.status,
                    created_at=pr.created_at,
                    merged_at=pr.merged_at or "Not merged"
                )
            )
            
            # Parse LLM response
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
    
    def _analyze_issue_legitimacy_with_llm(self, issue: Issue) -> bool:
        """Use LLM to semantically analyze if issue content is legitimate"""
        # TODO: This could be used for additional semantic analysis of issues
        # Currently not used in main evaluation but available for future enhancements
        
        prompt = ChatPromptTemplate.from_template("""
        Analyze if this issue seems legitimate or potentially suspicious.
        
        Issue Title: {title}
        Issue Body: {body}
        Issue Author: {author}
        Issue Status: {status}
        Created: {created_at}
        
        Perform semantic analysis to determine legitimacy:
        - Does the issue describe a real problem or feature request?
        - Is the language professional and constructive?
        - Does it seem like spam or malicious content?
        
        Respond with a JSON object containing:
        - is_legitimate: boolean (true if legitimate)
        - confidence: float (0.0 to 1.0)
        - reason: string (brief explanation)
        """)
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    title=issue.title,
                    body=issue.body[:500] if issue.body else "No description",
                    author=issue.author,
                    status=issue.status,
                    created_at=issue.created_at
                )
            )
            
            response_content = response.content
            if isinstance(response_content, str):
                analysis = IssueSemanticAnalysis.model_validate_json(response_content)
                return analysis.is_legitimate and analysis.confidence > 0.7
            else:
                logger.warning(f"Unexpected response type from LLM for issue {issue.number}")
                return True  # Default to legitimate if analysis fails
            
        except Exception as e:
            logger.warning(f"LLM analysis failed for issue {issue.number}: {str(e)}")
            return True  # Default to legitimate if analysis fails
