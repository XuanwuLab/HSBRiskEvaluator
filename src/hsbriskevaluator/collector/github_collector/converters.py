"""
Conversion utilities for transforming GitHub API objects to Pydantic models.
"""

import logging
from typing import Optional
from datetime import datetime

from github.NamedUser import NamedUser
from github.Issue import Issue as GithubIssue
from github.PullRequest import PullRequest as GithubPR
from github.PullRequestReview import PullRequestReview as GithubPullRequestReview
from github.PullRequestComment import PullRequestComment
from github.IssueComment import IssueComment
from github.Commit import Commit as GithubCommit
from github.Event import Event as GithubEventObj
from github import GithubException

from ..repo_info import (
    User,
    Issue,
    PullRequest,
    PullRequestReview,
    Comment,
    GithubEvent,
    Commit,
)

logger = logging.getLogger(__name__)


class GitHubConverter:
    """Handles conversion from GitHub API objects to Pydantic models"""

    @staticmethod
    def to_user(user: NamedUser) -> User:
        """Convert GitHub NamedUser to User model"""
        return User(username=user.login, email=user.email or "", type=user.type)

    @staticmethod
    def to_comment(comment: IssueComment | PullRequestComment) -> Comment:
        """Convert PyGithub IssueComment to Comment model"""
        return Comment(
            user=GitHubConverter.to_user(comment.user),
            content=comment.body or "",
            timestamp=comment.created_at.isoformat() if comment.created_at else "",
        )

    @staticmethod
    def to_pr_review(review: GithubPullRequestReview) -> PullRequestReview:
        """Convert PyGithub PullRequestReview to PullRequestReview model"""
        return PullRequestReview(
            user=GitHubConverter.to_user(review.user),
            body=review.body or "",
            commit_id=review.commit_id or "",
            html_url=review.html_url or "",
            id=review.id,
            pull_request_url=review.pull_request_url or "",
            state=review.state or "",
            submitted_at=review.submitted_at.isoformat() if review.submitted_at else "",
        )

    @staticmethod
    def to_commit(commit: GithubCommit) -> Commit:
        """Convert PyGithub Commit to Commit model"""
        return Commit(
            hash=commit.sha,
            author=GitHubConverter.to_user(commit.author or commit.committer)
            if commit.author or commit.committer
            else User(username="", email="", type=""),
            message=commit.commit.message or "",
            timestamp=commit.commit.committer.date.isoformat()
            if commit.commit.committer.date
            else "",
            pull_numbers=list(map(lambda pr: pr.number, commit.get_pulls())),
        )

    @staticmethod
    def to_github_event(event: GithubEventObj) -> Optional[GithubEvent]:
        """
        Convert PyGithub Event to GithubEvent model

        Args:
            event: PyGithub Event object

        Returns:
            GithubEvent: Converted event or None if event type should be filtered out
        """
        # Filter out unwanted event types
        excluded_types = {"WatchEvent", "ForkEvent", "SponsorshipEvent"}
        if event.type in excluded_types:
            logger.info(f"Skipping excluded event type: {event.type}")
            return None

        try:
            # Type cast to satisfy Pydantic's Literal type checking
            return GithubEvent(
                type=event.type,  # type:ignore
                actor=event.actor.login if event.actor else "unknown",
                timestamp=event.created_at.isoformat() if event.created_at else "",
                payload=event.payload or {},
            )
        except Exception as e:
            logger.warning(f"Error converting event {event.type}: {e}")
            return None

    @staticmethod
    def to_issue(issue: GithubIssue, with_comment: bool = True) -> Issue:
        """Convert PyGithub Issue to Issue model"""
        comments = []
        if with_comment:
            comments = [
                GitHubConverter.to_comment(comment) for comment in issue.get_comments()
            ]

        # Ensure status is valid for Issue model
        status = "open" if issue.state == "open" else "closed"
        assignees = []
        for assignee in issue.assignees:
            assignees.append(GitHubConverter.to_user(assignee))

        return Issue(
            number=issue.number,
            author=GitHubConverter.to_user(issue.user),
            title=issue.title,
            body=issue.body or "",
            comments=comments,
            assignees=assignees,
            status=status,
            url=issue.html_url,
            created_at=issue.created_at.isoformat() if issue.created_at else "",
            updated_at=issue.updated_at.isoformat() if issue.updated_at else "",
        )

    @staticmethod
    def to_pull_request(pr: GithubPR, with_comment: bool = True) -> PullRequest:
        """Convert PyGithub PullRequest to PullRequest model"""
        # Get the last approving review
        approvers: list[User] = []
        reviewers: list[User] = []
        assignees: list[User] = []
        reviews = []
        try:
            for review in pr.get_reviews():
                reviewers.append(GitHubConverter.to_user(review.user))
                reviews.append(GitHubConverter.to_pr_review(review))
                if review.state == "APPROVED":
                    approvers.append(GitHubConverter.to_user(review.user))
        except GithubException:
            logger.warning(f"Could not fetch reviews for PR #{pr.number}")

        for reviewer in pr.requested_reviewers:
            reviewers.append(GitHubConverter.to_user(reviewer))

        reviewers = list(
            {reviewer.username: reviewer for reviewer in reviewers}.values())

        for assignee in pr.assignees:
            assignees.append(GitHubConverter.to_user(assignee))

        # Get merger information
        merger = User(username="Not merged", email="", type="None")
        if pr.merged and pr.merged_by:
            merger = GitHubConverter.to_user(pr.merged_by)

        # Determine status - ensure it matches the Literal type
        if pr.merged:
            status = "merged"
        elif pr.state == "open":
            status = "open"
        else:
            status = "closed"

        comments = []
        if with_comment:
            comments = [
                GitHubConverter.to_comment(comment) for comment in pr.get_comments()
            ] + [GitHubConverter.to_comment(comment) for comment in pr.get_issue_comments()]
            comments = sorted(
                comments, key=lambda x: datetime.fromisoformat(x.timestamp))

        # Handle timestamps safely
        created_at_str = pr.created_at.isoformat() if pr.created_at else ""
        updated_at_str = pr.updated_at.isoformat() if pr.updated_at else ""
        merged_at_str = pr.merged_at.isoformat() if pr.merged_at else None

        if with_comment:
            changed_files = list(
                map(lambda file: file.filename, pr.get_files()))
        else:
            changed_files = []

        return PullRequest(
            number=pr.number,
            title=pr.title,
            author=GitHubConverter.to_user(pr.user),
            reviewers=reviewers,
            reviews=reviews,
            assignees=assignees,
            body=pr.body or "",
            comments=comments,
            approvers=approvers,
            merger=merger,
            status=status,
            url=pr.html_url,
            created_at=created_at_str,
            updated_at=updated_at_str,
            merged_at=merged_at_str,
            changed_files=changed_files,
        )
