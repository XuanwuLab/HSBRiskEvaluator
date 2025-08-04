from pydantic import BaseModel, Field, ConfigDict
from github.Event import Event
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Dependent(BaseModel):
    """Information about a recursive dependency from apt-rdepends"""

    parent: str
    type: str
    name: str
    version: Optional[str] = None


class GithubEvent(BaseModel):
    # don't record WatchEvent, ForkEvent and SponsorshipEvent, they are not useful for repo info
    type: Literal[
        "CommitCommentEvent",
        "CreateEvent",
        "DeleteEvent",
        "GollumEvent",
        "IssueCommentEvent",
        "IssuesEvent",
        "MemberEvent",
        "PublicEvent",
        "PullRequestEvent",
        "PullRequestReviewEvent",
        "PullRequestReviewCommentEvent",
        "PullRequestReviewThreadEvent",
        "PushEvent",
        "ReleaseEvent",
    ] = Field(description="type of the event, such as push, pull_request, issue, etc.")
    actor: str = Field(
        description="username of the user who triggered the event")
    timestamp: str = Field(description="timestamp when the event occurred")
    payload: dict = Field(
        description="payload of the event, containing additional information such as issue"
    )


class UserRole(BaseModel):
    type: Literal["Admin", "Maintainer", "Write", "Triage", "Read"] = Field(
        description="role type of the user in the repository"
    )
    timestamp: str = Field(description="timestamp when the role was assigned")


class User(BaseModel):
    username: str = Field(description="username of the user")
    email: str = Field(description="email of the user")
    type: str = Field(description="type of the user, such as User or Bot")
    model_config = ConfigDict(frozen=True)


class Commit(BaseModel):
    hash: str = Field(description="SHA of the commit")
    author: User = Field(description="the user who made the commit")
    message: str = Field(description="commit message")
    timestamp: str = Field(description="timestamp when the commit was made")
    pull_numbers: list[int] = Field(
        description="pull request numbers to for the commit"
    )


class Comment(BaseModel):
    user: User = Field(description="the user who made the comment")
    content: str = Field(description="content of the comment")
    timestamp: str = Field(description="timestamp when the comment was made")


class PullRequestReview(BaseModel):
    user: User = Field(description="the user who made the review")
    body: str = Field(description="content of the review")
    commit_id: str = Field(description="commit ID associated with the review")
    html_url: str = Field(description="HTML URL of the review")
    id: int = Field(description="unique ID of the review")
    pull_request_url: str = Field(
        description="URL of the pull request associated with the review")
    state: str = Field(
        description="state of the review, such as APPROVED, CHANGES_REQUESTED, or COMMENTED")
    submitted_at: str = Field(
        description="timestamp when the review was submitted")


class Issue(BaseModel):
    number: int = Field(description="issue number")
    author: User = Field(description="author of the issue")
    title: str = Field(description="title of the issue")
    body: str = Field(description="body content of the issue")
    comments: list[Comment] = Field(
        description="list of comments in the issue")
    status: Literal["open", "closed"] = Field(
        description="status of the issue, can be open or closed"
    )
    url: str = Field(description="URL of the issue")
    assignees: list[User] = Field(
        description="list of users assigned to the issue", default_factory=list
    )
    created_at: str = Field(description="creation timestamp")
    updated_at: str = Field(description="last update timestamp")


class PullRequest(Issue):
    approvers: list[User] = Field(description="approver of the pull request")
    reviewers: list[User] = Field(
        description="list of reviewers for the pull request")
    reviews: list[PullRequestReview] = Field(
        description="list of reviews for the pull request, including comments", default_factory=list
    )
    assignees: list[User] = Field(
        description="list of assignees for the pull request", default_factory=list)
    merger: User = Field(description="merger of the pull request")
    status: Literal["open", "closed", "merged"] = Field(
        description="status of the pull request"
    )
    merged_at: str | None = Field(
        description="merge timestamp if merged", default=None)
    changed_files: list[str] = Field(
        description="list of changed files in the pull request"
    )


class Workflow(BaseModel):
    name: str = Field(description="name of the workflow")
    content: str = Field(description="content of the workflow file")
    path: str = Field(
        description="path of the workflow file in the repository")


class CheckRun(BaseModel):
    name: str = Field(description="name of the check run")


class BasicInfo(BaseModel):
    description: str = Field(description="description of the repository")
    stargazers_count: int = Field(
        description="number of stars in the repository")
    watchers_count: int = Field(
        description="number of watchers in the repository")
    forks_count: int = Field(description="number of forks in the repository")
    url: str = Field(
        description="HTML URL of the repository, usually the same as repo_id",
        default="",  # Default to empty string if not available
    )
    clone_url: str = Field(
        description="Clone URL of the repository, used for cloning via git",
        default="",  # Default to empty string if not available
    )


class RepoInfo(BaseModel):
    pkt_type: Literal["debian", "others"] = "debian"  # only support debian
    pkt_name: str | list[str] = Field(
        description="package name list in package management system such as apt."
    )
    repo_id: str = Field(
        description="unique identifier for the repository, usually the orgname-repo_name format"
    )
    basic_info: BasicInfo = Field(
        description="Basic information of the repository")
    commit_list: list[Commit] = Field(
        description="List of commits in the repository")
    pr_list: list[PullRequest] = Field(
        description="List of pull requests in the repository"
    )
    issue_list: list[Issue] = Field(
        description="List of issues in the repository")
    issue_without_comment_list: list[Issue] = Field(
        default_factory=list,
        description="List of issues without comments in the repository"
    )
    pr_without_comment_list: list[PullRequest] = Field(
        default_factory=list,
        description="List of pull requests without comments in the repository"
    )
    binary_file_list: list[str] = Field(
        description="List of binary files in the repository"
    )
    local_repo_dir: str | None = Field(
        description="Local repository directory path relative to data_dir", default=None
    )
    event_list: list[GithubEvent] = Field(
        description="List of events in the repository, such as pushes, issues, pull requests, etc."
    )
    workflow_list: list[Workflow] = Field(
        description="List of workflows in the repository"
    )
    check_run_list: list[CheckRun] = Field(
        description="List of check runs in the repository"
    )
