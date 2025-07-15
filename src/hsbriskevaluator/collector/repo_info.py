from pydantic import BaseModel, Field
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


class Commit(BaseModel):
    hash: str = Field(description="SHA of the commit")
    committer: str = Field(
        description="username of the user who made the commit")
    message: str = Field(description="commit message")
    timestamp: str = Field(description="timestamp when the commit was made")


class GithubUser(BaseModel):
    username: str = Field(description="username of the user")
    name: str = Field(description="name of the user")
    email: str = Field(description="email of the user")
    roles: list[UserRole] = Field(
        description="list of roles assigned to the user in the repository",
        default_factory=list,
    )


class Comment(BaseModel):
    username: str = Field(
        description="username of the user who made the comment")
    content: str = Field(description="content of the comment")
    timestamp: str = Field(description="timestamp when the comment was made")


class Issue(BaseModel):
    number: int = Field(description="issue number")
    author: str = Field(description="author of the issue")
    title: str = Field(description="title of the issue")
    body: str = Field(description="body content of the issue")
    comments: list[Comment] = Field(
        description="list of comments in the issue")
    status: Literal["open", "closed"] = Field(
        description="status of the issue, can be open or closed"
    )
    url: str = Field(description="URL of the issue")
    created_at: str = Field(description="creation timestamp")
    updated_at: str = Field(description="last update timestamp")


class PullRequest(Issue):
    approvers: list[str] = Field(description="approver of the pull request")
    reviewers: list[str] = Field(
        description="list of reviewers for the pull request")
    merger: str = Field(description="merger of the pull request")
    status: Literal["open", "closed", "merged"] = Field(
        description="status of the pull request"
    )
    merged_at: str | None = Field(
        description="merge timestamp if merged", default=None)
    changed_files: list[str] = Field(
        description="list of changed files in the pull request")
    commits: list[Commit] = Field(
        description="list of commits in the pull request")

class Workflow(BaseModel):
    name: str = Field(description="name of the workflow")
    content: str = Field(description="content of the workflow file")
    path: str = Field(description="path of the workflow file in the repository")

class CheckRun(BaseModel):
    name: str = Field(description="name of the check run")

class RepoInfo(BaseModel):
    pkt_type: Literal["debian", "others"] = "debian"  # only support debian
    pkt_name: str = Field(
        description="package name in package management system such as apt."
    )
    repo_id: str = Field(
        description="unique identifier for the repository, usually the orgname-repo_name format"
    )
    url: str = Field(description="URL of the repository")
    commit_list: list[Commit] = Field(
        description="List of commits in the repository")
    contributor_list: list[GithubUser] = Field(
        description="List of contributors to the repository"
    )
    pr_list: list[PullRequest] = Field(
        description="List of pull requests in the repository"
    )
    issue_list: list[Issue] = Field(
        description="List of issues in the repository")
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
    dependent_list: list[Dependent] = Field(
        description="List of recursive dependencies from package_management system",
        default_factory=list,
    )
    
