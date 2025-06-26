from pydantic import BaseModel, Field
from typing import Literal
class GithubUser(BaseModel):
    username: str = Field(description="username of the user")
    name: str = Field(description="name of the user")
    email: str = Field(description="email of the user")

class Comment(BaseModel):
    username: str = Field(description="username of the user who made the comment")
    content: str = Field(description="content of the comment")

class Issue(BaseModel):
    number: int = Field(description="issue number")
    author: str = Field(description="author of the issue")
    title: str = Field(description="title of the issue")
    body: str = Field(description="body content of the issue")
    comments: list[Comment] = Field(description="list of comments in the issue")
    status: Literal['open', 'closed'] = Field(description="status of the issue, can be open or closed")
    url: str = Field(description="URL of the issue")
    created_at: str = Field(description="creation timestamp")
    updated_at: str = Field(description="last update timestamp")

class PullRequest(BaseModel):
    number: int = Field(description="pull request number")
    title: str = Field(description="title of the pull request")
    author: str = Field(description="author of the pull request")
    approver: str = Field(description="approver of the pull request")
    merger: str = Field(description="merger of the pull request")
    status: Literal['open', 'closed', 'merged'] = Field(description="status of the pull request")
    url: str = Field(description="URL of the pull request")
    created_at: str = Field(description="creation timestamp")
    updated_at: str = Field(description="last update timestamp")
    merged_at: str | None = Field(description="merge timestamp if merged", default=None)

class RepoInfo(BaseModel):
    pkt_type: Literal['debian'] = "debian" #only support debian
    pkt_name: str = Field(description="package name in package management system such as apt.")
    repo_id: str = Field(description="unique identifier for the repository, usually the orgname-repo_name format")
    url: str = Field(description="URL of the repository") 
    contributor_list: list[GithubUser] = Field(description="List of contributors to the repository")
    pr_list: list[PullRequest] = Field(description="List of pull requests in the repository")
    issue_list: list[Issue] = Field(description="List of issues in the repository")
    binary_file_list: list[str] = Field(description="List of binary files in the repository")
