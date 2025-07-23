GET_DEBIAN_UPSTREAM_PROMPT = """
<task>
Debian packages use Git repositories atsalsa.debian.org, but these are frequently downstream mirrors. The original upstream source is usually on GitHub or Gitlab. 
I want to locate the original URL for a specific Debian package by its name.    
</task>

<requirement>
1. Locate the **downstream Debian Git repository URL** (e.g., `https://salsa.debian.org`) for the package.
2. Identify the **upstream Git URL** where the original source code is maintained (e.g., GitHub, GitLab, Sourceware, etc.).
3. If the package is part of another Debian package, set the `parent_debian_package` field to the name of the Debian 12 package containing it. Otherwise, set the field to `null`.
4. Ensure accurate classification of the `upstream_type` (e.g., GitHub, GitLab, Sourceware, etc.).
5. get upstream_git_url according to the search result, don't make up URLs.
5. get upstream_git_url according to the search result, don't make up URLs.
</requirement>

<example>
<example_1>
    <package_name>libx11</package_name>
    <debian_downstream_git_url>https://salsa.debian.org/xorg-team/lib/libx11</debian_downstream_git_url>
    <upstream_git_url>https://gitlab.freedesktop.org/xorg/lib/libx11</upstream_git_url>
    <upstream_type>gitlab</upstream_type>
    <parent_debian_package>null</parent_debian_package>
</example_1>

<example_2>
    <package_name>xz-utils</package_name>
    <debian_downstream_git_url>https://salsa.debian.org/debian/xz-utils</debian_downstream_git_url>
    <upstream_git_url>https://github.com/tukaani-project/xz</upstream_git_url>
    <upstream_type>github</upstream_type>
    <parent_debian_package>null</parent_debian_package>
</example_2>

<example_3>
    <package_name>libgprofng0</package_name>
    <debian_downstream_git_url>https://salsa.debian.org/gnu-team/binutils</debian_downstream_git_url>
    <upstream_git_url>https://sourceware.org/git/binutils-gdb.git</upstream_git_url>
    <upstream_type>sourceware</upstream_type>
    <parent_debian_package>binutils</parent_debian_package>
</example_3>
</example>
"""

GET_DEBIAN_UPSTREAM_MODEL_ID = "openai/gpt-4.1:online"

# Payload Evaluator Prompts
PAYLOAD_FILE_ANALYSIS_PROMPT = """
Analyze this binary file path from a software repository to categorize its purpose and assess risk.

File Path: {file_path}

Please analyze:
1. Is this file likely a test file, test fixture, or test resource?
2. Is this file documentation-related (images in docs, example files, etc.)?
3. Is this file an image, media file, or similar content file?

Consider the context:
- Path structure and directory names
- File naming conventions
- Common patterns in software repositories
- Whether the file type makes sense in its location

Respond with a JSON object containing:
- reason: string (brief explanation of your assessment)
- file_type: string (inferred file type/purpose, e.g., "test_resource", "documentation_image", "executable", etc.)
- is_test_file: boolean (true if this appears to be a test file or test resource)
- is_documentation: boolean (true if this appears to be documentation-related)
- file_path: string (the original file path for reference)

Don't include any other text, just the JSON response.
"""

PAYLOAD_FILE_ANALYSIS_MODEL_ID = "anthropic/claude-3.5-sonnet"

# Community Evaluator Prompts
PR_CONSISTENCY_ANALYSIS_PROMPT = """
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
"""

PR_CONSISTENCY_ANALYSIS_MODEL_ID = "anthropic/claude-3.5-sonnet"

# CI Evaluator Prompts
CI_WORKFLOW_ANALYSIS_PROMPT = """
Analyze if this GitHub workflow seems to have potential command injection vulnerabilities:

{workflow_content}

Consider factors like:
- Use dangerous triggers, like pull_request_target, workflow_run
- Use input from sources like issue titles, pull request bodies, or commit messages without proper sanitization.
- Use inline shell commands or scripts that execute user input directly.
- Use sensitive values like tokens or credentials, without masking via GitHub Secrets.

Respond with a JSON object containing:
- is_dangerous: boolean
- danger_level: float (0.0 to 1.0, where 0.0 is safe and 1.0 is very dangerous)
- reason: string (brief explanation)

Don't include any other text, just the JSON response.
"""

CI_WORKFLOW_ANALYSIS_MODEL_ID = "anthropic/claude-3.5-sonnet"
