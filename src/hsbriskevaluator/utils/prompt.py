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
PAYLOAD_FILES_ANALYSIS_PROMPT = """
<task>
You will be given a list of binary file paths from a software repository. Your task is to analyze each file path to determine its purpose. Based on your analysis, you will return a JSON array, where each object in the array corresponds to one of the input file paths.
</task>

<requirement>
1.  Analyze each file path in the input list, considering directory names (e.g., `test`, `spec`, `doc`, `vendor`), file names, and file extensions.
2.  For each file path, generate a JSON object with the following exact keys:
    * `file_path`: The original, unmodified file path that was analyzed.
    * `file_type`: A string categorizing the file's purpose (e.g., "test_resource", "documentation_image", "build_artifact", "vendored_dependency", "application_icon").
    * `is_test_file`: A boolean (`true` or `false`) indicating if the file appears to be a test file or test resource.
    * `is_documentation`: A boolean (`true` or `false`) indicating if the file is part of the documentation.
    * `reason`: A brief string explaining your classification based on the path's structure and conventions.
3.  Combine all the individual JSON objects into a single JSON array for the final output.
</requirement>

<example>
<user_input>
["src/test/data/golden_files/approved_user_avatar.jpg", "docs/media/screenshot-v1.png", "assets/app_icon.ico"]
</user_input>
<model_output>
[
  {
    "reason": "File is located in a 'src/test/data' directory, a common pattern for test resources or fixtures used in unit or integration tests.",
    "file_type": "test_resource",
    "is_test_file": true,
    "is_documentation": false,
    "file_path": "src/test/data/golden_files/approved_user_avatar.jpg"
  },
  {
    "reason": "File is located in a 'docs/media' directory, which strongly indicates it's an image used within the project's documentation.",
    "file_type": "documentation_image",
    "is_test_file": false,
    "is_documentation": true,
    "file_path": "docs/media/screenshot-v1.png"
  },
]
</model_output>
</example>
"""

PAYLOAD_FILES_ANALYSIS_MODEL_ID = "openai/gpt-4.1"

# Community Evaluator Prompts
PR_CONSISTENCY_ANALYSIS_PROMPT = """
<task>
You will be given a JSON array where each object contains the metadata for a single Pull Request (PR). Your task is to analyze each PR in the array to assess whether it appears legitimate or suspicious. You will then return a JSON array containing an analysis object for each corresponding input PR.
</task>

<requirement>
1.  Iterate through each Pull Request object in the input JSON array.
2.  For each PR, perform the following analysis based on its metadata:
    * **Assess Consistency:** Check if the `title` and `body` are consistent with the `changed_files`.
    * **Evaluate Timing:** Analyze the duration between `created_at` and `merged_at`. A very short time can be a significant risk factor.
    * **Identify Vagueness:** Look for vague or generic titles (e.g., "Update", "Fix") that could hide risky changes.
3.  For each input PR, generate a single JSON object with the following exact keys:
    * `is_inconsistent`: A boolean (`true` if the PR seems suspicious or inconsistent, `false` otherwise).
    * `confidence`: A float between 0.0 and 1.0 representing your confidence in the assessment.
    * `reason`: A brief string explaining your conclusion, referencing specific evidence from the PR data.
4.  Combine all the individual analysis objects into a single JSON array. **The order of objects in the output array must match the order of the PRs in the input array.**
5.  The final output must be only the JSON array and nothing else.
</requirement>

<example>
<user_input>
[
  {
    "title": "Update",
    "author": "suspicious_user",
    "status": "MERGED",
    "created_at": "2025-07-26T02:30:05Z",
    "merged_at": "2025-07-26T02:31:10Z",
    "changed_files": ["pom.xml", "src/main/java/com/mybank/security/Connection.java"],
    "body": ""
  },
  {
    "title": "FEAT: Add user profile endpoint",
    "author": "good_dev",
    "status": "MERGED",
    "created_at": "2025-07-22T11:00:45Z",
    "merged_at": "2025-07-24T09:30:00Z",
    "changed_files": ["src/main/java/com/myapi/controller/UserController.java", "src/test/java/com/myapi/controller/UserControllerTest.java"],
    "body": "Implements the new GET /api/v1/user/{id} endpoint as per ticket JIRA-451. Includes service logic and unit tests."
  }
]
</user_input>
<model_output>
[
  {
    "is_inconsistent": true,
    "confidence": 0.9,
    "reason": "The PR has a vague title ('Update') and empty body, was merged in just over a minute, and modifies critical files like 'pom.xml' and a security-related class. This combination is highly suspicious."
  },
  {
    "is_inconsistent": false,
    "confidence": 1.0,
    "reason": "The PR has a clear, descriptive title and body. The changed files are directly related to the stated feature, and the merge time of nearly two days suggests a proper review cycle occurred."
  }
]
</model_output>
</example>
"""

PR_CONSISTENCY_ANALYSIS_MODEL_ID = "openai/gpt-4.1"

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

