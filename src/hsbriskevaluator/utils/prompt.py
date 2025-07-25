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
    * `is_code`: A boolean (`true` or `false`) indicating if the file is part of the codebase.
    * `is_asset`: A boolean (`true` or `false`) indicating if the file is part of the assets.
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
    "is_code": false
    "file_path": "src/test/data/golden_files/approved_user_avatar.jpg"
  },
  {
    "reason": "File is located in a 'docs/media' directory, which strongly indicates it's an image used within the project's documentation.",
    "file_type": "documentation_image",
    "is_test_file": false,
    "is_documentation": true,
    "is_code": false,
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
<task>
I need to analyze a GitHub Actions workflow file for potential command injection vulnerabilities. The analysis should identify common security risks and provide a structured assessment of its danger level.
</task>

<requirement>
1.  Carefully examine the provided `{workflow_content}` for security vulnerabilities.
2.  Pay close attention to these risk factors:
    * **Dangerous Triggers:** Use of `pull_request_target` or `workflow_run`, which can run with elevated permissions on untrusted code.
    * **Unsanitized Input:** Direct use of input from sources like issue titles/bodies, pull request titles/bodies, or commit messages within scripts.
    * **Inline Scripts:** `run` steps that construct shell commands by concatenating strings with user-provided input.
    * **Unmasked Secrets:** Use of sensitive values like tokens or credentials directly in the workflow instead of using GitHub Secrets.
3.  Your final output **must be a single JSON object** and nothing else. Do not include any explanatory text outside of the JSON structure.
4.  The JSON object must contain exactly these three keys:
    * `is_dangerous`: A boolean (`true` or `false`).
    * `danger_level`: A float between 0.0 (perfectly safe) and 1.0 (extremely dangerous).
    * `reason`: A concise string explaining the identified vulnerabilities or why the workflow is considered safe.
</requirement>

<example>
<example_1>
    <workflow_content>
name: CI
on:
  push:
    branches: [ "main" ]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run a one-line script
        run: echo Hello, world!
    </workflow_content>
    <response>
{
  "is_dangerous": false,
  "danger_level": 0.0,
  "reason": "The workflow is triggered on push to a protected branch and only runs a hardcoded, safe command. It does not process external input."
}
    </response>
</example_1>

<example_2>
    <workflow_content>
name: Label PR
on:
  pull_request:
    types: [opened]
jobs:
  labeler:
    runs-on: ubuntu-latest
    steps:
      - name: Greet user
        run: echo "Thanks for the PR, ${{ github.actor }}!"
    </workflow_content>
    <response>
{
  "is_dangerous": true,
  "danger_level": 0.4,
  "reason": "The workflow uses input from 'github.actor' directly in a script. While 'echo' is generally safe, this pattern is risky and could be exploited if the command were more complex."
}
    </response>
</example_2>

<example_3>
    <workflow_content>
name: "Run user script"
on: pull_request_target
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Run PR Body
        run: |
          ${{ github.event.pull_request.body }}
    </workflow_content>
    <response>
{
  "is_dangerous": true,
  "danger_level": 1.0,
  "reason": "This workflow uses the dangerous 'pull_request_target' trigger and executes the unsanitized pull request body directly in the shell, creating a severe command injection vulnerability."
}
    </response>
</example_3>
</example>
"""

CI_WORKFLOW_ANALYSIS_MODEL_ID = "openai/gpt-4.1"
