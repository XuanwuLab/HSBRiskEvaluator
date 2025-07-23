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