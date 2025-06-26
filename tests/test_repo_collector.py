#!/usr/bin/env python3
"""
Test script for the GitHub Collector 
"""

import asyncio
import sys
import os

from hsbriskevaluator.utils.file import get_data_dir

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hsbriskevaluator.collector.github_collector import collect_github_repo_info

async def test_collector():
    """Test the GitHub collector"""
    try:
        print("Testing GitHub collector...")
        
        # Test with a small public repo
        repo_info = await collect_github_repo_info(
            'tukaani-project/xz', 
            max_contributors=3, 
            max_issues=3, 
            max_prs=3
        )
        
        print(f"✅ Successfully collected info for {repo_info.repo_id}")
        print(f"   - URL: {repo_info.url}")
        print(f"   - Contributors: {len(repo_info.contributor_list)}")
        print(f"   - Issues: {len(repo_info.issue_list)}")
        print(f"   - Pull Requests: {len(repo_info.pr_list)}")
        print(f"   - Binary files: {len(repo_info.binary_file_list)}")
        
        # Print some sample data
        if repo_info.contributor_list:
            print(f"   - First contributor: {repo_info.contributor_list[0].username}")
        
        if repo_info.pr_list:
            pr = repo_info.pr_list[0]
            print(f"   - First PR: #{pr.number} - {pr.title[:50]}...")
            print(f"     Status: {pr.status}, Author: {pr.author}")
        
        print("\n🎉 Test successful!")
        with open(get_data_dir() / 'test_collector_output.json', 'w') as f:
            f.write(repo_info.model_dump_json(indent=2))
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_collector())
    sys.exit(0 if result else 1)
