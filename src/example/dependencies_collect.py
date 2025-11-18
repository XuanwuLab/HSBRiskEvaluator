#!/usr/bin/env python3
from hsbriskevaluator.dep_collector.deps_dev_collector import DepsDevCollector

def main():
    """Demonstrate usage of the DepsDevCollector"""
    
    # Example 1: Collect NPM package dependencies
    print("=== NPM Package Example ===")
    npm_collector = DepsDevCollector(platform="npm")
    result = npm_collector.collect("lodash")
    
    print(f"Package: {result['package_name']}")
    print(f"Platform: {result['platform']}")
    print(f"Version: {result['version']}")
    print(f"Total dependencies: {len(result['dependencies'])}")
    
    if result['error']:
        print(f"Error: {result['error']}")
    else:
        # Show direct vs indirect dependencies
        direct_deps = [dep for dep in result['dependencies'] if dep['relation'] == 'DIRECT']
        indirect_deps = [dep for dep in result['dependencies'] if dep['relation'] == 'INDIRECT']
        
        print(f"Direct dependencies: {len(direct_deps)}")
        print(f"Indirect dependencies: {len(indirect_deps)}")
        
        if direct_deps:
            print("\nDirect dependencies:")
            for dep in direct_deps[:3]:  # Show first 3
                print(f"  - {dep['name']}@{dep['version']} (requirement: {dep['requirement']})")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Collect PyPI package dependencies
    print("=== PyPI Package Example ===")
    pypi_collector = DepsDevCollector(platform="pypi")
    result = pypi_collector.collect("flask")
    
    print(f"Package: {result['package_name']}")
    print(f"Platform: {result['platform']}")
    print(f"Version: {result['version']}")
    print(f"Total dependencies: {len(result['dependencies'])}")
    
    if result['error']:
        print(f"Error: {result['error']}")
    else:
        # Show direct vs indirect dependencies
        direct_deps = [dep for dep in result['dependencies'] if dep['relation'] == 'DIRECT']
        indirect_deps = [dep for dep in result['dependencies'] if dep['relation'] == 'INDIRECT']
        
        print(f"Direct dependencies: {len(direct_deps)}")
        print(f"Indirect dependencies: {len(indirect_deps)}")
        
        if direct_deps:
            print("\nDirect dependencies:")
            for dep in direct_deps[:3]:  # Show first 3
                print(f"  - {dep['name']}@{dep['version']} (requirement: {dep['requirement']})")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Different platforms
    print("=== Multiple Platform Support ===")
    platforms = ["npm", "pypi", "maven", "cargo"]
    packages = ["express", "django", "junit", "serde"]
    
    for platform, package in zip(platforms, packages):
        try:
            collector = DepsDevCollector(platform=platform)
            result = collector.collect(package)
            
            status = "✓" if not result['error'] else "✗"
            dep_count = len(result['dependencies'])
            print(f"{status} {platform.upper()}: {package} - {dep_count} dependencies")
            
        except Exception as e:
            print(f"✗ {platform.upper()}: {package} - Error: {e}")

if __name__ == "__main__":
    main()
