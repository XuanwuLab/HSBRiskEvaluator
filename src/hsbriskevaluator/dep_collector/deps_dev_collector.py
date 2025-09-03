import requests
import logging
from typing import Dict, List, Optional
from urllib.parse import quote
from .base import BaseDepCollector

logger = logging.getLogger(__name__)


class DepsDevCollector(BaseDepCollector):
    """
    Dependency collector that queries dependency information through deps.dev API
    """
    
    def __init__(self, platform):
        """
        Initialize the deps.dev collector
        
        Args:
            platform: The package management system (GO, RUBYGEMS, NPM, CARGO, MAVEN, PYPI, NUGET)
        """
        super().__init__()
        self.platform = platform.upper()
        self.base_url = "https://api.deps.dev"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HSBRiskEvaluator/1.0'
        })
        
        # Validate platform
        valid_platforms = {"GO", "RUBYGEMS", "NPM", "CARGO", "MAVEN", "PYPI", "NUGET"}
        if self.platform not in valid_platforms:
            raise ValueError(f"Platform {self.platform} not supported. Valid platforms: {valid_platforms}")
    
    def _make_request(self, url: str) -> Optional[Dict]:
        """
        Make HTTP request to deps.dev API with error handling
        
        Args:
            url: The API endpoint URL
            
        Returns:
            JSON response as dict or None if request failed
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
        except ValueError as e:
            logger.error(f"Failed to parse JSON response from {url}: {e}")
            return None
    
    def _get_package_info(self, package_name: str) -> Optional[Dict]:
        """
        Get package information including available versions
        
        Args:
            package_name: Name of the package
            
        Returns:
            Package information dict or None if not found
        """
        encoded_name = quote(package_name, safe='')
        url = f"{self.base_url}/v3/systems/{self.platform}/packages/{encoded_name}"
        
        logger.info(f"Fetching package info for {package_name} from {url}")
        return self._make_request(url)


    def _get_sorted_versions(self, package_info: Dict) -> List[str]:
        """
        Get all versions sorted by publication date (newest first)
        
        Args:
            package_info: Package information from GetPackage API
            
        Returns:
            List of version strings sorted by publication date
        """
        if not package_info or 'versions' not in package_info:
            return []
        
        versions = package_info['versions']
        if not versions:
            return []
        
        # Sort by publishedAt date (newest first), fallback to version order
        def sort_key(version):
            published_at = version.get('publishedAt', '')
            return published_at if published_at else '0'
        
        sorted_versions = sorted(versions, key=sort_key, reverse=True)
        return [v['versionKey']['version'] for v in sorted_versions]
    
    def _get_dependencies(self, package_name: str, version: str) -> Optional[Dict]:
        """
        Get dependency information for a specific package version
        
        Args:
            package_name: Name of the package
            version: Version of the package
            
        Returns:
            Dependency information dict or None if not found
        """
        encoded_name = quote(package_name, safe='')
        encoded_version = quote(version, safe='')
        url = f"{self.base_url}/v3/systems/{self.platform}/packages/{encoded_name}/versions/{encoded_version}:dependencies"
        
        logger.info(f"Fetching dependencies for {package_name}@{version} from {url}")
        return self._make_request(url)
    
    def _extract_all_dependencies(self, dep_info: Dict) -> List[Dict]:
        """
        Extract all dependencies (direct and indirect) from dependency graph
        
        Args:
            dep_info: Dependency information from GetDependencies API
            
        Returns:
            List of all dependency dicts with name, version, relation, and bundled info
        """
        if not dep_info or 'nodes' not in dep_info:
            return []
        
        nodes = dep_info['nodes']
        
        if not nodes:
            return []
        
        # Extract all dependencies except the root node (first node)
        all_deps = []
        
        for i, node in enumerate(nodes):
            # Skip the root node (index 0)
            if i == 0:
                continue
                
            version_key = node.get('versionKey', {})
            relation = node.get('relation', 'UNKNOWN')
            
            # Find the requirement for this dependency
            requirement = ''
            if 'edges' in dep_info:
                for edge in dep_info['edges']:
                    if edge.get('toNode') == i:
                        requirement = edge.get('requirement', '')
                        break
            
            all_deps.append({
                'name': version_key.get('name', ''),
                'version': version_key.get('version', ''),
                'relation': relation,
                'requirement': requirement,
                'bundled': node.get('bundled', False)
            })
        
        return all_deps
    
    def collect(self, package_name: str) -> Dict:
        """
        Collect dependency information for a package
        
        Args:
            package_name: Name of the package to collect dependencies for
            
        Returns:
            Dict containing dependency information with structure:
            {
                'package_name': str,
                'platform': str,
                'version': str,
                'dependencies': List[Dict],
                'error': Optional[str]
            }
        """
        result = {
            'package_name': package_name,
            'platform': self.platform,
            'version': None,
            'dependencies': [],
            'error': None
        }
        
        try:
            # Step 1: Get package information
            package_info = self._get_package_info(package_name)
            if not package_info:
                result['error'] = f"Package {package_name} not found"
                return result
            
            # Step 2: Get sorted versions (newest first)
            versions = self._get_sorted_versions(package_info)
            if not versions:
                result['error'] = f"No versions found for package {package_name}"
                return result
            
            # Step 3: Try to get dependencies, starting with latest version
            dep_info = None
            used_version = None
            
            for version in versions:
                logger.info(f"Trying to get dependencies for {package_name}@{version}")
                dep_info = self._get_dependencies(package_name, version)
                
                if dep_info and 'nodes' in dep_info:
                    # Check if we have meaningful dependency data
                    if len(dep_info.get('nodes', [])) > 1 or dep_info.get('edges'):
                        used_version = version
                        break
                    else:
                        logger.info(f"No dependency data found for {package_name}@{version}, trying next version")
                else:
                    logger.info(f"Failed to get dependency info for {package_name}@{version}, trying next version")
            
            if not dep_info or not used_version:
                result['error'] = f"No dependency information available for any version of {package_name}"
                return result
            
            # Step 4: Extract all dependencies
            result['version'] = used_version
            result['dependencies'] = self._extract_all_dependencies(dep_info)
            
            logger.info(f"Successfully collected {len(result['dependencies'])} dependencies for {package_name}@{used_version}")
            
        except Exception as e:
            logger.error(f"Error collecting dependencies for {package_name}: {e}")
            result['error'] = str(e)
        
        return result
