import logging
import yaml
from pathlib import Path
from typing import List, Dict, Set, Optional
from hsbriskevaluator.evaluator.base import BaseEvaluator, DependencyEvalResult, DependencyDetail
from hsbriskevaluator.collector.repo_info import RepoInfo
from hsbriskevaluator.utils.apt_utils import AptUtils, Dependent

logger = logging.getLogger(__name__)


class DependencyEvaluator(BaseEvaluator):
    """Evaluator for Software Supply Chain Dependency Location metrics"""
    
    def __init__(self, repo_info: RepoInfo, max_concurrency: int = 5):
        super().__init__(repo_info)
        self.apt_utils = AptUtils(max_concurrency=max_concurrency)
        self.cloud_product_packages = self._load_cloud_product_packages()
        
        # Cache for OS default and mainstream packages discovered via apt
        self._os_default_cache: Set[str] = set()
        self._mainstream_os_cache: Set[str] = set()
        self._cache_initialized = False
    
    def _load_cloud_product_packages(self) -> Set[str]:
        """Load cloud product packages from YAML file"""
        try:
            yaml_path = Path(__file__).parent / "cloud_products.yaml"
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                cloud_packages = set(data.get('cloud_products', []))
                logger.info(f"Loaded {len(cloud_packages)} cloud product packages from YAML")
                return cloud_packages
        except Exception as e:
            logger.error(f"Failed to load cloud product packages: {str(e)}")
            # Fallback to minimal set
            return {
                'aws-cli', 'awscli', 'azure-cli', 'google-cloud-sdk', 'kubectl',
                'terraform', 'ansible', 'docker', 'kubernetes', 'helm'
            }
        
    def evaluate(self) -> DependencyEvalResult:
        """Evaluate dependency location metrics"""
        logger.info(f"Starting dependency evaluation for repository: {self.repo_info.repo_id}")
        
        try:
            # Analyze package classification
            dependency_detail = self._analyze_package_dependencies()
            
            
            # Calculate supply chain risk score
            
            result = DependencyEvalResult(
                is_os_default_dependency=len(dependency_detail.os_default_dependent) > 0 ,
                is_mainstream_os_dependency=len(dependency_detail.mainstream_dependent) > 0,
                is_cloud_product_dependency= len(dependency_detail.cloud_product_dependent) > 0,
                details=dependency_detail
            )
            
            logger.info(f"Dependency evaluation completed for {self.repo_info.repo_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error during dependency evaluation: {str(e)}")
            raise
    
    def _analyze_package_dependencies(self) -> DependencyDetail:
        """Analyze what type of dependencies this package represents using apt commands"""
        package_name = self.repo_info.pkt_name
        
        # Use apt-cache show to determine package classification
        is_os_default, is_mainstream_os = self.apt_utils.get_package_classification(package_name)
        
        # Check if it's a cloud product package
        is_cloud_product = package_name in self.cloud_product_packages
        
        # Identify potential dependents using apt rdepends
        identified_dependents = self._identify_potential_dependents()
        os_default_dependents = []
        mainstream_dependents = []
        cloud_product_dependents = []

        for dependent in identified_dependents:
            is_os_default, is_mainstream_os = self.apt_utils.get_package_classification(package_name)
            if is_os_default:
                os_default_dependents.append(dependent)
            if is_mainstream_os:
                mainstream_dependents.append(dependent)
            if dependent.name in self.cloud_product_packages:
                cloud_product_dependents.append(dependent)
            
            
        
        result = DependencyDetail(
            os_default_dependent=os_default_dependents,
            mainstream_dependent=mainstream_dependents,
            cloud_product_dependent=cloud_product_dependents,
        )
        
        logger.debug(f"Package dependency analysis: OS_default={is_os_default}, Mainstream_OS={is_mainstream_os}, Cloud={is_cloud_product}")
        return result
    
    def _identify_potential_dependents(self) -> List[Dependent]:
        """Identify potential software that depends on this package using apt rdepends"""
        package_name = self.repo_info.pkt_name.lower()
        dependents = []
        
        try:
            # Use the new get_reverse_dependencies_with_info method
            dependents = self.apt_utils.get_recursive_dependencies(package_name)
            
            # Convert DependencyInfo objects to Dependency objects
            
        except Exception as e:
            logger.warning(f"Failed to identify dependents for {package_name}: {str(e)}")
        
        return dependents
    