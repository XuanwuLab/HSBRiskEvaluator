#!/usr/bin/env python3
"""
Script to merge YAML files with configurable merge rules.

Supports configurable source and target directories, and flexible field-level
merge behaviors (overwrite, merge, ignore).
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from enum import Enum

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from hsbriskevaluator.utils.file import get_data_dir
from hsbriskevaluator.utils.progress_manager import get_progress_manager


class MergeBehavior(str, Enum):
    """Enum for different merge behaviors."""
    OVERWRITE = "overwrite"  # Replace field completely with source data
    MERGE = "merge"          # Merge/combine field data
    IGNORE = "ignore"        # Keep target data, ignore source


class FieldMergeRule(BaseModel):
    """Configuration for how to merge a specific field."""
    field_name: str = Field(description="Name of the field to configure")
    behavior: MergeBehavior = Field(
        default=MergeBehavior.OVERWRITE,
        description="How to handle this field during merge"
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the merge rule"
    )


class MergeSettings(BaseSettings):
    """Settings for the repo info merge process."""
    
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_file=".env",
        extra="allow"
    )
    
    # Directory settings
    source_dir: str = Field(
        default="repo_info_orig",
        description="Source directory name (relative to data dir)"
    )
    target_dir: str = Field(
        default="repo_info", 
        description="Target directory name (relative to data dir)"
    )
    output_dir: str = Field(
        default="repo_info_new",
        description="Output directory name (relative to data dir)"
    )
    
    # Processing settings
    max_workers: int = Field(
        default=10,
        description="Maximum number of worker threads"
    )
    
    # Default merge behavior
    default_behavior: MergeBehavior = Field(
        default=MergeBehavior.OVERWRITE,
        description="Default merge behavior for fields not explicitly configured"
    )
    
    # Field-specific merge rules
    field_rules: List[FieldMergeRule] = Field(
        default_factory=lambda: [
            # Core identification fields - typically keep from target
            FieldMergeRule(
                field_name="pkt_type",
                behavior=MergeBehavior.IGNORE,
                description="Keep target package type (debian/others)"
            ),
            FieldMergeRule(
                field_name="pkt_name",
                behavior=MergeBehavior.IGNORE,
                description="Keep target package name"
            ),
            FieldMergeRule(
                field_name="repo_id",
                behavior=MergeBehavior.IGNORE,
                description="Keep target repository ID"
            ),
            
            # Basic repository information - typically stable, keep from target
            FieldMergeRule(
                field_name="basic_info",
                behavior=MergeBehavior.IGNORE,
                description="Keep target basic repo info (description, stars, forks, URLs)"
            ),
            
            # Local/system-specific fields - keep from target
            FieldMergeRule(
                field_name="local_repo_dir",
                behavior=MergeBehavior.IGNORE,
                description="Keep target local directory path"
            ),
            FieldMergeRule(
                field_name="binary_file_list",
                behavior=MergeBehavior.OVERWRITE,
                description="Keep target binary file list (system-specific)"
            ),
            # Main data fields - overwrite with source (newer data)
            FieldMergeRule(
                field_name="commit_list", 
                behavior=MergeBehavior.IGNORE,
                description="Replace commit list with source data (newer commits)"
            ),
            FieldMergeRule(
                field_name="pr_list",
                behavior=MergeBehavior.OVERWRITE,
                description="Merge pull request lists (combine both datasets)"
            ),
            FieldMergeRule(
                field_name="issue_list",
                behavior=MergeBehavior.OVERWRITE,
                description="Merge issue lists (combine both datasets)"
            ),
            
            # Comment-specific fields - overwrite with source (performance optimized)
            FieldMergeRule(
                field_name="issue_without_comment_list",
                behavior=MergeBehavior.IGNORE,
                description="Replace issues without comments with source data"
            ),
            FieldMergeRule(
                field_name="pr_without_comment_list",
                behavior=MergeBehavior.IGNORE,
                description="Replace PRs without comments with source data"
            ),
            
            # Activity/event data - merge for comprehensive view
            FieldMergeRule(
                field_name="event_list",
                behavior=MergeBehavior.OVERWRITE,
                description="Merge event lists (combine repository activities)"
            ),
            
            # CI/CD and automation - merge for comprehensive view
            FieldMergeRule(
                field_name="workflow_list",
                behavior=MergeBehavior.OVERWRITE,
                description="Merge workflow lists (combine CI/CD definitions)"
            ),
            FieldMergeRule(
                field_name="check_run_list",
                behavior=MergeBehavior.OVERWRITE,
                description="Merge check run lists (combine CI results)"
            ),
            
            # Dependency information - merge for comprehensive view
            FieldMergeRule(
                field_name="dependent_list",
                behavior=MergeBehavior.OVERWRITE,
                description="Merge dependency lists (combine package dependencies)"
            )
        ],
        description="List of field-specific merge rules for all RepoInfo fields"
    )
    
    def get_field_behavior(self, field_name: str) -> MergeBehavior:
        """Get the merge behavior for a specific field."""
        for rule in self.field_rules:
            if rule.field_name == field_name:
                return rule.behavior
        return self.default_behavior
    
    def get_source_path(self, data_dir: Path) -> Path:
        """Get full path to source directory."""
        return data_dir / self.source_dir
    
    def get_target_path(self, data_dir: Path) -> Path:
        """Get full path to target directory."""
        return data_dir / self.target_dir
    
    def get_output_path(self, data_dir: Path) -> Path:
        """Get full path to output directory.""" 
        return data_dir / self.output_dir
    
    def validate_field_coverage(self) -> None:
        """Validate that all RepoInfo fields are covered by merge rules."""
        # Import RepoInfo to get all field names
        from hsbriskevaluator.collector.repo_info import RepoInfo
        
        # Get all field names from RepoInfo model
        repo_info_fields = set(RepoInfo.model_fields.keys())
        
        # Get configured field names from rules
        configured_fields = set(rule.field_name for rule in self.field_rules)
        
        # Check for missing fields
        missing_fields = repo_info_fields - configured_fields
        if missing_fields:
            logger.warning(f"Fields not configured in merge rules: {sorted(missing_fields)}")
            for field in sorted(missing_fields):
                logger.warning(f"  - {field}: will use default behavior ({self.default_behavior.value})")
        
        # Check for extra fields (configured but not in RepoInfo)
        extra_fields = configured_fields - repo_info_fields
        if extra_fields:
            logger.warning(f"Configured fields not found in RepoInfo: {sorted(extra_fields)}")
        
        logger.info(f"Field coverage validation: {len(configured_fields)}/{len(repo_info_fields)} fields configured")


# Default settings instance
DEFAULT_MERGE_SETTINGS = MergeSettings()

# Setup logging
progress_manager = get_progress_manager()
logger = progress_manager.logger

# Disable YAML aliases for cleaner output
yaml.Dumper.ignore_aliases = lambda *args: True


def load_yaml_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a YAML file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary containing the YAML data, or None if file doesn't exist or error occurs
    """
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading YAML file {file_path}: {e}")
        return None


def save_yaml_file(data: Dict[str, Any], file_path: Path) -> bool:
    """
    Save a dictionary to a YAML file.
    
    Args:
        data: Dictionary to save
        file_path: Path where to save the YAML file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        return True
    except Exception as e:
        logger.error(f"Error saving YAML file {file_path}: {e}")
        return False


def merge_repo_info(target_data: Dict[str, Any], source_data: Dict[str, Any], settings: MergeSettings) -> Dict[str, Any]:
    """
    Merge repo info data according to configurable rules.
    
    Args:
        target_data: Target repo info data (base for merge)
        source_data: Source repo info data (data to merge from)
        settings: Merge settings with field-level rules
        
    Returns:
        Merged repo info data
    """
    # Start with target data
    merged_data = target_data.copy()
    
    # Process each field in source data according to merge rules
    for field_name, field_value in source_data.items():
        behavior = settings.get_field_behavior(field_name)
        
        if behavior == MergeBehavior.IGNORE:
            # Keep target data, ignore source
            logger.debug(f"Ignoring field '{field_name}' from source")
            continue
            
        elif behavior == MergeBehavior.OVERWRITE:
            # Replace with source data
            merged_data[field_name] = field_value
            if isinstance(field_value, list):
                logger.debug(f"Overwrote field '{field_name}' with {len(field_value)} items from source")
            else:
                logger.debug(f"Overwrote field '{field_name}' from source")
                
        elif behavior == MergeBehavior.MERGE:
            # Merge/combine data (specific logic depends on field type)
            if field_name not in target_data:
                # Field doesn't exist in target, just copy from source
                merged_data[field_name] = field_value
                logger.debug(f"Added new field '{field_name}' from source")
            else:
                merged_value = merge_field_data(
                    target_data[field_name], 
                    field_value, 
                    field_name
                )
                merged_data[field_name] = merged_value
                logger.debug(f"Merged field '{field_name}'")
    
    return merged_data


def merge_field_data(target_value: Any, source_value: Any, field_name: str) -> Any:
    """
    Merge two field values based on their types.
    
    Args:
        target_value: Value from target data
        source_value: Value from source data
        field_name: Name of the field being merged
        
    Returns:
        Merged value
    """
    # Handle list fields by combining them (removing duplicates for simple types)
    if isinstance(target_value, list) and isinstance(source_value, list):
        if not target_value:
            return source_value
        if not source_value:
            return target_value
        
        # For lists of simple types, remove duplicates
        if target_value and isinstance(target_value[0], (str, int, float)):
            combined = list(set(target_value + source_value))
            logger.debug(f"Merged list field '{field_name}': {len(target_value)} + {len(source_value)} = {len(combined)} unique items")
            return combined
        else:
            # For complex objects, just concatenate
            combined = target_value + source_value
            logger.debug(f"Merged list field '{field_name}': {len(target_value)} + {len(source_value)} = {len(combined)} items")
            return combined
    
    # Handle dict fields by merging them
    elif isinstance(target_value, dict) and isinstance(source_value, dict):
        merged_dict = target_value.copy()
        merged_dict.update(source_value)
        return merged_dict
    
    # For other types, prefer source value
    else:
        logger.debug(f"Field '{field_name}': using source value (type: {type(source_value).__name__})")
        return source_value


def process_single_file(yaml_filename: str, target_dir: Path, source_dir: Path, output_dir: Path, settings: MergeSettings) -> Dict[str, Any]:
    """
    Process a single YAML file for merging.
    
    Args:
        yaml_filename: Name of the YAML file to process
        target_dir: Directory containing target files (base for merge)
        source_dir: Directory containing source files (data to merge from)
        output_dir: Directory to save merged files
        settings: Merge settings with field-level rules
        
    Returns:
        Dictionary with processing results
    """
    result = {
        "filename": yaml_filename,
        "status": "unknown",
        "error": None,
        "target_exists": False,
        "source_exists": False,
        "merged": False
    }
    
    target_path = target_dir / yaml_filename
    source_path = source_dir / yaml_filename
    output_path = output_dir / yaml_filename
    
    # Check if files exist
    result["target_exists"] = target_path.exists()
    result["source_exists"] = source_path.exists()
    
    try:
        if not result["target_exists"] and not result["source_exists"]:
            result["status"] = "skipped"
            result["error"] = "Neither target nor source file exists"
            return result
            
        elif result["target_exists"] and not result["source_exists"]:
            # Only target exists - copy it to output
            target_data = load_yaml_file(target_path)
            if target_data and save_yaml_file(target_data, output_path):
                result["status"] = "copied_target"
            else:
                result["status"] = "error"
                result["error"] = "Failed to copy target file"
                
        elif not result["target_exists"] and result["source_exists"]:
            # Only source exists - skip merge, don't copy source-only files
            result["status"] = "skipped"
            result["error"] = "Source file has no matching target file - skipping"
                
        else:
            # Both exist - merge them
            target_data = load_yaml_file(target_path)
            source_data = load_yaml_file(source_path)
            
            if target_data is None:
                result["status"] = "error"
                result["error"] = "Failed to load target file"
                return result
                
            if source_data is None:
                result["status"] = "error" 
                result["error"] = "Failed to load source file"
                return result
                
            merged_data = merge_repo_info(target_data, source_data, settings)
            
            if save_yaml_file(merged_data, output_path):
                result["status"] = "merged"
                result["merged"] = True
            else:
                result["status"] = "error"
                result["error"] = "Failed to save merged file"
                
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"Error processing {yaml_filename}: {e}")
    
    return result


def merge_repo_info_files(settings: MergeSettings = None) -> None:
    """
    Main function to merge repo info files.
    
    Args:
        settings: Merge settings configuration
    """
    if settings is None:
        settings = DEFAULT_MERGE_SETTINGS
    
    data_dir = get_data_dir()
    target_dir = settings.get_target_path(data_dir)
    source_dir = settings.get_source_path(data_dir)
    output_dir = settings.get_output_path(data_dir)
    
    logger.info(f"Starting merge process...")
    logger.info(f"  Target directory: {target_dir}")
    logger.info(f"  Source directory: {source_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Default merge behavior: {settings.default_behavior.value}")
    logger.info(f"  Field-specific rules: {len(settings.field_rules)} configured")
    
    # Validate field coverage
    settings.validate_field_coverage()
    
    # Log field rules
    for rule in settings.field_rules:
        logger.debug(f"  - {rule.field_name}: {rule.behavior.value} ({rule.description})")
    
    # Check if directories exist
    if not target_dir.exists():
        logger.error(f"Target directory does not exist: {target_dir}")
        return
        
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all YAML files from target directory (we only process files that exist in target)
    target_files = set()
    source_files = set()
    
    if target_dir.exists():
        target_files = {f.name for f in target_dir.glob("*.yaml")}
        
    if source_dir.exists():
        source_files = {f.name for f in source_dir.glob("*.yaml")}
    
    # Only process files that exist in target directory
    files_to_process = target_files
    total_files = len(files_to_process)
    
    if total_files == 0:
        logger.warning("No YAML files found in target directory")
        return
    
    # Count how many target files have matching source files
    matching_files = target_files & source_files
    target_only_files = target_files - source_files
    
    logger.info(f"Found {len(target_files)} files in target directory")
    logger.info(f"Found {len(source_files)} files in source directory") 
    logger.info(f"Files with both target and source: {len(matching_files)} (will be merged)")
    logger.info(f"Files with target only: {len(target_only_files)} (will be copied)")
    logger.info(f"Processing {total_files} total files")
    
    # Process files in parallel
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, filename, target_dir, source_dir, output_dir, settings): filename
            for filename in files_to_process
        }
        
        # Process completed tasks with progress tracking
        with progress_manager.create_main_progress(total_files, "Merging files", "files") as pbar:
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Log progress
                    if result["status"] == "merged":
                        logger.debug(f"✓ Merged: {filename}")
                    elif result["status"] == "copied_target":
                        logger.debug(f"✓ Copied (target only): {filename}")
                    elif result["status"] == "skipped":
                        logger.debug(f"⚠ Skipped: {filename} - {result['error']}")
                    elif result["status"] == "error":
                        logger.warning(f"✗ Error: {filename} - {result['error']}")
                    
                except Exception as e:
                    logger.error(f"Task failed for {filename}: {e}")
                    results.append({
                        "filename": filename,
                        "status": "error", 
                        "error": str(e)
                    })
                
                pbar.update(1)
    
    # Summary statistics
    elapsed_time = time.time() - start_time
    stats = {}
    for result in results:
        status = result["status"]
        stats[status] = stats.get(status, 0) + 1
    
    logger.info(f"Merge completed in {elapsed_time:.2f} seconds")
    logger.info(f"Statistics: {dict(stats)}")
    
    # Show errors if any
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        logger.warning(f"Found {len(errors)} errors:")
        for error in errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error['filename']}: {error['error']}")
        if len(errors) > 5:
            logger.warning(f"  ... and {len(errors) - 5} more errors")
    
    logger.info(f"Merge process completed. Output saved to: {output_dir}")


def load_settings_from_file(settings_file: Path) -> MergeSettings:
    """Load merge settings from a YAML file."""
    try:
        with open(settings_file, 'r', encoding='utf-8') as f:
            settings_data = yaml.safe_load(f)
        return MergeSettings(**settings_data)
    except Exception as e:
        logger.error(f"Failed to load settings from {settings_file}: {e}")
        raise


def save_default_settings(output_file: Path) -> None:
    """Save default settings to a YAML file for user customization."""
    try:
        settings_dict = DEFAULT_MERGE_SETTINGS.model_dump()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(settings_dict, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"Default settings saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save default settings: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge repo info YAML files with configurable rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all default settings (no arguments needed)
  python merge_repo_info.py

  # Use custom settings file
  python merge_repo_info.py --settings merge_config.yaml

  # Generate default settings file for customization
  python merge_repo_info.py --save-defaults merge_config.yaml

  # Override specific settings via CLI
  python merge_repo_info.py --source-dir repo_info_v2 --workers 20
  
  # Enable verbose logging with defaults
  python merge_repo_info.py --verbose

Default behavior without arguments:
  - Source: data/repo_info
  - Target: data/repo_info_orig  
  - Output: data/repo_info_new
  - Workers: 10
  - Merge behavior: overwrite (with field-specific rules)
        """
    )
    
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="Path to YAML settings file (default: use built-in settings)"
    )
    parser.add_argument(
        "--save-defaults",
        type=Path,
        default=None,
        help="Save default settings to specified file and exit"
    )
    parser.add_argument(
        "--workers", 
        type=int,
        default=None,
        help="Maximum number of worker threads (default: 10, overrides settings file)"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Source directory name (default: 'repo_info', overrides settings file)"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Target directory name (default: 'repo_info_orig', overrides settings file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory name (default: 'repo_info_new', overrides settings file)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        default=False,
        help="Enable verbose logging (default: False)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Handle save-defaults option
        if args.save_defaults:
            save_default_settings(args.save_defaults)
            sys.exit(0)
        
        # Load settings
        if args.settings:
            settings = load_settings_from_file(args.settings)
        else:
            settings = DEFAULT_MERGE_SETTINGS.model_copy()
        
        # Apply CLI overrides
        if args.workers:
            settings.max_workers = args.workers
        if args.source_dir:
            settings.source_dir = args.source_dir
        if args.target_dir:
            settings.target_dir = args.target_dir
        if args.output_dir:
            settings.output_dir = args.output_dir
        
        # Run merge process
        merge_repo_info_files(settings)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)