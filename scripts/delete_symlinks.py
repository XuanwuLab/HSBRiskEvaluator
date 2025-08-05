#! /usr/bin/env python3

from pathlib import Path
from typing import Dict
from hsbriskevaluator.collector.repo_info import RepoInfo
from hsbriskevaluator.utils.file import get_data_dir
import yaml
import logging
logger = logging.getLogger(__name__)

def load_yaml(file_path: Path) -> Dict:
    """Load a YAML file into a Python dictionary."""
    if not file_path.exists():
        logger.error(f"File {file_path} does not exist.")
        exit(1)
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        exit(1)


def save_yaml(data: Dict, file_path: Path):
    """Save a Python dictionary to a YAML file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(file_path, "w", 0o666) as f:
            yaml.dump(data, f, allow_unicode=True)
        logger.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving YAML file {file_path}: {e}")


path = get_data_dir() / "repo_info"
def add_pkt_name(target: Path, link: Path):
    try:
        info = RepoInfo.model_validate(load_yaml(target))
        if isinstance(info.pkt_name, str):
            info.pkt_name = [info.pkt_name]
        if link.name[:-5] not in info.pkt_name:
            info.pkt_name.append(link.name[:-5])
            logger.info(f"Adding pkt_name {link.name[:-5]} to {target.name}")
        save_yaml(info.model_dump(), target)

    except Exception as e:
        logger.error(f"Error processing {target}: {e}")

for file in path.iterdir():
    if file.is_symlink():
        target = file.resolve()
        if target.exists():
            add_pkt_name(target, file)
            logger.info(f"Deleting symlink: {file} -> {target}")
            file.unlink()
        else:
            logger.info(f"Symlink target does not exist, deleting symlink: {file} -> {target}")
            file.unlink()
