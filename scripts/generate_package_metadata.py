from multiprocessing.managers import ValueProxy
import os
import time
import yaml
from pathlib import Path
from typing import Dict
import logging
import asyncio
from multiprocessing import Pool, Manager, current_process
from tqdm import tqdm
import logging

from hsbriskevaluator.collector.github_collector import CollectorSettings
from hsbriskevaluator.utils.file import get_data_dir
from hsbriskevaluator.collector import collect_all
from hsbriskevaluator.utils.progress_manager import get_progress_manager

yaml.Dumper.ignore_aliases = lambda *args: True  # Disable all aliases


# Utility functions
def load_yaml(file_path: Path) -> Dict:
    """Load a YAML file into a Python dictionary."""
    if not file_path.exists():
        logging.error(f"File {file_path} does not exist.")
        exit(1)
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {file_path}: {e}")
        exit(1)


def save_yaml(data: Dict, file_path: Path):
    """Save a Python dictionary to a YAML file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(file_path, "w", 0o666) as f:
            yaml.dump(data, f, allow_unicode=True)
        logging.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logging.error(f"Error saving YAML file {file_path}: {e}")


meta_data = {}
meta_data_by_url = {}
repo_info_path = get_data_dir() / "repo_info"
all_repo_info = list(repo_info_path.glob('*.yaml'))
for path in tqdm(all_repo_info): 
    data = load_yaml(path)
    if data.get("url"):
        data["basic_info"]["url"] = data.pop("url")
        save_yaml(data, path)
    url = data.get("basic_info").get("url")
    pkt_name = data["pkt_name"]
    meta_data[pkt_name] = {
        "url" : url ,
        "sibling": []
    }
    if meta_data_by_url.get(url) == None:
        meta_data_by_url[url] = []
    meta_data_by_url[url].append(pkt_name)

for pkt_name in meta_data:
    meta_data_by_url[meta_data[pkt_name]["url"]] = list(set(meta_data_by_url[meta_data[pkt_name]["url"]]))
    meta_data[pkt_name]["sibling"] = meta_data_by_url[meta_data[pkt_name]["url"]]
save_yaml(meta_data,get_data_dir()  /"meta_data.yaml")
save_yaml(meta_data_by_url,get_data_dir() /"meta_data_by_url.yaml")


