from pathlib import Path
import os
from pygments.lexers import guess_lexer_for_filename
from pygments.util import ClassNotFound


def get_data_dir():
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_cache_dir():
    cache_dir = Path(__file__).parent.parent.parent.parent / "cache"
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def is_binary(file_path: str) -> bool:
    if os.path.isdir(file_path):
        return False
    with open(file_path, "rb") as f:
        data = f.read(1024)
    if not data:
        return False
    if b"\x00" in data:
        return True
    allowed_controls = {9, 10, 13}
    non_text = 0
    for byte in data:
        if byte < 32 and byte not in allowed_controls or byte == 127:
            non_text += 1
    ratio = non_text / len(data)
    return ratio > 0.3


def detect_language(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            content = f.read(4096)
        lexer = guess_lexer_for_filename(file_path, content)
        return lexer.name
    except ClassNotFound:
        return "Unknown"
