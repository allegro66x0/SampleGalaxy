import os
import json

def normalize_path(path):
    """Normalize file path for consistent comparison."""
    if path is None:
        return None
    return os.path.normpath(path)

def load_json(path):
    """Load JSON file safely."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON {path}: {e}")
        return None

def save_json(path, data):
    """Save data to JSON file."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Failed to save JSON {path}: {e}")
        return False
