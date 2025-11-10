import os

def normalize_path(path: str) -> str:
    """
    Normalize a path by expanding ~ and resolving relative paths.
    Handles common directory names by mapping them to home directory paths.
    """
    # Map common directory names to home paths
    common_dirs = {
        "documents": "~/Documents",
        "desktop": "~/Desktop",
        "downloads": "~/Downloads",
        "pictures": "~/Pictures",
        "videos": "~/Videos",
        "music": "~/Music",
        "home": "~",
        "application documents": "~/Documents/APPLICATION DOCUMENTS",
        "application documents folder": "~/Documents/APPLICATION DOCUMENTS",
    }

    # Check if the path is a common directory name
    lower_path = path.lower().strip()
    if lower_path in common_dirs:
        path = common_dirs[lower_path]

    # Expand ~ and resolve to absolute path
    return os.path.abspath(os.path.expanduser(path))

def expand_home_path(path: str) -> str:
    """
    Expand ~ in path to home directory.
    """
    return os.path.expanduser(path)
