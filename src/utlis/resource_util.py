import importlib.util
from pathlib import Path


def installed_resources_dir() -> Path:
    # src is installed as a package and we need to locate the resources directory relative to it
    # It is required because easylocai is installed as a package and the resources directory is not in the current working directory
    spec = importlib.util.find_spec("src")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate installed package 'src'.")

    src_pkg_dir = Path(spec.origin).resolve().parent
    resources_dir = src_pkg_dir.parent / "resources"

    if not resources_dir.exists():
        raise FileNotFoundError(f"resources dir not found: {resources_dir}")

    return resources_dir
