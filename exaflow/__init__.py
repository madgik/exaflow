import hashlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict
from typing import List
from typing import Tuple

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm as ExaflowAlgorithm
from exaflow.algorithms.exareme3.utils.preprocessing_step import (
    PreprocessingStep as ExaflowPreprocessingStep,
)
from exaflow.datatypes import DType
from exaflow.utils import AttrDict

__all__ = [
    "DType",
    "AttrDict",
    "flower_algorithm_folder_paths",
    "FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE",
    "FLOWER_ALGORITHM_FOLDERS",
    "EXAREME3_ALGORITHM_FOLDERS_ENV_VARIABLE",
    "EXAREME3_ALGORITHM_FOLDERS",
    "exareme3_algorithm_classes",
    "exareme3_preprocessing_step_classes",
]


def _resolve_package_import(module_path: str):
    """
    Try to derive a canonical dotted module path for ``module_path`` by walking
    upwards while __init__.py files are present. Returns ``(import_path,
    package_root)`` when found, otherwise ``(None, None)`` to signal that the
    module lives outside a package.
    """
    module_dir = os.path.dirname(module_path)
    package_parts = []
    search_dir = module_dir

    while os.path.isfile(os.path.join(search_dir, "__init__.py")):
        package_parts.append(os.path.basename(search_dir))
        search_dir = os.path.dirname(search_dir)

    if not package_parts:
        return None, None

    package_parts.reverse()
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    import_path = ".".join(package_parts + [module_name])
    package_root = os.path.abspath(search_dir or os.curdir)
    return import_path, package_root


def _module_key_for_path(module_path: Path, root_folder: Path) -> str:
    """Create a deterministic module key scoped to the folder root."""
    relative = module_path.relative_to(root_folder)
    return ".".join(relative.with_suffix("").parts)


def _iter_algorithm_module_paths(algorithm_folder: str) -> List[Tuple[Path, str]]:
    """
    Recursively collect python module paths under ``algorithm_folder``.
    Excludes __init__.py files and non-runtime directories.
    """
    root_folder = Path(algorithm_folder).resolve()
    if not root_folder.is_dir():
        return []

    excluded_dir_names = {"__pycache__", "docs"}
    module_paths = []
    for module_path in sorted(root_folder.rglob("*.py")):
        if module_path.name == "__init__.py":
            continue
        if excluded_dir_names.intersection(module_path.parts):
            continue
        module_paths.append(
            (module_path, _module_key_for_path(module_path, root_folder))
        )
    return module_paths


def import_algorithm_modules(algorithm_folders: str) -> Dict[str, ModuleType]:
    """
    Import all algorithm modules from the given folder paths.

    :param algorithm_folders: Comma-separated string of folder paths.
    :return: A dictionary mapping module names to imported module objects.
    """
    all_modules = {}
    for algorithm_folder in algorithm_folders.split(","):
        modules = {}
        for module_path_obj, module_key in _iter_algorithm_module_paths(
            algorithm_folder
        ):
            module_path = str(module_path_obj)
            import_path, package_root = _resolve_package_import(module_path)
            module_obj = None

            if import_path:
                if package_root not in sys.path:
                    sys.path.append(package_root)
                try:
                    module_obj = importlib.import_module(import_path)
                except ModuleNotFoundError:
                    module_obj = None

            if module_obj is None:
                # When loading from non-package folders we must ensure we don't
                # re-exec the same module path, otherwise decorators may
                # double-register (e.g. @exareme3_udf) and crash startup.
                if module_path in _MODULES_BY_ABSPATH:
                    module_obj = _MODULES_BY_ABSPATH[module_path]
                else:
                    # Use a path-hashed synthetic module name to avoid collisions
                    # between equal basenames from different subfolders.
                    synthetic_module_name = (
                        f"exaflow_dynamic_"
                        f"{hashlib.sha1(module_path.encode('utf-8')).hexdigest()}"
                    )
                    spec = importlib.util.spec_from_file_location(
                        synthetic_module_name, module_path
                    )
                    module_obj = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module_obj)
                    _MODULES_BY_ABSPATH[module_path] = module_obj

            if module_key in modules and modules[module_key] is not module_obj:
                raise ValueError(
                    f"Duplicate module key '{module_key}' under folder '{algorithm_folder}'."
                )
            modules[module_key] = module_obj
        all_modules.update(modules)
    return all_modules


def find_flower_algorithm_folder_paths(algorithm_folders):
    # Split the input string into a list of folder paths
    folder_paths = algorithm_folders.split(",")

    # Initialize an empty dictionary to store the result
    algorithm_folder_paths = {}

    # Iterate over each folder path
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            continue  # Skip if the path is not a valid directory

        # List all files and folders in the current folder path
        items = os.listdir(folder_path)

        # Filter for .json files and corresponding folders
        for item in items:
            if item.endswith(".json"):
                algorithm_name = item[:-5]  # Remove '.json' to get the algorithm name
                algorithm_folder = os.path.join(folder_path, algorithm_name)
                if os.path.isdir(algorithm_folder):
                    # Store the algorithm name and the complete folder path in the dictionary
                    algorithm_folder_paths[algorithm_name] = algorithm_folder

    return algorithm_folder_paths


FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE = "FLOWER_ALGORITHM_FOLDERS"
FLOWER_ALGORITHM_FOLDERS = "./exaflow/algorithms/flower"
if flower_algorithm_folders := os.getenv(FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE):
    FLOWER_ALGORITHM_FOLDERS = flower_algorithm_folders

flower_algorithm_folder_paths = find_flower_algorithm_folder_paths(
    FLOWER_ALGORITHM_FOLDERS
)


EXAREME3_ALGORITHM_FOLDERS_ENV_VARIABLE = "EXAREME3_ALGORITHM_FOLDERS"
EXAREME3_ALGORITHM_FOLDERS = os.getenv(
    EXAREME3_ALGORITHM_FOLDERS_ENV_VARIABLE, "./exaflow/algorithms/exareme3"
)

_MODULES_BY_ABSPATH: Dict[str, ModuleType] = {}
_EXAREME3_MODULES_LOADED = False


def _ensure_exareme3_modules_loaded() -> None:
    global _EXAREME3_MODULES_LOADED
    if _EXAREME3_MODULES_LOADED:
        return
    import_algorithm_modules(EXAREME3_ALGORITHM_FOLDERS)
    _EXAREME3_MODULES_LOADED = True


def get_exareme3_algorithm_classes() -> Dict[str, type]:
    _ensure_exareme3_modules_loaded()
    return {
        cls.get_specification().name: cls for cls in ExaflowAlgorithm.__subclasses__()
    }


exareme3_algorithm_classes = get_exareme3_algorithm_classes()


def get_exareme3_preprocessing_step_classes() -> Dict[str, type]:
    _ensure_exareme3_modules_loaded()
    return {
        cls.get_specification().name: cls
        for cls in ExaflowPreprocessingStep.__subclasses__()
    }


exareme3_preprocessing_step_classes = get_exareme3_preprocessing_step_classes()
