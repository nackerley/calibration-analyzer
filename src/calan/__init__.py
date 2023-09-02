"""Initialize package."""
from pathlib import Path
from ._version import get_versions

__version__ = get_versions()['version']
PACKAGE = Path(__file__).parts[-1]
DATA_PATH = Path(__file__).absolute().joinpath('data')
