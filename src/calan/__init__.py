"""Initialize package."""
from pathlib import Path
from ._version import get_versions

__version__ = get_versions()['version']
PACKAGE = Path(__file__).parent.name
DATA_PATH = Path(__file__).parent.joinpath('data')
