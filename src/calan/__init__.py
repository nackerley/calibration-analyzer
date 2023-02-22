"""Initialize package."""
import os
from ._version import get_versions

__version__ = get_versions()['version']  # type: ignore [attr-defined]
PACKAGE = os.path.basename(os.path.dirname(__file__))
DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
