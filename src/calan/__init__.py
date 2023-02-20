"""Initialize package."""
import os
from . import _version

__version__ = _version.get_versions()['version']
PACKAGE = os.path.basename(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
