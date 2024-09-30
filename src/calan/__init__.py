"""Initialize package."""
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

if __package__ == '':
    # pylint: disable=redefined-builtin
    __package__ = Path(__file__).parent.name

try:
    __version__ = version(__package__)
except PackageNotFoundError:
    # package is not installed
    pass

PACKAGE_VERSION = f'{__package__} v{__version__}'
DATA_PATH = Path(__file__).parent.joinpath('data')
