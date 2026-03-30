"""rivia — A modern Python interface for HEC-RAS hydraulic modeling software."""

import logging

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("rivia")
except PackageNotFoundError:
    # Package not installed (e.g. running from source without pip install)
    from rivia._version import version as __version__  # type: ignore[no-redef]

logging.getLogger("numba").setLevel(logging.WARNING)

