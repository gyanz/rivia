"""rivia — A modern Python interface for HEC-RAS hydraulic modeling software."""

import logging

from rivia.model import Project
from rivia import controller, geo, hdf, model

try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("rivia")
except PackageNotFoundError:
    from rivia._version import version as __version__  # type: ignore[no-redef]  # noqa: F401, I001

logging.getLogger("numba").setLevel(logging.WARNING)

__all__ = ["Project", "controller", "geo", "hdf", "model"]
