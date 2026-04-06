"""rivia — A modern Python interface for HEC-RAS hydraulic modeling software."""

import logging

from rivia.hdf import GeometryHdf, SteadyPlanHdf, UnsteadyPlanHdf
from rivia.model import Model, ProjectFile, GeometryFile, PlanFile, SteadyFlowFile, UnsteadyFlowFile 

try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("rivia")
except PackageNotFoundError:
    # Package not installed (e.g. running from source without pip install)
    from rivia._version import version as __version__  # type: ignore[no-redef]  # noqa: F401, I001

logging.getLogger("numba").setLevel(logging.WARNING)

__all__ = ["Model", "ProjectFile", "GeometryFile", "PlanFile", "SteadyFlowFile", "UnsteadyFlowFile", 
           "GeometryHdf", "SteadyPlanHdf", "UnsteadyPlanHdf"]
