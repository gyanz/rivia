"""Read/write HEC-RAS HDF5 output files."""

# ruff: noqa: I001
# Geometry/structure/results classes are returned by method calls, not
# user-constructed.  They are importable on demand for type annotations:
#   from rivia.hdf import FlowArea, CrossSection, Bridge  # works
# but are intentionally absent from __all__.
from .log import (  # noqa: F401
    ComputeProcess,
    RuntimeLog,
    SteadyRuntimeLog,
    UnsteadyRuntimeLog,
)
from .geometry import (  # noqa: F401
    BoundaryConditionCollection,
    BoundaryConditionLine,
    Bridge,
    CrossSection,
    CrossSectionCollection,
    FlowArea,
    FlowAreaCollection,
    GateGroup,
    GateOpening,
    Geometry,
    InlineStructure,
    LateralStructure,
    SA2DConnection,
    StorageArea,
    StorageAreaCollection,
    Structure,
    StructureCollection,
    StructureIndex,
    Weir,
)
from .steady_plan import (  # noqa: F401
    ComputeSummary as SteadyComputeSummary,
    CrossSectionResults as SteadyCrossSectionResults,
    CrossSectionResultsCollection as SteadyCrossSectionResultsCollection,
    LateralResults as SteadyLateralResults,
    RunStatus as SteadyRunStatus,
    SteadyPlan,
    StorageAreaResults as SteadyStorageAreaResults,
    StorageAreaResultsCollection as SteadyStorageAreaResultsCollection,
    StructureResultsCollection as SteadyStructureResultsCollection,
)
from .unsteady_plan import (  # noqa: F401
    BridgeResults,
    CrossSectionResults as UnsteadyCrossSectionResults,
    CrossSectionResultsCollection as UnsteadyCrossSectionResultsCollection,
    CrossSectionResultsDSS,
    CrossSectionResultsInstantaneous,
    FlowAreaResults,
    FlowAreaResultsCollection,
    InlineResults,
    LateralResults,
    SA2DConnectionResults,
    StorageAreaResults,
    StorageAreaResultsCollection,
    StructureResultsCollection,
    UnsteadyPlan,
)

__all__ = [
    "ComputeProcess",
    "Geometry",
    "RuntimeLog",
    "SteadyPlan",
    "SteadyRuntimeLog",
    "UnsteadyPlan",
    "UnsteadyRuntimeLog",
]
