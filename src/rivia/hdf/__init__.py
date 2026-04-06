"""Read/write HEC-RAS HDF5 output files."""

from ._geometry import (
    BoundaryConditionCollection,
    BoundaryConditionLine,
    Bridge,
    FlowArea,
    FlowAreaCollection,
    GateGroup,
    GeometryHdf,
    HdfCrossSection,
    HdfCrossSectionCollection,
    HdfGateOpening,
    Inline,
    Lateral,
    SA2DConnection,
    StorageArea,
    StorageAreaCollection,
    Structure,
    StructureCollection,
    StructureIndex,
    Weir,
)

# Backward-compatible aliases (deprecated — use Hdf-prefixed names)
CrossSection = HdfCrossSection
CrossSectionCollection = HdfCrossSectionCollection
GateOpening = HdfGateOpening
from ._steady_plan import (
    SteadyCrossSectionResults,
    SteadyCrossSectionResultsCollection,
    SteadyLateralResults,
    SteadyPlanHdf,
    SteadyStorageAreaResults,
    SteadyStorageAreaResultsCollection,
    SteadyStructureCollection,
)
from ._unsteady_plan import (
    BridgeResults,
    FlowAreaResults,
    FlowAreaResultsCollection,
    InlineResults,
    LateralResults,
    PlanStructureCollection,
    SA2DConnectionResults,
    StorageAreaResults,
    StorageAreaResultsCollection,
    UnsteadyCrossSectionResults,
    UnsteadyCrossSectionResultsCollection,
    UnsteadyCrossSectionResultsDss,
    UnsteadyCrossSectionResultsInst,
    UnsteadyPlanHdf,
)

# Backward-compatible aliases for renamed unsteady results classes
CrossSectionResults = UnsteadyCrossSectionResults
CrossSectionResultsDss = UnsteadyCrossSectionResultsDss
CrossSectionResultsInst = UnsteadyCrossSectionResultsInst
CrossSectionResultsCollection = UnsteadyCrossSectionResultsCollection

__all__ = [
    "GeometryHdf",
    "SteadyPlanHdf",
    "SteadyCrossSectionResults",
    "SteadyCrossSectionResultsCollection",
    "SteadyLateralResults",
    "SteadyStorageAreaResults",
    "SteadyStorageAreaResultsCollection",
    "SteadyStructureCollection",
    "UnsteadyPlanHdf",
    # HDF geometry classes
    "HdfCrossSection",
    "HdfCrossSectionCollection",
    "HdfGateOpening",
    # Unsteady results classes
    "UnsteadyCrossSectionResults",
    "UnsteadyCrossSectionResultsDss",
    "UnsteadyCrossSectionResultsInst",
    "UnsteadyCrossSectionResultsCollection",
    "FlowAreaCollection",
    "FlowAreaResultsCollection",
    "FlowArea",
    "FlowAreaResults",
    "StorageArea",
    "StorageAreaCollection",
    "StorageAreaResults",
    "StorageAreaResultsCollection",
    "SA2DConnectionResults",
    "PlanStructureCollection",
    "BridgeResults",
    "InlineResults",
    "LateralResults",
    "BoundaryConditionLine",
    "BoundaryConditionCollection",
    "Structure",
    "Bridge",
    "Inline",
    "Lateral",
    "SA2DConnection",
    "StructureCollection",
    "StructureIndex",
    "Weir",
    "GateGroup",
    # Backward-compatible aliases
    "CrossSection",
    "CrossSectionCollection",
    "GateOpening",
]
