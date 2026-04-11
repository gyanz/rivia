"""SteadyPlan - read HEC-RAS steady-flow plan HDF5 files (.p*.hdf).

Steady plan HDF files embed the same ``Geometry/`` group as geometry HDF files
*plus* ``Results/Steady/...`` profile-based output.

``SteadyPlan`` inherits ``Geometry`` so all geometry accessors are
available.  ``CrossSectionResults``, ``StorageAreaResults``, and
``LateralResults`` carry geometry attributes *and* steady-profile result
arrays.  All result arrays have shape ``(n_profiles,)`` (or ``(n_profiles,
n_segments)`` for segment-level lateral data) where the first axis corresponds
to a named steady-flow profile (e.g. ``"Big"``, ``"Bigger"``, ``"Biggest"``).

Bridge, culvert, and inline structure nodes in HEC-RAS 1D steady flow do not
produce separate result datasets in the HDF file; their results are embedded in
the adjacent upstream/downstream cross-section output.  ``SteadyPlan``
therefore returns plain geometry objects (no result access) for those types.
Only :class:`LateralResults` carries dedicated result data.

Derived from examination of HEC-RAS 6.6 steady-flow plan HDF output at
``Results/Steady/Output/Output Blocks/Base Output/Steady Profiles/``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, overload

import numpy as np

from ._base import _PlanHdf
from .geometry import (
    _SA_ROOT,
    CrossSection,
    CrossSectionCollection,
    Geometry,
    LateralStructure,
    StorageArea,
    StorageAreaCollection,
    Structure,
    _decode,
)
from .geometry import (
    StructureCollection as _GeomStructureCollection,
)
from .log import SteadyRuntimeLog

if TYPE_CHECKING:
    import h5py

logger = logging.getLogger("rivia.hdf")


# ---------------------------------------------------------------------------
# HDF path constants
# ---------------------------------------------------------------------------

_STEADY_ROOT = "Results/Steady/Output"
_STEADY_GEOM_ATTRS = f"{_STEADY_ROOT}/Geometry Info/Cross Section Attributes"
_STEADY_PROFILES_ROOT = f"{_STEADY_ROOT}/Output Blocks/Base Output/Steady Profiles"
_STEADY_PROFILE_NAMES = f"{_STEADY_PROFILES_ROOT}/Profile Names"
_STEADY_XS = f"{_STEADY_PROFILES_ROOT}/Cross Sections"
_STEADY_SA = f"{_STEADY_PROFILES_ROOT}/Storage Areas"
_STEADY_LATERAL = f"{_STEADY_PROFILES_ROOT}/Lateral Structures"


# ---------------------------------------------------------------------------
# CrossSectionResults
# ---------------------------------------------------------------------------


class CrossSectionResults(CrossSection):
    """Geometry *and* steady-profile results for one 1-D cross section.

    Inherits all geometry attributes from :class:`~rivia.hdf.CrossSection`.

    All result properties return a ``numpy`` array of shape ``(n_profiles,)``
    where each index corresponds to a named steady-flow profile.  Use
    :attr:`SteadyPlan.profile_names` to map indices to profile names.

    Parameters
    ----------
    geom:
        Geometry object from :class:`~rivia.hdf.CrossSectionCollection`.
    hdf:
        Open ``h5py.File`` — kept alive by the parent ``SteadyPlan`` context.
    index:
        Column index of this XS in the ``(n_profiles, n_xs)`` result datasets.
    root:
        HDF path prefix — ``_STEADY_XS``.
    """

    def __init__(
        self,
        geom: CrossSection,
        hdf: "h5py.File",
        index: int,
        root: str,
    ) -> None:
        CrossSection.__init__(
            self,
            river=geom.river,
            reach=geom.reach,
            rs=geom.rs,
            name=geom.name,
            left_bank=geom.left_bank,
            right_bank=geom.right_bank,
            len_left=geom.len_left,
            len_channel=geom.len_channel,
            len_right=geom.len_right,
            contraction=geom.contraction,
            expansion=geom.expansion,
            station_elevation=geom.station_elevation,
            mannings_n=geom.mannings_n,
            centerline=geom.centerline,
        )
        self._hdf = hdf
        self._index = index
        self._root = root
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    def _load(self, dataset: str) -> np.ndarray:
        """Load column ``self._index`` from ``{root}/{dataset}``, cached.

        Parameters
        ----------
        dataset:
            Path relative to ``self._root``, e.g. ``"Water Surface"`` or
            ``"Additional Variables/Flow Total"``.

        Returns
        -------
        ndarray, shape ``(n_profiles,)``
        """
        if dataset not in self._cache:
            ds = self._hdf.get(f"{self._root}/{dataset}")
            if ds is None:
                raise KeyError(
                    f"Dataset '{dataset}' not found at '{self._root}'."
                )
            self._cache[dataset] = np.array(ds[:, self._index])
        return self._cache[dataset]

    # ------------------------------------------------------------------
    # Top-level datasets
    # ------------------------------------------------------------------

    @property
    def wse(self) -> np.ndarray:
        """Water surface elevation.  Shape ``(n_profiles,)``."""
        return self._load("Water Surface")

    @property
    def flow(self) -> np.ndarray:
        """Total flow.  Shape ``(n_profiles,)``."""
        return self._load("Flow")

    @property
    def energy_grade(self) -> np.ndarray:
        """Energy grade line elevation.  Shape ``(n_profiles,)``."""
        return self._load("Energy Grade")

    # ------------------------------------------------------------------
    # Additional Variables — generic accessor
    # ------------------------------------------------------------------

    def additional_variable(self, name: str) -> np.ndarray:
        """Load one column from the ``Additional Variables`` sub-group.

        Parameters
        ----------
        name:
            Dataset name inside ``Additional Variables/``, e.g.
            ``"Flow Total"``, ``"Velocity Channel"``, ``"Conveyance Total"``.

        Returns
        -------
        ndarray, shape ``(n_profiles,)``

        Raises
        ------
        KeyError
            If *name* is not present in ``Additional Variables/``.
        """
        return self._load(f"Additional Variables/{name}")

    # ------------------------------------------------------------------
    # Additional Variables — explicit properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> np.ndarray:
        """Velocity-head correction factor α.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Alpha")

    @property
    def beta(self) -> np.ndarray:
        """Momentum correction factor β.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Beta")

    @property
    def flow_area_channel(self) -> np.ndarray:
        """Channel flow area.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Area Flow Channel")

    @property
    def flow_area_left_ob(self) -> np.ndarray:
        """Left overbank flow area.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Area Flow Left OB")

    @property
    def flow_area_right_ob(self) -> np.ndarray:
        """Right overbank flow area.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Area Flow Right OB")

    @property
    def flow_area_total(self) -> np.ndarray:
        """Total flow area.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Area Flow Total")

    @property
    def ineffective_area_channel(self) -> np.ndarray:
        """Channel area including ineffective zones.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Area including Ineffective Channel")

    @property
    def ineffective_area_left_ob(self) -> np.ndarray:
        """Left overbank area including ineffective zones.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Area including Ineffective Left OB")

    @property
    def ineffective_area_right_ob(self) -> np.ndarray:
        """Right overbank area including ineffective zones.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Area including Ineffective Right OB")

    @property
    def ineffective_area_total(self) -> np.ndarray:
        """Total area including ineffective zones.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Area including Ineffective Total")

    @property
    def conveyance_channel(self) -> np.ndarray:
        """Channel conveyance.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Conveyance Channel")

    @property
    def conveyance_left_ob(self) -> np.ndarray:
        """Left overbank conveyance.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Conveyance Left OB")

    @property
    def conveyance_right_ob(self) -> np.ndarray:
        """Right overbank conveyance.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Conveyance Right OB")

    @property
    def conveyance_total(self) -> np.ndarray:
        """Total conveyance.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Conveyance Total")

    @property
    def critical_energy_grade(self) -> np.ndarray:
        """Critical energy grade line elevation.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Critical Energy Grade")

    @property
    def critical_water_surface(self) -> np.ndarray:
        """Critical water surface elevation.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Critical Water Surface")

    @property
    def energy_grade_slope(self) -> np.ndarray:
        """Energy grade slope (m/m or ft/ft).  Shape ``(n_profiles,)``."""
        return self.additional_variable("EG Slope")

    @property
    def friction_slope(self) -> np.ndarray:
        """Friction slope.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Friction Slope")

    @property
    def flow_channel(self) -> np.ndarray:
        """Channel flow.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Flow Channel")

    @property
    def flow_left_ob(self) -> np.ndarray:
        """Left overbank flow.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Flow Left OB")

    @property
    def flow_right_ob(self) -> np.ndarray:
        """Right overbank flow.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Flow Right OB")

    @property
    def flow_total(self) -> np.ndarray:
        """Total flow.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Flow Total")

    @property
    def hydraulic_depth_channel(self) -> np.ndarray:
        """Channel hydraulic depth.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Hydraulic Depth Channel")

    @property
    def hydraulic_depth_left_ob(self) -> np.ndarray:
        """Left overbank hydraulic depth.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Hydraulic Depth Left OB")

    @property
    def hydraulic_depth_right_ob(self) -> np.ndarray:
        """Right overbank hydraulic depth.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Hydraulic Depth Right OB")

    @property
    def hydraulic_depth_total(self) -> np.ndarray:
        """Total hydraulic depth.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Hydraulic Depth Total")

    @property
    def hydraulic_radius_channel(self) -> np.ndarray:
        """Channel hydraulic radius.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Hydraulic Radius Channel")

    @property
    def hydraulic_radius_left_ob(self) -> np.ndarray:
        """Left overbank hydraulic radius.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Hydraulic Radius Left OB")

    @property
    def hydraulic_radius_right_ob(self) -> np.ndarray:
        """Right overbank hydraulic radius.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Hydraulic Radius Right OB")

    @property
    def hydraulic_radius_total(self) -> np.ndarray:
        """Total hydraulic radius.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Hydraulic Radius Total")

    @property
    def mannings_n_channel(self) -> np.ndarray:
        """Weighted/composite channel Manning's n.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Manning n Channel")

    @property
    def mannings_n_left_ob(self) -> np.ndarray:
        """Left overbank Manning's n.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Manning n Left OB")

    @property
    def mannings_n_right_ob(self) -> np.ndarray:
        """Right overbank Manning's n.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Manning n Right OB")

    @property
    def mannings_n_total(self) -> np.ndarray:
        """Total weighted Manning's n.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Manning n Total")

    @property
    def max_depth_total(self) -> np.ndarray:
        """Total maximum water depth.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Maximum Depth Total")

    @property
    def shear(self) -> np.ndarray:
        """Bed shear stress.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Shear")

    @property
    def top_width_channel(self) -> np.ndarray:
        """Channel top width.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Top Width Channel")

    @property
    def top_width_channel_with_ineffective(self) -> np.ndarray:
        """Channel top width including ineffective areas.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Top Width Channel including Ineffective")

    @property
    def top_width_left_ob(self) -> np.ndarray:
        """Left overbank top width.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Top Width Left OB")

    @property
    def top_width_left_ob_with_ineffective(self) -> np.ndarray:
        """Left overbank top width including ineffective areas.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Top Width Left OB including Ineffective")

    @property
    def top_width_right_ob(self) -> np.ndarray:
        """Right overbank top width.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Top Width Right OB")

    @property
    def top_width_right_ob_with_ineffective(self) -> np.ndarray:
        """Right overbank top width including ineffective areas.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Top Width Right OB including Ineffective")

    @property
    def top_width_total(self) -> np.ndarray:
        """Total top width.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Top Width Total")

    @property
    def top_width_total_with_ineffective(self) -> np.ndarray:
        """Total top width including ineffective areas.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Top Width Total including Ineffective")

    @property
    def velocity_channel(self) -> np.ndarray:
        """Channel velocity.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Velocity Channel")

    @property
    def velocity_left_ob(self) -> np.ndarray:
        """Left overbank velocity.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Velocity Left OB")

    @property
    def velocity_right_ob(self) -> np.ndarray:
        """Right overbank velocity.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Velocity Right OB")

    @property
    def velocity_total(self) -> np.ndarray:
        """Total velocity.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Velocity Total")

    @property
    def wse_total(self) -> np.ndarray:
        """Total stage / water surface elevation from ``Additional Variables``.

        Sourced from ``Additional Variables/Water Surface Total``.  Equivalent
        to :attr:`wse` for most configurations.
        Shape ``(n_profiles,)``.
        """
        return self.additional_variable("Water Surface Total")

    @property
    def wetted_perimeter_channel(self) -> np.ndarray:
        """Channel wetted perimeter.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Wetted Perimeter Channel")

    @property
    def wetted_perimeter_left_ob(self) -> np.ndarray:
        """Left overbank wetted perimeter.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Wetted Perimeter Left OB")

    @property
    def wetted_perimeter_right_ob(self) -> np.ndarray:
        """Right overbank wetted perimeter.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Wetted Perimeter Right OB")

    @property
    def wetted_perimeter_total(self) -> np.ndarray:
        """Total wetted perimeter.  Shape ``(n_profiles,)``."""
        return self.additional_variable("Wetted Perimeter Total")


# ---------------------------------------------------------------------------
# CrossSectionResultsCollection
# ---------------------------------------------------------------------------


class CrossSectionResultsCollection(CrossSectionCollection):
    """Steady-plan cross section collection with profile results.

    Combines geometry from ``Geometry/Cross Sections`` with steady-flow
    result data from ``Results/Steady/Output/...``.  Each item is a
    :class:`CrossSectionResults` instance.

    Parameters
    ----------
    hdf:
        Open ``h5py.File`` handle.
    """

    def __init__(self, hdf: "h5py.File") -> None:
        super().__init__(hdf)
        self._result_items: dict[str, CrossSectionResults] | None = None

    def _load_results(self) -> dict[str, CrossSectionResults]:
        if self._result_items is not None:
            return self._result_items

        geom_items = CrossSectionCollection._load(self)

        attrs_ds = self._hdf.get(_STEADY_GEOM_ATTRS)
        if attrs_ds is None:
            self._result_items = {}
            return self._result_items

        result_attrs = np.array(attrs_ds)
        fn = attrs_ds.dtype.names

        result_index: dict[tuple[str, str, str], int] = {}
        for i, row in enumerate(result_attrs):
            r  = _decode(row["River"])   if "River"   in fn else ""
            rc = _decode(row["Reach"])   if "Reach"   in fn else ""
            st = _decode(row["Station"]) if "Station" in fn else ""
            result_index[(r, rc, st)] = i

        items: dict[str, CrossSectionResults] = {}
        for key, geom in geom_items.items():
            idx = result_index.get((geom.river, geom.reach, geom.rs))
            if idx is not None:
                items[key] = CrossSectionResults(
                    geom, self._hdf, idx, _STEADY_XS
                )

        self._result_items = items
        return self._result_items

    @overload
    def __getitem__(self, key: int) -> CrossSectionResults: ...
    @overload
    def __getitem__(self, key: str) -> CrossSectionResults: ...
    @overload
    def __getitem__(self, key: tuple[str, str, str]) -> CrossSectionResults: ...

    def __getitem__(
        self, key: int | str | tuple[str, str, str]
    ) -> CrossSectionResults:
        items = self._load_results()
        if isinstance(key, int):
            keys = list(items)
            try:
                return items[keys[key]]
            except IndexError:
                raise IndexError(
                    f"Index {key} out of range (n={len(items)})"
                ) from None
        if isinstance(key, tuple):
            str_key = self._loc_index.get(key)
            if str_key is None:
                raise KeyError(f"Cross section {key!r} not found.")
            if str_key not in items:
                raise KeyError(
                    f"Cross section {key!r} has no results in this plan."
                )
            return items[str_key]
        if key not in items:
            raise KeyError(
                f"Cross section {key!r} not found. Available: {self.names}"
            )
        return items[key]

    def __len__(self) -> int:
        return len(self._load_results())

    def __iter__(self) -> Iterator[CrossSectionResults]:
        return iter(self._load_results().values())

    @property
    def names(self) -> list[str]:
        """Keys of all cross sections with steady results."""
        return list(self._load_results().keys())


# ---------------------------------------------------------------------------
# StorageAreaResults
# ---------------------------------------------------------------------------


class StorageAreaResults(StorageArea):
    """Geometry *and* steady-profile results for one storage area.

    Inherits all geometry properties from :class:`~rivia.hdf.StorageArea`.

    All result properties return a ``numpy`` array of shape ``(n_profiles,)``
    where each index corresponds to a named steady-flow profile.  Use
    :attr:`SteadyPlan.profile_names` to map indices to profile names.

    Parameters
    ----------
    sa:
        Parent geometry object whose fields are copied into this instance.
    sa_index:
        0-based column index of this SA in the ``(n_profiles, n_sa)`` datasets
        ``Water Surface`` and ``Flow`` stored under
        ``Results/Steady/.../Storage Areas``.
    sa_group:
        ``h5py.Group`` at ``Results/Steady/.../Steady Profiles/Storage Areas``,
        or ``None`` when the plan has no storage-area results.
    """

    def __init__(
        self,
        sa: StorageArea,
        sa_index: int,
        sa_group: "h5py.Group | None",
    ) -> None:
        super().__init__(
            name=sa.name,
            mode=sa.mode,
            boundary=sa.boundary,
            volume_elevation=sa.volume_elevation,
        )
        self._i = sa_index
        self._g = sa_group
        self._sub = sa_group.get(sa.name) if sa_group else None
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_flat(self, key: str) -> np.ndarray:
        """Load column *i* from a flat ``(n_profiles, n_sa)`` dataset."""
        if key not in self._cache:
            if self._g is None:
                raise KeyError(
                    f"No steady results for storage area {self.name!r}. "
                    "Has the plan been computed?"
                )
            self._cache[key] = np.array(self._g[key])[:, self._i]
        return self._cache[key]

    def _load_vars(self) -> np.ndarray:
        """Load and cache the ``(n_profiles, n_cols)`` Storage Area Variables array."""
        if "_vars" not in self._cache:
            if self._sub is None or "Storage Area Variables" not in self._sub:
                raise KeyError(
                    f"'Storage Area Variables' not found for storage area "
                    f"{self.name!r}."
                )
            self._cache["_vars"] = np.array(self._sub["Storage Area Variables"])
        return self._cache["_vars"]

    # ------------------------------------------------------------------
    # Flat profile results (one value per profile)
    # ------------------------------------------------------------------

    @property
    def wse(self) -> np.ndarray:
        """Water-surface elevation.  Shape ``(n_profiles,)``."""
        return self._load_flat("Water Surface")

    @property
    def flow(self) -> np.ndarray:
        """Net flow.  Shape ``(n_profiles,)``."""
        return self._load_flat("Flow")

    # ------------------------------------------------------------------
    # Connection flows
    # ------------------------------------------------------------------

    @property
    def connections(self) -> np.ndarray | None:
        """Flow from each named connection.

        Shape ``(n_profiles, n_conns)``, or ``None`` when absent.
        Column names are in :attr:`connection_names`.
        """
        if "_conns" not in self._cache:
            if self._sub is None:
                return None
            ds = self._sub.get("Connections to Storage Area")
            if ds is None:
                return None
            self._cache["_conns"] = np.array(ds)
        return self._cache["_conns"]

    @property
    def connection_names(self) -> list[str]:
        """Names of the inflow connection sources.

        Falls back to index-based names if the attribute is absent.
        """
        if self._sub is None:
            return []
        ds = self._sub.get("Connections to Storage Area")
        if ds is None:
            return []
        attr = ds.attrs.get("Connections")
        if attr is None:
            n = ds.shape[1] if ds.ndim > 1 else 1
            return [f"connection_{i}" for i in range(n)]
        return [_decode(v) for v in attr]


# ---------------------------------------------------------------------------
# StorageAreaResultsCollection
# ---------------------------------------------------------------------------


class StorageAreaResultsCollection(StorageAreaCollection):
    """Collection of :class:`StorageAreaResults` backed by a steady plan HDF.

    Overrides :class:`~rivia.hdf.StorageAreaCollection` to return
    ``StorageAreaResults`` with both geometry *and* steady results.
    """

    def _load(self) -> dict[str, StorageAreaResults]:  # type: ignore[override]
        if self._items is not None:
            return self._items  # type: ignore[return-value]

        if _SA_ROOT not in self._hdf:
            self._items = {}
            return self._items  # type: ignore[return-value]

        root = self._hdf[_SA_ROOT]
        attrs = np.array(root["Attributes"])
        poly_info = np.array(root["Polygon Info"])
        poly_pts = np.array(root["Polygon Points"])
        ve_info = np.array(root["Volume Elevation Info"])
        ve_vals = np.array(root["Volume Elevation Values"])

        sa_group = self._hdf.get(_STEADY_SA)

        items: dict[str, StorageAreaResults] = {}
        for i, row in enumerate(attrs):
            name = _decode(row["Name"])
            mode = _decode(row["Mode"])

            start_pt = int(poly_info[i, 0])
            n_pts = int(poly_info[i, 1])
            boundary = poly_pts[start_pt : start_pt + n_pts].astype(float)

            ve_start = int(ve_info[i, 0])
            ve_count = int(ve_info[i, 1])
            volume_elevation = ve_vals[ve_start : ve_start + ve_count].astype(float)

            sa_geom = StorageArea(
                name=name,
                mode=mode,
                boundary=boundary,
                volume_elevation=volume_elevation,
            )
            items[name] = StorageAreaResults(sa_geom, i, sa_group)

        self._items = items  # type: ignore[assignment]
        return self._items  # type: ignore[return-value]

    def __getitem__(self, key: int | str) -> StorageAreaResults:
        items = self._load()
        if isinstance(key, int):
            keys = list(items)
            try:
                return items[keys[key]]
            except IndexError:
                raise IndexError(
                    f"Index {key} out of range (n={len(items)})"
                ) from None
        if key not in items:
            raise KeyError(
                f"Storage area {key!r} not found. Available: {self.names}"
            )
        return items[key]

    @property
    def names(self) -> list[str]:
        """Names of all storage areas in the collection."""
        return list(self._load().keys())


# ---------------------------------------------------------------------------
# LateralResults
# ---------------------------------------------------------------------------


class LateralResults(LateralStructure):
    """Geometry *and* steady-profile results for one lateral structure.

    Inherits all geometry attributes from :class:`~rivia.hdf.LateralStructure`.

    All result properties return ``numpy`` arrays.  Scalar results (one value
    per profile) have shape ``(n_profiles,)``.  Segment-level results have
    shape ``(n_profiles, n_segments)`` or ``(n_segments,)`` for static arrays.

    .. note::
        Bridge, culvert, and inline structure nodes do not have separate result
        datasets in HEC-RAS 1D steady-flow HDF files.
        :class:`StructureCollection` returns plain geometry objects for
        those types.

    Parameters
    ----------
    geom:
        Geometry object from :class:`~rivia.hdf.StructureCollection`.
    group:
        ``h5py.Group`` at
        ``Results/Steady/.../Steady Profiles/Lateral Structures/<name>``.
    """

    def __init__(self, geom: LateralStructure, group: "h5py.Group") -> None:
        LateralStructure.__init__(
            self,
            mode=geom.mode,
            upstream_type=geom.upstream_type,
            downstream_type=geom.downstream_type,
            centerline=geom.centerline,
            location=geom.location,
            upstream_node=geom.upstream_node,
            downstream_node=geom.downstream_node,
            weir=geom.weir,
            gate_groups=geom.gate_groups,
        )
        self._g = group
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, key: str) -> np.ndarray:
        """Load ``{group}/{key}`` as a numpy array, cached."""
        if key not in self._cache:
            ds = self._g.get(key)
            if ds is None:
                raise KeyError(f"Dataset '{key}' not found in lateral group.")
            self._cache[key] = np.array(ds)
        return self._cache[key]

    def _load_seg(self, key: str) -> np.ndarray:
        """Load ``HW TW Segments/{key}`` as a numpy array, cached."""
        cache_key = f"_seg_{key}"
        if cache_key not in self._cache:
            ds = self._g.get(f"HW TW Segments/{key}")
            if ds is None:
                raise KeyError(
                    f"Dataset 'HW TW Segments/{key}' not found in lateral group."
                )
            self._cache[cache_key] = np.array(ds)
        return self._cache[cache_key]

    def _col_index(self, *candidates: str) -> int:
        """Column index of first ``Structure Variables`` column whose name
        contains any *candidates* (case-insensitive)."""
        names_lower = [n.lower() for n in self.variable_names]
        for cand in candidates:
            cand_l = cand.lower()
            for i, n in enumerate(names_lower):
                if cand_l in n:
                    return i
        raise KeyError(
            f"No column matching {candidates!r} in {self.variable_names!r}"
        )

    # ------------------------------------------------------------------
    # Structure Variables
    # ------------------------------------------------------------------

    @property
    def variable_names(self) -> list[str]:
        """Column names from the ``Structure Variables`` ``Variable_Unit`` attribute.

        Falls back to ``col_0``, ``col_1``, ... when the attribute is absent.
        """
        ds = self._g["Structure Variables"]
        attr = ds.attrs.get("Variable_Unit")
        if attr is not None:
            return [_decode(v[0]) for v in attr]
        return [f"col_{i}" for i in range(ds.shape[1])]

    @property
    def structure_variables(self) -> np.ndarray:
        """All structure variables.  Shape ``(n_profiles, n_vars)``.

        Column names are in :attr:`variable_names`.
        """
        return self._load("Structure Variables")

    @property
    def flow_total(self) -> np.ndarray:
        """Total flow through the structure.  Shape ``(n_profiles,)``."""
        return self._load("Structure Variables")[
            :, self._col_index("total flow", "flow")
        ]

    @property
    def flow_weir(self) -> np.ndarray:
        """Weir flow component.  Shape ``(n_profiles,)``."""
        return self._load("Structure Variables")[
            :, self._col_index("weir flow")
        ]

    @property
    def stage_hw(self) -> np.ndarray:
        """Headwater stage (weighted average along structure).  Shape ``(n_profiles,)``."""
        return self._load("Structure Variables")[
            :, self._col_index("stage hw", "hw", "headwater")
        ]

    @property
    def stage_tw(self) -> np.ndarray:
        """Tailwater stage.  Shape ``(n_profiles,)``."""
        return self._load("Structure Variables")[
            :, self._col_index("stage tw", "tw", "tailwater")
        ]

    @property
    def flow_hw_us(self) -> np.ndarray:
        """Flow at the upstream bounding cross section.  Shape ``(n_profiles,)``."""
        return self._load("Structure Variables")[
            :, self._col_index("flow hw us")
        ]

    @property
    def flow_hw_ds(self) -> np.ndarray:
        """Flow at the downstream bounding cross section.  Shape ``(n_profiles,)``."""
        return self._load("Structure Variables")[
            :, self._col_index("flow hw ds")
        ]

    @property
    def stage_hw_us(self) -> np.ndarray:
        """Stage at the upstream bounding cross section.  Shape ``(n_profiles,)``."""
        return self._load("Structure Variables")[
            :, self._col_index("stage hw us")
        ]

    @property
    def stage_hw_ds(self) -> np.ndarray:
        """Stage at the downstream bounding cross section.  Shape ``(n_profiles,)``."""
        return self._load("Structure Variables")[
            :, self._col_index("stage hw ds")
        ]

    # ------------------------------------------------------------------
    # Weir Variables (per HW-TW segment)
    # ------------------------------------------------------------------

    @property
    def weir_variables(self) -> np.ndarray | None:
        """Detailed weir hydraulics per HW-TW segment, or ``None`` if absent.

        Shape ``(n_profiles, n_segments)``.
        """
        ds = self._g.get("Weir Variables")
        if ds is None:
            return None
        if "_wv" not in self._cache:
            self._cache["_wv"] = np.array(ds)
        return self._cache["_wv"]

    # ------------------------------------------------------------------
    # HW TW Segments
    # ------------------------------------------------------------------

    @property
    def segment_stations(self) -> np.ndarray:
        """Lateral structure chainage station for each HW-TW segment.

        Shape ``(n_segments,)``.
        """
        return self._load_seg("HW TW Station")

    @property
    def segment_headwater_rs(self) -> np.ndarray:
        """Headwater reach station (RS) for each HW-TW segment.

        Shape ``(n_segments,)``, dtype byte-string — decode with
        ``seg.astype(str)`` if needed.
        """
        return self._load_seg("Headwater River Stations")

    @property
    def segment_flow(self) -> np.ndarray:
        """Flow through each HW-TW segment per profile.

        Shape ``(n_profiles, n_segments)``.
        """
        return self._load_seg("Flow")

    @property
    def segment_wse_hw(self) -> np.ndarray:
        """Headwater water-surface elevation at each segment per profile.

        Shape ``(n_profiles, n_segments)``.
        """
        return self._load_seg("Water Surface HW")

    @property
    def segment_wse_tw(self) -> np.ndarray:
        """Tailwater water-surface elevation at each segment per profile.

        Shape ``(n_profiles, n_segments)``.
        """
        return self._load_seg("Water Surface TW")


# ---------------------------------------------------------------------------
# StructureResultsCollection
# ---------------------------------------------------------------------------


class StructureResultsCollection(_GeomStructureCollection):
    """Steady-plan structure collection.

    Upgrades :class:`~rivia.hdf.LateralStructure` geometry objects to
    :class:`LateralResults` when a matching result group exists under
    ``Results/Steady/.../Steady Profiles/Lateral Structures``.

    All other structure types (Bridge, Culvert, Inline) are returned as plain
    geometry objects — HEC-RAS 1D steady-flow HDF files do not store separate
    result datasets for those node types.
    """

    def _load(self) -> dict[str, Structure]:  # type: ignore[override]
        if self._items is not None:
            return self._items

        import h5py as _h5

        geom_items = _GeomStructureCollection._load(self)

        lateral_root = self._hdf.get(_STEADY_LATERAL)
        lateral_groups: dict[str, "h5py.Group"] = (
            {k: v for k, v in lateral_root.items() if isinstance(v, _h5.Group)}
            if lateral_root is not None
            else {}
        )

        items: dict[str, Structure] = {}
        for key, geom in geom_items.items():
            if isinstance(geom, LateralStructure):
                plan_key = " ".join(geom.location)
                grp = lateral_groups.get(plan_key)
                items[key] = (
                    LateralResults(geom, grp) if grp is not None else geom
                )
            else:
                # Bridge, Culvert, Inline: no separate steady result datasets
                items[key] = geom

        self._items = items
        return self._items


# ---------------------------------------------------------------------------
# SteadyPlan - public entry point
# ---------------------------------------------------------------------------


class SteadyPlan(_PlanHdf, Geometry):
    """Read HEC-RAS steady-flow plan HDF5 output files (``*.p*.hdf``).

    A steady plan HDF file contains the same ``Geometry/`` data as a geometry
    HDF file, *plus* ``Results/Steady/...`` profile-based output.

    Results are indexed by steady-flow profile name (e.g. ``"Big"``,
    ``"Bigger"``, ``"Biggest"``).  Each result array has shape
    ``(n_profiles,)`` where each index corresponds to a profile.

    Parameters
    ----------
    filename:
        Path to the plan HDF file.  The ``.hdf`` suffix is appended
        automatically if absent.

    Examples
    --------
    ::

        with SteadyPlan("Baxter.p01") as hdf:
            profiles = hdf.profile_names          # ["Big", "Bigger", "Biggest"]
            xs = hdf.cross_sections["Baxter River Lower Reach 27470."]
            wse = xs.wse                           # shape (3,)
            vel = xs.velocity_channel              # shape (3,)

            sa = hdf.storage_areas["Northside"]
            sa_wse = sa.wse                        # shape (3,)
    """

    def __init__(self, filename: str | Path) -> None:
        super().__init__(filename)
        self._steady_cross_sections: CrossSectionResultsCollection | None = None
        self._steady_storage_areas: StorageAreaResultsCollection | None = None
        self._steady_structures: StructureResultsCollection | None = None

    # ------------------------------------------------------------------
    # Runtime log
    # ------------------------------------------------------------------

    def runtime_log(self) -> SteadyRuntimeLog:
        """Read the runtime compute log from ``Results/Summary/``.

        Returns
        -------
        SteadyRuntimeLog
            Log container with the full text/RTF compute messages and the
            compute-process table.  Steady-specific parsing methods will be
            added to :class:`SteadyRuntimeLog` as the library evolves.

        Raises
        ------
        KeyError
            If ``Results/Summary`` is absent from the HDF file.
        """
        return SteadyRuntimeLog(*self._runtime_log_raw())

    # ------------------------------------------------------------------
    # File metadata
    # ------------------------------------------------------------------

    @property
    def profile_names(self) -> list[str]:
        """Steady-flow profile names written by HEC-RAS.

        Returns
        -------
        list[str]
            Ordered list of profile names, e.g. ``["Big", "Bigger", "Biggest"]``.
            The index of each name corresponds to the first axis of all result
            arrays.

        Raises
        ------
        KeyError
            If ``Results/Steady`` is absent — e.g. the plan has not been run
            or this is not a steady-flow plan.
        """
        ds = self._hdf.get(_STEADY_PROFILE_NAMES)
        if ds is None:
            raise KeyError(
                f"'{_STEADY_PROFILE_NAMES}' not found. "
                "Ensure this is a steady-flow plan HDF file that has been run."
            )
        return [_decode(v) for v in np.array(ds)]

    @property
    def ras_version(self) -> str:
        """HEC-RAS version string from the plan HDF root attribute.

        Returns the ``File Version`` root attribute, e.g.
        ``'HEC-RAS 6.6 September 2024'``.
        """
        raw = self._hdf.attrs["File Version"]
        return raw.decode() if isinstance(raw, (bytes, bytes)) else str(raw)

    @property
    def projection(self) -> str | None:
        """WKT projection string stored in the plan HDF root, or ``None``.

        Returns ``None`` when the attribute is absent (older files or models
        without a defined projection).
        """
        raw = self._hdf.attrs.get("Projection")
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, (bytes, bytes)) else str(raw)

    # ------------------------------------------------------------------
    # Collections (override Geometry equivalents with results-aware types)
    # ------------------------------------------------------------------

    @property
    def cross_sections(self) -> CrossSectionResultsCollection:
        """1-D cross sections with geometry and steady-profile results."""
        if self._steady_cross_sections is None:
            self._steady_cross_sections = CrossSectionResultsCollection(
                self._hdf
            )
        return self._steady_cross_sections

    @property
    def storage_areas(self) -> StorageAreaResultsCollection:
        """Storage areas with geometry and steady-profile results."""
        if self._steady_storage_areas is None:
            self._steady_storage_areas = StorageAreaResultsCollection(
                self._hdf
            )
        return self._steady_storage_areas

    @property
    def structures(self) -> StructureResultsCollection:
        """All structures with geometry and, where available, steady-profile results.

        :class:`~rivia.hdf.LateralStructure` items are upgraded to
        :class:`LateralResults` when result data is present.
        :class:`~rivia.hdf.Bridge`, culvert, and inline structure items are
        returned as plain geometry objects — HEC-RAS 1D steady-flow HDF files
        do not store separate result datasets for those node types.
        """
        if self._steady_structures is None:
            self._steady_structures = StructureResultsCollection(self._hdf)
        return self._steady_structures
