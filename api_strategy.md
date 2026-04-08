# rivia API Strategy & Recommendations

---

## Completed Improvements (2026-04-07)

All 9 tasks from Section 9 were implemented. 535 tests pass, 6 skipped.
Three pre-existing failures remain (unrelated to these changes):
- `tests/model/test_plan.py::TestSimulationDate` — `PlanFile.simulation_date` not yet implemented
- `tests/model/test_plan.py::TestMutation::test_set_simulation_date` — same
- `tests/test_rivia.py::test_version` — version string mismatch

### A — Sort parameter rename (`flow_unsteady.py`)
Renamed `ascending: bool = False` to `descending: bool = True` on:
- `UnsteadyFlowEditor._sort_type`
- `UnsteadyFlowEditor.sort_flow_hydrographs`
- `UnsteadyFlowEditor.sort_gate_boundaries`
- `UnsteadyFlowEditor.sort_lateral_inflows`

`descending=True` is the default (highest station first = upstream → downstream in standard RAS
numbering). All tests updated from `ascending=True/False` to `descending=False/True`.

### B — Plural/singular fix (`flow_unsteady.py`)
`set_gate_opening` (singular) is the correct name — reverted to singular.
`set_all_gate_opening` → `set_all_gate_openings` rename was kept.

### C — `ManningEntry` validator (`geometry.py`)
Added `__post_init__` to `ManningEntry` raising `ValueError` if `n_value <= 0`.

Note: `IneffArea` validator (`x_start < x_end`) was considered but dropped — HEC-RAS files
legitimately store stations in either order, so no ordering constraint can be enforced.

### D — Cache invalidation docs (`model/__init__.py`)
Added "Cached after first access. Call `X.save()` then `reload()` to write changes back to
disk and refresh the cache." to the `Model.plan`, `Model.geom`, and `Model.flow` property
docstrings.

### E — None-vs-raise convention (`geometry.py`, `flow_steady.py`, `flow_unsteady.py`, `CLAUDE.md`)
Added a `Convention` section to the module docstring of all three model file modules:

> `get_*` methods return `None` when the requested item is not found.
> `set_*` methods raise `KeyError` when the target does not exist.

Also added one line to the `Conventions` section in `CLAUDE.md`.

### F — `ProjectFile` methods → properties (`project.py`, `model/__init__.py`, `docs/`)
Converted three zero-argument read-only methods to `@property`:
- `plans()` → `plans`
- `plan_titles()` → `plan_titles`
- `plan_short_ids()` → `plan_short_ids`

Updated all callers in `model/__init__.py` (`self.project.plans()` → `self.project.plans`)
and the usage example in `docs/guide/readdata.md`.

### G — `all_cross_sections()` (`geometry.py`)
Added `GeometryFile.all_cross_sections() -> list[CrossSection]` which iterates all reaches
and returns every cross section in file order. Inserted above the existing `cross_sections()`
method for discoverability.

### H — `Mapping` protocol for HDF collections (`hdf/_geometry.py`)
Made three collection classes inherit from `collections.abc.Mapping[str, T]`:
- `FlowAreaCollection(Mapping[str, FlowArea])`
- `StorageAreaCollection(Mapping[str, StorageArea])`
- `BoundaryConditionCollection(Mapping[str, BoundaryConditionLine])`

The abstract methods (`__getitem__`, `__iter__`, `__len__`) were already implemented;
inheriting from `Mapping` provides `keys()`, `values()`, `items()`, `get()`, and `__eq__`
for free. Also updated the `collections.abc` import to include `Mapping`.

### I — `NodeType(IntEnum)` and `_BcType(IntEnum)` (`geometry.py`, `flow_steady.py`, `model/__init__.py`)
**`geometry.py`:**
- Replaced 6 bare integer `NODE_*` constants with `class NodeType(IntEnum)` with members
  `CROSS_SECTION`, `CULVERT`, `BRIDGE`, `MULTIPLE_OPENING`, `INLINE_STRUCTURE`, `LATERAL_STRUCTURE`.
- Kept module-level aliases (`NODE_XS = NodeType.CROSS_SECTION` etc.) for backward compatibility.
- Updated `_NODE_TYPE_NAMES` to use `NodeType` keys.
- Exported `NodeType` from `model/__init__.py`.

**`flow_steady.py`:**
- Replaced 5 private `_BC_*` integer constants with `class _BcType(IntEnum)` with members
  `NONE`, `KNOWN_WS`, `CRITICAL_DEPTH`, `NORMAL_DEPTH`, `RATING_CURVE`.
- Kept `_BC_*` aliases pointing at the enum members (no external callers, but kept for
  internal readability continuity).

### Bonus — Pre-existing import error fixed (`hdf/__init__.py`)
Removed a broken import of `PlanStructureCollection` from `hdf/_unsteady_plan.py` that was
blocking all tests from collecting. The class exists only in development temp files, not in
the actual source file.

---

## Executive Summary

rivia's **internals are production-ready** — file I/O, HDF reading, COM control,
and pixel-perfect rasterization are all mature and well-tested. The weakness is
the **seams between subpackages**: users must understand too much internal
structure to accomplish common tasks. The `Model` class is the right
orchestration point, but it currently acts more as a *file handle container*
than a *hydraulic modelling toolkit*.

This document proposes changes in three tiers: **Critical** (API gaps that will
frustrate external users on day one), **Important** (ergonomic improvements
that separate a good library from a great one), and **Future** (nice-to-haves
that can wait for user feedback).

---

## 1. What Works Well (Keep As-Is)

These are strengths that should **not** be refactored:

| Aspect | Why it works |
|---|---|
| **Lazy caching on `Model` properties** | Users pay zero cost for files they don't touch. Clean invalidation via `reload()`. |
| **Verbatim-line editor pattern** | Round-trip fidelity for HEC-RAS files. Users can edit one field without corrupting the rest. |
| **Subpackage isolation** (`com/`, `geo/`, `hdf/`) | Clean dependency boundaries. Users who only need HDF reading never pull in `pywin32`. |
| **Standalone file classes** | `GeometryFile`, `PlanFile`, etc. work without COM or a `Model` instance. Power users and CI pipelines benefit. |
| **`hdf` dispatch by plan type** | Automatically choosing `SteadyPlanHdf` vs `UnsteadyPlanHdf` is the right default. |
| **`MapperExtension` convenience methods** | `export_wse()`, `open_depth()`, etc. are exactly the right level of abstraction. |
| **Flexible key types for collections** | `hdf.cross_sections["River Reach RS"]`, `[index]`, and `[(river, reach, rs)]` — all three are useful. |
| **Existing documentation** | `quickstart.md` and `readdata.md` are thorough and well-structured. |

---

## 2. Critical Issues

### 2.1 Top-level `rivia/__init__.py` exports nothing useful

**Problem:** `from rivia import Model` doesn't work. Users must write
`from rivia.model import Model`. For the single most important class in the
library, this is an unnecessary barrier.

**Recommendation:**

```python
# rivia/__init__.py
from rivia.model import Model
from rivia.hdf import GeometryHdf, SteadyPlanHdf, UnsteadyPlanHdf

__all__ = ["Model", "GeometryHdf", "SteadyPlanHdf", "UnsteadyPlanHdf"]
```

The top-level namespace should expose the 3-5 classes that 90% of users need.
Everything else stays importable from subpackages. This is how mature Python
libraries work (pandas exposes `DataFrame` at top level, not just in
`pandas.core.frame`).

### 2.2 `Model.hdf` silently falls back to `GeometryHdf` (uncached)

**Problem:** When the plan HDF doesn't exist (plan not yet run), `model.hdf`
silently returns a `GeometryHdf`. This fallback is:

- **Uncached** — every access re-opens the file
- **Silent** — users don't know they're getting geometry-only data
- **Type-unsafe** — `model.hdf.flow_areas["X"].max_water_surface` will fail at
  runtime with an opaque `AttributeError`

**Recommendation:**

Option A (preferred): Remove the silent fallback. If the plan HDF doesn't exist,
raise `FileNotFoundError` with a clear message: *"Plan HDF not found — run the
model first, or use `GeometryHdf(model.geom_hdf_file)` for geometry-only access."*

Option B: Add explicit properties:

```python
@property
def has_results(self) -> bool:
    """True if the plan HDF exists and contains results."""

@property
def geom_hdf(self) -> GeometryHdf:
    """Geometry HDF — always available, no results."""
```

### 2.3 No context manager protocol on `Model`

**Problem:** `Model` opens a COM server in `__init__` and relies on `__del__`
(unreliable) or `atexit` for cleanup. In scripts that open multiple models,
or in notebooks where kernels restart, this leaks HEC-RAS processes.

**Recommendation:**

```python
class Model:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Close the COM connection and release all file handles."""
        if self._hdf is not None:
            self._hdf.close()
            self._hdf = None
        self.controller.close()
```

Usage:
```python
with Model("project.prj") as model:
    model.run()
    wse = model.hdf.flow_areas["P1"].max_water_surface
# COM server closed, HDF handle released
```

### 2.4 No `model.plans` property for quick plan overview

**Problem:** Users must call `model.project.plans()` (a method, not a property)
to list plans, then manually match `plan["title"]` strings to switch. The
`change_plan()` method requires knowing the title, short_id, or index in advance.

**Recommendation:** Add a `plans` property that returns a clean summary:

```python
@property
def plans(self) -> list[PlanSummary]:
    """All plans in the project with their index, title, short_id, and active flag."""

# PlanSummary is a simple dataclass:
@dataclass
class PlanSummary:
    index: int
    title: str
    short_id: str
    path: Path
    active: bool  # True for the currently loaded plan
```

Then `model.plans` gives users immediate orientation:

```python
for p in model.plans:
    print(f"[{'*' if p.active else ' '}] {p.index}: {p.title} ({p.short_id})")
```

### 2.5 `flow_file` property returns `None` silently

**Problem:** `Model.flow_file` (line 232-241) returns `None` when no flow file
entry exists, rather than raising. This violates the principle of failing loudly.
Downstream code (`model.flow`) then re-checks and raises, but with a different
error message.

**Recommendation:** Make `flow_file` raise `ValueError` directly, consistent
with `plan_file` and `geom_file` which always return valid paths (via COM).

---

## 3. Important Ergonomic Improvements

### 3.1 Edit-save-reload cycle is too manual

**Problem:** The most common workflow is:

```python
model.plan.simulation_window = (start, end)
model.plan.save()
model.reload()
```

Three lines for one logical operation. Users will forget `save()` or `reload()`
and get confusing stale-state bugs.

**Recommendation:** Context manager for batch edits:

```python
with model.editing() as m:
    m.plan.simulation_window = (start, end)
    m.plan.computation_interval = "30SEC"
    m.flow.set_flow_hydrograph(0, new_values)
# On __exit__: saves all modified files, then reloads
```

This is a thin wrapper:

```python
@contextmanager
def editing(self):
    yield self
    self.reload(save_if_modified=True)
```

The beauty is that `reload()` already saves modified files. We just need to
make the pattern discoverable.

### 3.2 Run chaining should be a context manager

**Problem:** Current pattern is error-prone:

```python
model.enable_chaining(True)
for window in windows:
    model.plan.simulation_window = window
    model.plan.save()
    model.reload()
    model.run()
model.enable_chaining(False)
```

If any run fails, chaining is never disabled. The state machine in
`_ChainingState` is complex but necessary — the API on top of it should be
simpler.

**Recommendation:**

```python
with model.chaining():
    for window in windows:
        with model.editing():
            model.plan.simulation_window = window
        model.run()
# chaining auto-disabled, restart files cleaned up
```

Implementation:

```python
@contextmanager
def chaining(self, cleanup: bool = False):
    self.enable_chaining(True)
    try:
        yield self
    finally:
        self.enable_chaining(False)
        if cleanup:
            self.delete_restart_files()
```

### 3.3 Missing `__repr__` on key classes

**Problem:** In a notebook or REPL:

```python
>>> model
<rivia.model.Model object at 0x...>
>>> model.plan
<rivia.model.plan.PlanFile object at 0x...>
>>> model.hdf.flow_areas["P1"]
<rivia.hdf._unsteady_plan.FlowAreaResults object at 0x...>
```

None of these tell the user anything useful.

**Recommendation:**

```python
# Model
def __repr__(self):
    return f"Model({self.project_file.name!r}, plan={self.plan_file.name!r})"

# PlanFile
def __repr__(self):
    return f"PlanFile({self._path.name!r}, title={self.plan_title!r})"

# FlowAreaResults
def __repr__(self):
    return f"FlowAreaResults({self.name!r}, cells={self.n_cells}, faces={self.n_faces})"
```

This costs nothing and massively improves the interactive experience.

### 3.4 Ambiguous `CrossSection` name collision between `model` and `hdf`

**Problem:** Both `rivia.model` and `rivia.hdf` export a `CrossSection` class,
but they represent completely different things:

- `rivia.model.CrossSection` — parsed text-file geometry (stations, elevations,
  Manning's n)
- `rivia.hdf.CrossSection` — HDF mesh geometry (station-elevation array, river/
  reach/RS metadata)

Users who do `from rivia.model import *` and `from rivia.hdf import *` get a
silent collision.

**Recommendation:** Rename one or both for disambiguation:

- `rivia.model.CrossSection` stays as-is (it's the "editing" dataclass)
- `rivia.hdf.CrossSection` becomes `HdfCrossSection` or
  `CrossSectionGeometry`

Or keep the names but ensure the top-level `rivia` namespace only exports the
`model` variant, since that's the one users edit. The HDF one is accessed via
`hdf.cross_sections[key]` and rarely imported directly.

### 3.5 Consistent naming: `UnsteadyFlowEditor` vs `SteadyFlowFile`

**Problem:** The steady flow class is `SteadyFlowFile` but the unsteady one is
`UnsteadyFlowEditor`. The inconsistency is jarring. Additionally,
`UnsteadyFlowFile` is also exported (appears to be a base/legacy name).

**Recommendation:** Settle on one convention. Since both read *and* write,
`SteadyFlowFile` / `UnsteadyFlowFile` is more natural. `Editor` suffix implies
a different interaction model. If `UnsteadyFlowEditor` has a meaningfully
different API shape that justifies the name, document why; otherwise, alias or
rename.

---

## 4. API Surface Refinements

### 4.1 Add convenience queries on `Model`

These are the questions users ask most often. Each is a one-liner that currently
requires navigating 2-3 objects:

```python
# Current
areas = model.hdf.flow_areas
area = areas["Perimeter 1"]
wse_max = area.max_water_surface

# Proposed convenience
model.hdf.flow_area_names       # list[str] — already on FlowAreaCollection
model.hdf.structure_names       # list[str]
```

Don't add convenience methods on `Model` that simply proxy `model.hdf.X` — that
creates two ways to do the same thing. Instead, ensure the *collections* have
good discoverability (`.names`, `len()`, `__contains__`, iteration).

### 4.2 `model.summary()` — one-call project overview

For notebooks and debugging, a single method that prints the project state:

```python
def summary(self) -> str:
    """Human-readable summary of the current project state."""
    # Project: Baxter.prj
    # Plan: Base Condition (p01) [active]
    # Geometry: g01 — 3 reaches, 45 cross sections, 2 bridges
    # Flow: u01 (unsteady) — 5 hydrographs, 2 lateral inflows
    # HDF: p01.hdf — 2 flow areas, 1 storage area
    # Last run: 2026-04-05T09:30:00 — success
```

### 4.3 Type annotations on `Model.hdf`

**Problem:** `hdf` property has no return type annotation, so static analysis
and IDE autocompletion don't work.

**Recommendation:** Use `Union` with `overload` or a protocol, or at minimum:

```python
@property
def hdf(self) -> SteadyPlanHdf | UnsteadyPlanHdf:
```

The `GeometryHdf` fallback complicates this — another reason to remove it
(see 2.2).

### 4.4 `HdfFile.close()` and resource lifecycle

**Problem:** HDF file handles are managed via `_HdfFile.__del__`, which is
unreliable. The `Model.reload()` method explicitly closes HDF handles, but
standalone `GeometryHdf`/`PlanHdf` usage doesn't have a clean `close()` or
context manager.

**Recommendation:** Add `__enter__`/`__exit__` to `_HdfFile`:

```python
with UnsteadyPlanHdf("model.p01.hdf") as hdf:
    wse = hdf.flow_areas["P1"].max_water_surface
# HDF file handle closed
```

---

## 5. Documentation & Discoverability

### 5.1 Add `_repr_html_` for notebook rendering

`FlowAreaResults`, `CrossSectionResults`, and collection classes should have
`_repr_html_` methods for rich Jupyter rendering. Show a summary table with
cell/face counts, available variables, time range, etc.

### 5.2 Docstring consistency

Some properties have docstrings, some don't. All public properties should have
at least a one-line docstring. The `Model` class properties are mostly good,
but `hdf` and `flow` could explain *when* to use them vs alternatives.

### 5.3 Error messages should suggest next steps

Bad:
```
FileNotFoundError: Z:\project\model.p01.hdf
```

Good:
```
FileNotFoundError: Plan HDF 'model.p01.hdf' does not exist.
Run the model first with model.run(), or use GeometryHdf(model.geom_hdf_file)
for geometry-only access.
```

The library already does this in some places (e.g., chaining errors are
excellent). Apply the pattern consistently.

---

## 6. Future Considerations (Post-Launch)

### 6.1 Async/background run support

`model.run(blocking=False)` returns immediately but provides no way to await
completion except polling `controller.Compute_Complete()`. A future version
could return a `RunHandle` with `.wait()`, `.cancel()`, and callback support.

### 6.2 Multi-plan batch runner

```python
results = model.run_plans(["Base", "Alternative 1", "Alternative 2"])
# Returns dict[str, (bool, tuple[str, ...])]
```

This would handle plan switching, running, and collecting results.

### 6.3 DataFrame export helpers

```python
model.hdf.flow_areas["P1"].to_dataframe(variables=["wse", "depth"], timesteps=[0, 5, 10])
```

Returns a tidy DataFrame with columns `[cell_index, timestep, wse, depth]` — 
ready for plotting or analysis.

### 6.4 Comparison utilities

```python
diff = rivia.compare(model_a.hdf, model_b.hdf)
diff.max_wse_difference       # per-cell max WSE difference
diff.summary()                # human-readable comparison
```

### 6.5 Plugin/extension API

Allow users to register custom post-processing steps that run after `model.run()`.
This is low priority but would be useful for teams with standard QA checks.

---

## 7. Implementation Priority

### Phase 1 — Ship-blockers (do before any public release)

1. Top-level exports in `rivia/__init__.py` (2.1)
2. Context manager on `Model` (2.3)
3. `__repr__` on all major classes (3.3)
4. Type annotation on `Model.hdf` (4.3)
5. Fix `flow_file` returning `None` (2.5)

### Phase 2 — First-week ergonomics

6. `model.editing()` context manager (3.1)
7. `model.chaining()` context manager (3.2)
8. Remove silent `GeometryHdf` fallback on `model.hdf` (2.2)
9. `model.plans` property (2.4)
10. `__enter__`/`__exit__` on `_HdfFile` (4.4)

### Phase 3 — Polish

11. `model.summary()` (4.2)
12. `_repr_html_` for Jupyter (5.1)
13. Consistent naming audit (3.5, 3.4)
14. Error message improvement pass (5.3)

### Phase 4 — Post-launch

15. Async run support (6.1)
16. Multi-plan batch runner (6.2)
17. DataFrame export helpers (6.3)

---

## 8. Additional API Refinements (Code Audit)

These items were identified from a detailed code audit and are not yet covered above.

### 8.1 Plural/singular mismatches

- `flow_unsteady.py:1058` — getter `get_gate_openings()` vs setter `set_gate_opening()` (missing `s`)
- `flow_unsteady.py:1517` — `set_all_gate_opening()` vs `set_all_lateral_inflows()` (inconsistent plural)

### 8.2 Method vs property inconsistency on `ProjectFile`

`FlowAreaCollection.names` and `.summary` are properties, but `ProjectFile.plans()`,
`plan_titles()`, `geom_titles()` etc. are methods. Standardize as properties.
File: `project.py:245-285`.

### 8.3 No consistent None-vs-raise convention

Getters silently return `None` on miss; setters raise `KeyError`. Fine as a rule, but it is
nowhere documented and a few places break it. Document the convention explicitly in each
module docstring and in CLAUDE.md.

Files: `geometry.py:1635`, `flow_steady.py:385,430`, `flow_unsteady.py:916`.

### 8.4 Missing dataclass validators

- `geometry.py:235` — `ManningEntry`: add `__post_init__` asserting `n_value > 0`
- `geometry.py:251` — `IneffArea`: no validator added — HEC-RAS files store stations in either
  order (`x_start` may be greater than `x_end`), so no ordering constraint can be enforced.

### 8.5 Sort parameter naming — make default explicit

`ascending: bool = False` is non-obvious because the default behaviour (descending) is
the opposite of the parameter name. Renamed to `descending: bool = True` so the default
is self-documenting and the name matches the domain meaning (river stations decrease
along the list by default).

- `flow_unsteady.py:1441` — `sort_flow_hydrographs`
- `flow_unsteady.py:1445` — `sort_gate_boundaries`
- `flow_unsteady.py:1457` — `sort_lateral_inflows`

### 8.6 Inconsistent collection protocol

HDF geometry collections (`FlowAreaCollection`, `StorageAreaCollection`, etc.) implement
`__getitem__`, `__iter__`, `__len__` but not `keys()`/`values()`/`items()`, unlike
`StructureIndex`. Either inherit from `collections.abc.Mapping` uniformly, or drop the
dict-like methods everywhere. Files: `hdf/_geometry.py:1385, 1565, 1688`.

### 8.7 Node/BC type constants → `IntEnum`

```python
# Before
NODE_XS = 1
NODE_CULVERT = 2

# After
class NodeType(IntEnum):
    CROSS_SECTION = 1
    CULVERT = 2
    ...
```

Files: `geometry.py:76-87`, `flow_steady.py:127-132`.

### 8.8 Cache invalidation undocumented

`Model.plan`, `.geom`, `.flow` are cached properties but callers have no obvious signal
that they must call `reload()` after saving edits. Add a one-liner to each property
docstring: *"Cached; call `reload()` to refresh after saving."*
File: `model/__init__.py:228-278`.

### 8.9 Missing batch / summary helper on `GeometryFile`

No single call returns all cross-sections across all reaches. Add:

```python
def all_cross_sections(self) -> list[CrossSection]:
    """Return every cross-section in the geometry, in file order."""
```

File: `model/geometry.py`.

---

## 9. Implementation Plan (Code Audit Items)

Ordered by effort and risk. Each task is scoped to ≤ 3 files.

### Task A — Rename sort parameters (`flow_unsteady.py`) ✦ trivial ✅

**Items:** 8.5  
**Files:** `src/rivia/model/flow_unsteady.py`

Renamed the `ascending` parameter to `descending` on three methods. The default
behaviour is "descending by station" (highest first), which maps to `descending=True`.

```python
# Before
def sort_flow_hydrographs(self, ascending: bool = False) -> None:
# After
def sort_flow_hydrographs(self, *, descending: bool = True) -> None:
```

---

### Task B — Fix plural/singular naming (`flow_unsteady.py`) ✦ trivial ✅

**Items:** 8.1  
**Files:** `src/rivia/model/flow_unsteady.py`, `src/rivia/model/__init__.py`

- `set_gate_opening()` — singular is correct; no rename needed
- `set_all_gate_opening()` → `set_all_gate_openings()` — renamed

---

### Task C — Dataclass validators (`geometry.py`) ✦ small

**Items:** 8.4  
**Files:** `src/rivia/model/geometry.py`

Add `__post_init__` to two dataclasses:

```python
@dataclass
class ManningEntry:
    station: float
    n_value: float
    variation: float = 0.0

    def __post_init__(self) -> None:
        if self.n_value <= 0:
            raise ValueError(f"n_value must be > 0, got {self.n_value}")

@dataclass
class IneffArea:
    x_start: float
    x_end: float
    elevation: float
    permanent: bool = False

    def __post_init__(self) -> None:
        if self.x_start >= self.x_end:
            raise ValueError(
                f"x_start ({self.x_start}) must be less than x_end ({self.x_end})"
            )
```

Add test cases asserting the `ValueError` is raised on bad input.

---

### Task D — Document cache invalidation (`model/__init__.py`) ✦ small

**Items:** 8.8  
**Files:** `src/rivia/model/__init__.py`

Add "Cached; call `reload()` to refresh after saving." to the docstrings of `.plan`,
`.geom`, and `.flow` properties. No behaviour change.

---

### Task E — Document None-vs-raise convention ✦ small

**Items:** 8.3  
**Files:** `src/rivia/model/geometry.py`, `src/rivia/model/flow_steady.py`,
           `src/rivia/model/flow_unsteady.py`, `CLAUDE.md`

Add a module-level docstring section to each file (or a comment block at the top of the
public methods section) stating:

> **Convention:** `get_*` methods return `None` when the requested item is not found.
> `set_*` / `del_*` methods raise `KeyError` when the target does not exist.

Also add one line to the **Conventions** section of `CLAUDE.md`.

---

### Task F — `ProjectFile` methods → properties (`project.py`) ✦ medium

**Items:** 8.2  
**Files:** `src/rivia/model/project.py`, `src/rivia/model/__init__.py`,
           any tests that call these as methods

Convert `plans()`, `plan_titles()`, `geom_titles()`, and any similar zero-argument
read-only accessors to `@property`. The call-site change is `project.plans()` →
`project.plans`. Grep for all usages first and list the files to change before editing.

---

### Task G — `all_cross_sections()` on `GeometryFile` (`geometry.py`) ✦ medium

**Items:** 8.9  
**Files:** `src/rivia/model/geometry.py`, `src/rivia/model/__init__.py`

Add a method that iterates all reaches and collects every `CrossSection` in file order:

```python
def all_cross_sections(self) -> list[CrossSection]:
    """Return every cross-section in the geometry, in file order."""
    result = []
    for river, reach in self.reaches:
        result.extend(self.cross_sections(river, reach))
    return result
```

Export from `model/__init__.py` if `CrossSection` is already exported. Add a test.

---

### Task H — Uniform collection protocol (`hdf/_geometry.py`) ✦ medium

**Items:** 8.6  
**Files:** `src/rivia/hdf/_geometry.py`

Make `FlowAreaCollection`, `StorageAreaCollection`, and `BoundaryConditionCollection`
all inherit from `collections.abc.Mapping[str, T]`. This requires adding `keys()`,
`values()`, `items()` (all free once `__getitem__` + `__iter__` + `__len__` are
present via `Mapping` mixin). Verify existing tests still pass.

---

### Task I — `NodeType` and `BoundaryType` enums ✦ medium

**Items:** 8.7  
**Files:** `src/rivia/model/geometry.py`, `src/rivia/model/flow_steady.py`,
           `src/rivia/model/__init__.py`

1. In `geometry.py`, replace the `NODE_*` integer constants with:
   ```python
   class NodeType(IntEnum):
       CROSS_SECTION = 1
       CULVERT = 2
       BRIDGE = 3
       MULTIPLE_OPENING = 4
       INLINE_STRUCTURE = 5
       LATERAL_STRUCTURE = 6
   ```
   Keep the old names as module-level aliases (`NODE_XS = NodeType.CROSS_SECTION`) for
   one release cycle if any external code might reference them, then remove.

2. In `flow_steady.py`, do the same for `_BC_*` constants → `BoundaryConditionType(IntEnum)`.
   These are currently private so no backward-compat aliases needed.

3. Export `NodeType` from `model/__init__.py`.

---

## 10. Guiding Principles

1. **One obvious way to do common things.** If 80% of users need max WSE, make
   it easy. Don't force them through `model.hdf.flow_areas["name"].max_water_surface`.
   But also don't add redundant paths — make the primary path obvious and short.

2. **Fail loudly, fail early.** Never silently return the wrong type or stale data.
   A clear `FileNotFoundError` with a suggestion is infinitely better than a
   mysterious `AttributeError` three calls later.

3. **Context managers for stateful operations.** Anything that opens resources
   (COM, HDF), modifies state (editing files), or needs cleanup (chaining) should
   have a `with` form.

4. **Interactive-first design.** `__repr__`, tab-completion, notebook rendering.
   Most HEC-RAS users are engineers, not software developers. Make the library
   feel like an interactive tool, not a code framework.

5. **Don't hide the power.** Keep the standalone file classes, keep the raw HDF
   access, keep the low-level COM controller. The convenience layer should be
   *on top of* the power layer, not *instead of* it. Users should be able to
   drop down when they need to.
