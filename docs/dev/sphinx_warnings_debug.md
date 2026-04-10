# Sphinx Warnings Debug Log

## Status (as of 2026-04-09)

Most Sphinx warnings have been resolved. **250 `py:attribute` duplicate-object warnings** for
`rivia.hdf.*` dataclass attributes remain open.

---

## Resolved warnings

| Warning | Root cause | Fix applied |
|---------|-----------|-------------|
| `autosummary.import_cycle` in `controller.rst`, `geo.rst`, `utils.rst` | Full dotted paths in autosummary entries | Changed to relative names (`controller`, `ras`, etc.) |
| RST formatting errors in `geo/raster.py` | Enumerated-list items (`A.`, `4a.`, `3.5.`) and hanging-indent continuation lines in Parameters | Converted to inline-bold paragraphs; merged continuation lines |
| Undefined substitution refs (`|diff|`, `|Q|`, `|V_n|`) in `unsteady_plan.py`, `velocity.py` | Bare pipe syntax in docstrings | Used `` ``diff`` `` and `\|Q\|/\|V_n\|` |
| Hanging indent in `export_raster` `variable:` parameter | Continuation lines under a definition-list term | Merged to single lines per value |
| Duplicate object descriptions for `rivia.model.*` classes | autosummary class stubs double-registered classes | Added `:no-index:` to `autoclass` in `docs/_templates/autosummary/class.rst` |
| Adjacent `---` transitions in `introduction.md` | Two consecutive `---` lines with nothing between | Removed the redundant `---` at line 24 |
| `HecRasComputeError` cross-reference ambiguity | Same class in both `rivia.controller` and `rivia.controller.controller` | Changed Raises entry to `~rivia.controller.HecRasComputeError` |

---

## Remaining: 250 `rivia.hdf.*` duplicate `py:attribute` warnings

### Pattern

```
<unknown>:1: WARNING: duplicate object description of
    rivia.hdf.geometry.BoundaryConditionLine.bc_type,
    other instance in api/generated/rivia.hdf.geometry,
    use :no-index: for one of them
```

All 250 warnings follow this pattern (all are `py:attribute` entries; all first-instances at `<unknown>:1`; all second-instances in one of the five HDF module stub pages).

### Root cause (confirmed)

The duplicates arise **within a single `.. automodule::` call** for an HDF submodule stub.  When
`automodule:: rivia.hdf.geometry` runs:

1. **Napoleon inline path**: autodoc processes the class docstring of (e.g.) `BoundaryConditionLine`,
   which has a NumPy-style `Attributes` section.  Napoleon converts that section to
   `.. attribute:: bc_type` directives.  These directives are nested-parsed **inline** (no source
   file → location reported as `<unknown>:1`) and register `py:attribute` entries in the Sphinx
   domain.

2. **Autodoc member path**: Sphinx 9.x has special `DataclassAttributeDocumenter` handling that
   documents typed dataclass fields even when `undoc-members = False`.  Each field is documented
   as a second `py:attribute` entry, this time located in the stub RST file
   (`api/generated/rivia.hdf.geometry`).

The two paths document the same attribute twice → duplicate warning.

### Key diagnostic facts

- Model modules have **no** `Attributes ----------` sections in any class docstring → no duplicate
  from the Napoleon inline path → model duplicates fixed by earlier work.
- HDF modules (`geometry.py`, `unsteady_plan.py`) have `Attributes ----------` sections in 16
  dataclass docstrings → 250 duplicate warnings.
- Switching `hdf.rst` from `autosummary` to `toctree` (already done) did **not** help — the
  duplicates originate inside a single `automodule` call, not across two different RST directives.
- `suppress_warnings = ["py.duplicated_object"]` does **not** work in Sphinx 9.x — the warning is
  issued without a type/subtype and cannot be suppressed by that mechanism.

### Fix options

**Option A — Move attribute descriptions to field docstrings (recommended)**

Remove the `Attributes ----------` section from each HDF dataclass class docstring and place the
description as a string literal immediately after the field definition:

```python
@dataclass
class BoundaryConditionLine:
    """Short class description (no Attributes section)."""

    name: str
    """Name of the boundary condition line."""

    bc_type: str
    """Boundary condition type string from HEC-RAS."""
```

Napoleon no longer generates inline `.. attribute::` directives (no Attributes section to process),
so only one registration occurs (from `DataclassAttributeDocumenter`).

Scope: 16 dataclasses across `hdf/geometry.py` (11) and `hdf/unsteady_plan.py` (5).

**Option B — `napoleon_use_ivar = True` in `conf.py`**

Adds `napoleon_use_ivar = True` to `docs/conf.py`.  Napoleon then emits `:ivar field: desc`
(a field-list entry) instead of `.. attribute::`, which does **not** register a `py:attribute` in
the Sphinx domain.  The description still appears in the rendered class page.

Trade-off: `:attr:`BoundaryConditionLine.bc_type`` cross-references would stop working for
attributes documented only via the Attributes section (because no `py:attribute` domain entry is
created from Napoleon).  Cross-references to attributes that also have field docstrings would still
work.

**Option C — `autodoc_default_options["no-special-members"]` / override per stub**

Add `:no-special-members:` or use per-stub `:no-members:` overrides to prevent the
`DataclassAttributeDocumenter` path.  More invasive and hides useful member documentation.

### Recommended next step

Implement **Option A**: refactor the 16 HDF dataclasses to use field docstrings instead of the
class-level `Attributes` section, then run a full `sphinx-build -E` to confirm zero warnings.
If only partial attribute descriptions are needed, **Option B** (`napoleon_use_ivar = True`) is a
one-line conf change with the trade-off noted above.

---

## Files changed during this debug session

| File | Change |
|------|--------|
| `docs/api/controller.rst` | Autosummary entries → relative names |
| `docs/api/geo.rst` | Autosummary entries → relative names |
| `docs/api/utils.rst` | Autosummary entries → relative names |
| `docs/api/hdf.rst` | `autosummary` → `toctree` (cosmetic; did not fix duplicates) |
| `docs/_templates/autosummary/class.rst` | Added `:no-index:` to `autoclass` |
| `docs/_templates/autosummary/module.rst` | Removed inner autosummary blocks for classes/functions |
| `docs/api/generated/*.rst` (all 16) | Stripped to title + `automodule` only |
| `docs/conf.py` | `autosummary_generate_overwrite = False`, `suppress_warnings = [...]` |
| `docs/guide/introduction.md` | Removed duplicate `---` transition |
| `src/rivia/geo/raster.py` | Fixed `rasterize_results` docstring RST formatting |
| `src/rivia/hdf/unsteady_plan.py` | Fixed `|diff|` and `|Q|/|V_n|` in docstrings |
| `src/rivia/hdf/velocity.py` | Fixed `|Q|/|V_n|` in docstring |
| `src/rivia/model/__init__.py` | Qualified `HecRasComputeError` in Raises section |
