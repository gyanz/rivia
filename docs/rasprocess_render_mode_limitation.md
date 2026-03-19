# RasProcess.exe StoreAllMaps — Render Mode Limitation

## Summary

`RasProcess.exe -Command=StoreAllMaps` (and `-Command=StoreMap`) **always produces rasters using
basic "sloping" (cell-corners only) interpolation**, regardless of the `<RenderMode>`,
`<UseDepthWeightedFaces>`, or `<ReduceShallowToHorizontal>` settings in the `.rasmap` file.
These settings are silently ignored by the scripting commands.

This means `raspy.model.Model.store_map()`, which delegates to `RasProcess.exe`, always
generates maps equivalent to `rasterize_rasmap(..., with_faces=False, use_depth_weights=False,
shallow_to_flat=True)`.

---

## How RenderMode Works in the Interactive GUI

The `.rasmap` file stores three top-level render-mode elements:

```xml
<RenderMode>slopingPretty</RenderMode>
<UseDepthWeightedFaces>true</UseDepthWeightedFaces>
<ReduceShallowToHorizontal>true</ReduceShallowToHorizontal>
```

Valid values for `<RenderMode>`:

| Value | Description |
|---|---|
| `horizontal` | Flat per-cell water surface. No interpolation across cell edges. |
| `sloping` | Sloping interpolation using cell-corner facepoints only. |
| `slopingPretty` | Sloping using cell corners **and** face centroids (`CellStencilMethod.WithFaces`). Supports depth-weighted faces and shallow reduction. |

`<UseDepthWeightedFaces>` and `<ReduceShallowToHorizontal>` are only meaningful when
`<RenderMode>` is `slopingPretty`.

These settings are read exclusively by the interactive RasMapper GUI in
`RASMapper.XMLLoad()` (source: `RasMapperLib/RasMapperLib/RASMapper.cs`, ~line 12876):

```csharp
switch (XML.StringTextElement(node, "RenderMode"))
{
    case "horizontal":
        SharedData.SetHorizontalRenderingMode();
        break;
    case "sloping":
    case "hybrid":
        SharedData.SetSlopingRenderingMode();
        break;
    default:  // "slopingPretty" or absent → always falls here
        bool useDepthWeightedFaces =
            XML.StringTextElement(node, "UseDepthWeightedFaces").RoughlyEquals("true");
        SharedData.SetSlopingPrettyRenderingMode(
            XML.StringTextElement(node, "ReduceShallowToHorizontal").RoughlyEquals("true"),
            useDepthWeightedFaces);
        break;
}
```

---

## What StoreAllMapsCommand Actually Does

Source: `RasMapperLib/RasMapperLib.Scripting/StoreAllMapsCommand.cs`

```csharp
public override void Execute(ProgressReporter prog = null)
{
    using (new SetSRSHelper(RasMapFilename))  // only sets coordinate reference system
    {
        SharedData.RasMapFilename = RasMapFilename;
        XElement xElement = XElement.Load(RasMapFilename);  // reads <Terrains> and <Results> only
        // ... iterates result map layers and calls rASResultsMap.StoreMap() ...
    }
}
```

`SetSRSHelper` only sets the spatial reference system from the rasmap. It does not read render
mode settings.

`StoreAllMapsCommand` never calls `RASMapper.XMLLoad()` — it reads the rasmap file only to find
terrain filenames and result layer definitions. The `<RenderMode>` element is never accessed.

`ExecuteWith()` (the entry point for command-line arguments) accepts only two parameters:
- `RasMapFilename` — path to the `.rasmap` file
- `ResultFilename` — path to the plan HDF file (optional filter)

There is no argument for render mode.

---

## SharedData Defaults (Used by StoreAllMaps)

When `RasProcess.exe` starts, `SharedData` static fields are initialized to
(source: `SharedData.cs`, module-level initialization, ~line 1926):

```csharp
CellRenderMode        = CellRenderMethod.Sloping;
CellStencilMode       = CellStencilMethod.JustFacepoints;  // NOT WithFaces
FaceWSMode            = FaceWSMethod.Adjusted;
FacepointAdjustmentMode = FacepointAdjustmentMethod.None;
ShallowBehaviorMode   = ShallowBehavior.ReduceToHorizontal;
UseDepthWeightedFaces = false;
```

`CellStencilMethod.JustFacepoints` means only cell-corner facepoints are used in the
interpolation stencil — face centroids are excluded. This is the "sloping (cell corners)"
mode in the RasMapper options dialog.

---

## Render Mode Internal State Machine

`SharedData` exposes three render-mode setters
(source: `SharedData.cs`, ~line 1765):

| Setter | CellRenderMode | CellStencilMode | UseDepthWeightedFaces | ShallowBehavior |
|---|---|---|---|---|
| `SetHorizontalRenderingMode()` | `Horizontal` | `JustFacepoints` | false | None |
| `SetSlopingRenderingMode()` | `Sloping` | `JustFacepoints` | false | None |
| `SetSlopingPrettyRenderingMode(reduceShallow, depthWeights)` | `Sloping` | **`WithFaces`** | per argument | per argument |

Only `slopingPretty` mode uses `CellStencilMethod.WithFaces` and can enable depth-weighted faces
or shallow-to-horizontal reduction.

`WaterSurfaceRenderer` branches on these values at render time:

```csharp
// in WaterSurfaceRenderer.cs
if (SharedData.CellRenderMode == SharedData.CellRenderMethod.Horizontal)
    // ... flat rendering
if (SharedData.CellRenderMode == SharedData.CellRenderMethod.Sloping)
    // ... sloping rendering (corners or corners+faces depending on CellStencilMode)
if (SharedData.CellStencilMode == SharedData.CellStencilMethod.WithFaces
    && SharedData.UseDepthWeightedFaces)
    // ... compute depth-weighted face weights
```

---

## Mapping to `rasterize_rasmap` Parameters

The Python `rasterize_rasmap()` function in `src/raspy/geo/_rasmap.py` implements the same
interpolation logic and directly mirrors the RasMapper modes:

| RasMapper option | `<RenderMode>` XML | `rasterize_rasmap` equivalent |
|---|---|---|
| Horizontal | `horizontal` | `with_faces=False` + override to flat (not yet implemented) |
| Sloping (Cell Corners) | `sloping` | `with_faces=False, shallow_to_flat=True, use_depth_weights=False` |
| Sloping (Cell Corners + Face Centers) | `slopingPretty` | `with_faces=True, shallow_to_flat=False, use_depth_weights=False` |
| Sloping + Shallow Reduces to Horizontal | `slopingPretty` + `<ReduceShallowToHorizontal>true</ReduceShallowToHorizontal>` | `with_faces=True, shallow_to_flat=True, use_depth_weights=False` |
| Sloping + Depth-Weighted Faces | `slopingPretty` + `<UseDepthWeightedFaces>true</UseDepthWeightedFaces>` | `with_faces=True, shallow_to_flat=False, use_depth_weights=True` |

`store_map()` output is always equivalent to **Sloping (Cell Corners)**:
`with_faces=False, shallow_to_flat=True, use_depth_weights=False`.

---

## GUI RasMapper Call Chain

The interactive RasMapper GUI uses the **same rendering engine** as `RasProcess.exe` — the
difference is solely in how `SharedData` render mode is initialized before rendering begins.

**GUI call chain (triggered from the "Store Map" menu):**

```
ManageResultsMaps.StoreMapPopup()          // RasMapperLib/ManageResultsMaps.cs
  → RASResultsMap.StoreMap(reporter)       // RasMapperLib/RASResultsMap.cs (~line 5530)
      → MapProcessingEngine.StoreMap(...)  // RasMapperLib/MapProcessingEngine.cs (~line 1320)
          → Renderer.GetSimplifiedComputer(...)   // RasMapperLib.Render/Renderer.cs (~line 1674)
              → WaterSurfaceRenderer.GetComputer(...)   // for WSE maps
              → DepthRenderer.GetComputer(...)          // for depth maps
              → VelocityRenderer.GetComputer(...)       // for velocity maps
```

The GUI sets `SharedData` render mode **before** reaching this chain:

```
RASMapper.XMLLoad()  // called at application startup when a rasmap is opened
  → reads <RenderMode>, <UseDepthWeightedFaces>, <ReduceShallowToHorizontal>
  → calls SharedData.SetHorizontalRenderingMode()
       or SharedData.SetSlopingRenderingMode()
       or SharedData.SetSlopingPrettyRenderingMode(reduceShallow, depthWeights)
```

**Why `RasProcess.exe` produces different output:**

Both paths call identical `MapProcessingEngine.StoreMap()` code. The only difference is
`SharedData` state at the time of the call:

| Caller | SharedData state when StoreMap() runs |
|---|---|
| GUI (`ManageResultsMaps`) | Set by `RASMapper.XMLLoad()` from rasmap XML |
| `RasProcess.exe` (`StoreAllMapsCommand`) | Module-level defaults (basic `Sloping`, `JustFacepoints`) |

There is no per-call render mode argument to `MapProcessingEngine.StoreMap()` — it reads
`SharedData` global state implicitly. Since `RasProcess.exe` never calls `RASMapper.XMLLoad()`,
`SharedData` is never updated from the rasmap, and the defaults apply regardless of what
the `.rasmap` file contains.

---

## Depth-Weighted Rendering Bug (UseDepthWeightedFaces=true)

### Symptom

Calling `store_map()` with `render_mode="hybrid"` and `use_depth_weights=True` produces:

```
Store-Map error from '...p17.hdf', Error loading facepoint elevations for precip rendering method.
```

No raster is generated. Return code is 0 (StoreAllMapsCommand considers it a non-fatal per-layer
error), so the Python caller sees a "0 maps generated" result.

### What Data Is Required

Depth-weighted rendering requires a per-facepoint terrain elevation array in a separate file:

```
<model_dir>/<PlanShortID>/PostProcessing.hdf
  └─ Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/
       2D Flow Areas/<mesh_name>/Processed Data/Profile (Horizontal)/FacePoint Elevation
       (float32 array, one value per facepoint)
```

`PostProcessing.hdf` is **separate from the plan HDF** (`*.p*.hdf`). Its path is computed by
`RASResults.OutputFolder`:

```csharp
// RASResults.cs
public string OutputFolder => Path.GetDirectoryName(SourceFilename)
                            + Path.DirectorySeparatorChar
                            + PlanAttributes.PlanShortID;
// e.g. D:\Models\Tulloch\p17\PostProcessing.hdf
```

If `PlanShortID` is empty, `OutputFolder` returns `""` and `PostProcessing.hdf` resolves to the
current working directory — a silent misplacement. `PlanShortID` is read from the plan HDF's
`Plan Data / Plan Information` group as attribute `"Plan ShortID"`.

`PostProcessing.hdf` location is **independent of `OverwriteOutputFilename`** — it is always
derived from the plan HDF location, never from the custom output folder.

### Root Cause: In-Memory Cache Not Populated

The exception is thrown at
[`WaterSurfaceRenderer.GetComputer()`](../archive/DLLs/RasMapperLib/RasMapperLib.Render/WaterSurfaceRenderer.cs#L1979),
line ~2055:

```csharp
float[][] perMeshFacepointElevations = _geometry.D2FlowArea.PerMeshFacepointElevations;
if (perMeshFacepointElevations == null || perMeshFacepointElevations.Length != _geometry.D2FlowArea.MeshCount)
    throw new Exception("Error loading facepoint elevations for precip rendering method.");
```

`PerMeshFacepointElevations` is a **per-instance field** on `RASD2FlowArea` — it is `null` by
default on any freshly constructed `RASResults` object. It is populated only by
[`RASD2FlowArea.EnsureGetPerMeshFacepointElevations()`](../archive/DLLs/RasMapperLib/RasMapperLib/RASD2FlowArea.cs#L3834),
which:

1. Calls `EnsureFacepointElevationsPopup()` → writes the array to `PostProcessing.hdf` if absent.
2. Reads the array from `PostProcessing.hdf` into `PerMeshFacepointElevations` in memory.

The only call site of `EnsureGetPerMeshFacepointElevations()` in the rendering pipeline is in
[`WaterSurfaceRenderer.ComputeWaterSurfaceInternal2D()`](../archive/DLLs/RasMapperLib/RasMapperLib.Render/WaterSurfaceRenderer.cs#L2644)
(line ~2644, the **horizontal** sub-path). However, `GetComputer()` runs **before**
`ComputeWaterSurfaceInternal2D()` — the sloping preparation code reads the field first (line
~2049–2060) without ensuring it is loaded. This is a latent RasMapperLib bug for headless use.

**Why the GUI works:** In the GUI, the user is shown a popup
(`EnsureFacepointElevationsPopup()`) before any rendering starts. The popup both writes to
`PostProcessing.hdf` and (via the GUI's `RASD2FlowArea` object) sets `PerMeshFacepointElevations`
on the object that is later used for rendering. In headless scripting, this popup is skipped and
the field is never populated on the rendering object.

**Why setting `SharedData.RasMapFilename` alone is insufficient:** `EnsureFacepointElevations()`
checks `_geometry.Terrain == null` and returns silently (via `reporter.ReportError`) rather than
throwing — so no exception propagates and the caller cannot detect the failure. Even with terrain
set, `StoreAllMapsCommand` creates a **new** `RASResults` whose `PerMeshFacepointElevations` is
always `null` regardless of any pre-computation on a different instance.

### Fix in `RasMapperStoreMap.exe`

For `UseDepthWeightedFaces=true`, `RasMapperStoreMap.exe` bypasses `StoreAllMapsCommand`
entirely and implements the loop itself:

```
1. SharedData.RasMapFilename = rasMapFilename                       (mirrors StoreAllMapsCommand)
2. Parse rasmap XML → terrain dict + result map layers
3. new RASResults(resultFilename)                                    (our instance)
4. Ensure Geometry.Terrain is set:
     a. Try auto-resolve via RASGeometry.Terrain getter
     b. If null: parse rasmap <Terrains>, create TerrainLayer(name, file, canEdit=false),
        set Geometry.Terrain directly via reflection
5. PostProcessor.EnsureFacepointElevations(null)
     → writes FacePoint Elevation array to PostProcessing.hdf
6. RASD2FlowArea.EnsureGetPerMeshFacepointElevations()             ← KEY FIX
     → reads the array from PostProcessing.hdf into PerMeshFacepointElevations in memory
     (same rasResults instance as step 3)
7. For each RASResultsMap layer in the rasmap XML matching this result file:
     new RASResultsMap(rasResults)                                   ← SAME rasResults
     XMLLoad(mapLayerElement)
     SetOverrideTerrainFilenamesDictionary(terrainDict)
     StoreMap(progressReporter, showFinishedMessage: false)
```

By reusing the **same `rasResults`** object in steps 3–7, `WaterSurfaceRenderer._geometry` is
the same `RASGeometry` where `D2FlowArea.PerMeshFacepointElevations` was already populated in
step 6. The check in `GetComputer()` at line ~2055 passes, and rendering succeeds.

### Terrain Resolution in Headless Mode

`RASGeometry.Terrain` resolution order (headless, `SharedData.RasMapper == null`):

```
1. _terrain != null                  → cached, return immediately
2. Paths.CanReadFile(_terrainFilename)→ new TerrainLayer(layerName, _terrainFilename)
   _terrainFilename comes from plan HDF: Geometry attribute "Terrain Filename" (relative→absolute)
3. SharedData.RasMapDoc != null      → RASMapperCom.GetTerrainFromXML(doc, _terrainLayername)
   _terrainLayername: plan HDF Geometry attribute "Terrain Layername"
   RasMapDoc set when SharedData.RasMapFilename is assigned (via RefreshXMLDoc)
4. (fallback) RASMapperCom.TerrainFilesAvailable() → first valid terrain file found
```

If all four paths return `null`, `RasMapperStoreMap.exe` falls back to parsing the rasmap XML
`<Terrains>` block directly and setting `Geometry.Terrain` via the property setter.

---

## Recommendation

**`store_map()` now supports all render modes including `slopingPretty` with depth-weighted
faces** via `RasMapperStoreMap.exe` (the raspy stub that replaces `RasProcess.exe` for stored
map generation).

`RasMapperStoreMap.exe` selects the execution path based on `UseDepthWeightedFaces`:

| Condition | Execution path |
|---|---|
| `UseDepthWeightedFaces=false` | `StoreAllMapsCommand.Execute()` (standard) |
| `UseDepthWeightedFaces=true` | Custom loop (reuses `rasResults`, populates in-memory cache first) |

When full render-mode fidelity is needed and a stored GeoTIFF/VRT tile set is not required,
`rasterize_rasmap()` from `raspy.geo` remains an alternative — it is faster for single-variable
exports and does not require a terrain HDF.

`store_map()` / `RasMapperStoreMap.exe` is preferred when:
- You need a persistent VRT + GeoTIFF tile set on disk (native RasMapper format)
- You need exact byte-for-byte parity with RasMapper's stored-map output files
- The model has many terrain tiles (GDAL-parallelised tile writing is faster than `rasterize_rasmap`)

---

## Source Files Referenced

All source files are decompiled VB.NET/C# from `RasMapperLib.dll` v2.0.0.0 (HEC-RAS 6.6),
located in `archive/DLLs/RasMapperLib/`.

| File | Topic |
|---|---|
| `RasMapperLib.Scripting/StoreAllMapsCommand.cs` | `StoreAllMaps` entry point — does not read render mode |
| `RasMapperLib.Scripting/StoreMapCommand.cs` | `StoreMap` entry point — does not read render mode |
| `RasMapperLib.Scripting/SetSRSHelper.cs` | Only sets coordinate reference system from rasmap |
| `RasMapperLib/RASMapper.cs` (~line 12876) | GUI `XMLLoad` — only place `<RenderMode>` is read |
| `RasMapperLib/SharedData.cs` (~line 1765) | `SetHorizontalRenderingMode`, `SetSlopingRenderingMode`, `SetSlopingPrettyRenderingMode` |
| `RasMapperLib/RASResults.cs` (~line 551) | `OutputFolder` property — computes `PostProcessing.hdf` directory |
| `RasMapperLib/RASGeometry.cs` (~line 2139) | `Terrain` property — headless resolution chain |
| `RasMapperLib/PostProcessor.cs` (~line 532) | `EnsureFacepointElevations()` — writes terrain elevations to `PostProcessing.hdf` |
| `RasMapperLib/RASD2FlowArea.cs` (~line 3834) | `EnsureGetPerMeshFacepointElevations()` — populates in-memory `PerMeshFacepointElevations` cache |
| `RasMapperLib/MapperOptionWSRenderMode.cs` | GUI dialog — `LoadCellRenderMode` / `StoreCellRenderMode` |
| `RasMapperLib.Render/WaterSurfaceRenderer.cs` (~line 2055, ~2644) | Reads `PerMeshFacepointElevations` in `GetComputer()`; calls `EnsureGetPerMeshFacepointElevations` only in `ComputeWaterSurfaceInternal2D` |
| `RasMapperLib.Render/SlopingFactors.cs` | Computes face-point water surface factors for sloping modes |
| `RasMapperLib/ManageResultsMaps.cs` | GUI entry point — `StoreMapPopup()` → `RASResultsMap.StoreMap()` |
| `RasMapperLib/RASResultsMap.cs` (~line 5530) | `StoreMap()` — shared by GUI and scripting layer |
| `RasMapperLib/MapProcessingEngine.cs` (~line 1320) | Core raster generation engine — called by both GUI and RasProcess |
| `RasMapperLib.Render/Renderer.cs` (~line 1674) | `GetSimplifiedComputer()` — dispatches to variable-specific renderer |
| `tools/RasMapperStoreMap/Program.cs` | raspy stub — implements depth-weighted workaround |
