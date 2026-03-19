# StoreMap Execution: Reference Graph and XML Setup

> Derived by reverse-engineering `RasMapperLib.dll` (HEC-RAS 6.6) with ILSpy.
> Source files: `archive/DLLs/RasMapperLib/`

---

## 1. Entry Point

`RasProcess.exe` is invoked with a command-file argument:

```
RasProcess.exe -CommandFile=tmp.xml
```

The XML file contains a single `<Command>` element. `RasProcess` calls
[`Command.ParseExecute(args)`](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/Command.cs#L109),
which loads the file and dispatches to the correct command class.

---

## 2. XML Parsing — What Is and Is Not Read

[`StoreMapCommand.XMLLoadTags(el)`](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreMapCommand.cs#L150)
is the **only** place the command XML is consumed.

### Elements read

| XML Element | Property set | Code |
|---|---|---|
| `<MapType>` | `MapType` | [StoreMapCommand.cs:152](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreMapCommand.cs#L152) |
| `<Result>` | `Result` | [StoreMapCommand.cs:157](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreMapCommand.cs#L157) |
| `<Terrain>` | `Terrain` (optional) | [StoreMapCommand.cs:162](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreMapCommand.cs#L162) |
| `<OutputBaseFilename>` | `OutputBaseFilename` | [StoreMapCommand.cs:167](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreMapCommand.cs#L167) |
| `<ProfileName>` | `ProfileName` | [StoreMapCommand.cs:172](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreMapCommand.cs#L172) |
| `<ArrivalStartProfile>` | `ArrivalStartProfile` | [StoreMapCommand.cs:177](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreMapCommand.cs#L177) |
| `<ArrivalEndProfile>` | `ArrivalEndProfile` | [StoreMapCommand.cs:182](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreMapCommand.cs#L182) |
| `<TimeUnits>` | `TimeUnits` | [StoreMapCommand.cs:187](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreMapCommand.cs#L187) |

### Elements silently ignored

| XML Element | Why it looks relevant but isn't |
|---|---|
| `<RasMapFilename>` | Not in `XMLLoadTags`; only used by `StoreAllMapsCommand` |
| `<ProfileIndex>` | Not read; profile resolved by name only via `TrySetProfile` |
| `<OutputMode>` | Not read; always hard-coded to `StoredDefaultTerrain` |
| `<RASProjectionFilename>` | Not read; SRS resolved from terrain raster instead |

### Minimal correct XML

```xml
<Command Type="StoreMap">
  <MapType>elevation</MapType>
  <Result>D:\path\to\model.p03.hdf</Result>
  <ProfileName>03Jan1999 00:00:00</ProfileName>
  <OutputBaseFilename>D:\custom\output\stem</OutputBaseFilename>
</Command>
```

`<MapType>` string values: `elevation`, `depth`, `velocity`, `depthvelocity`,
`depthvelocitysquared`, `froudenumber`, `arrivaltime`, `duration`.

---

## 3. Execution Graph

```
RasProcess.exe -CommandFile=tmp.xml
│
├─ Command.ParseExecute(args)                 [Command.cs:109]
│   └─ StoreMapCommand.XMLLoadTags(el)        [StoreMapCommand.cs:150]
│       (reads elements listed in §2 above)
│
└─ StoreMapCommand.Execute()                  [StoreMapCommand.cs:40]
    │
    ├─ [1] RASResults.TryIdentifyResultsFile(Result)
    │       [RASResults.cs:1704]
    │       ├─ if file path readable → new RASResults(Result)  ← opens plan HDF
    │       └─ else → returns null → exception thrown
    │
    ├─ [2] Resolve TerrainLayer               [StoreMapCommand.cs:47]
    │       ├─ <Terrain> empty → rASResults.Geometry.Terrain   (embedded in HDF)
    │       └─ <Terrain> set  → new TerrainLayer(Terrain)
    │                           rASResults.Geometry.SetTerrainTemporary()
    │
    ├─ [3] terrainLayer.AllSourceFilesExist() [StoreMapCommand.cs:57]
    │       └─ throws if any terrain raster file is missing
    │
    ├─ [4] new RASResultsMap(rASResults, MapType)
    │       └─ OutputMode = StoredDefaultTerrain (hard-coded)
    │
    ├─ [5] rASResultsMap.TrySetProfile(ProfileName)
    │       [RASResultsMap.cs:8042]
    │       └─ matches ProfileName string → sets ProfileIndex integer
    │
    ├─ [6] rASResultsMap.OverwriteOutputFilename = OutputBaseFilename
    │       [RASResultsMap.cs:3322]
    │       └─ setter strips extension; stores dir+stem
    │       ⚠ See Bug §5.2 — path is still joined onto Results.OutputFolder
    │
    ├─ [7] SetSRSHelper(terrainLayer)         [SetSRSHelper.cs:16]
    │   ├─ saves SharedData.SRSFilename as _initialSRSFilename
    │   ├─ if SharedData.IsSRSFilenameOK() → skip (reuse existing)
    │   └─ else:
    │       ├─ GDALRaster(terrain.RasterFilename[0]).GetProjection()
    │       ├─ if projection valid:
    │       │   ├─ _tempSRSFilename = Path.GetTempFileName()  ← ".tmp" ⚠ Bug §5.1
    │       │   ├─ projection.ExportEsri(_tempSRSFilename)
    │       │   └─ SharedData.SRSFilename = _tempSRSFilename
    │       └─ else: SRSFilename stays empty/prior value
    │
    └─ [8] rASResultsMap.StoreMap(prog)       [RASResultsMap.cs:5530]
        │
        ├─ guard: _outputMode.IsDynamic → return false
        ├─ EnsureDataExists()
        ├─ StoredMapBaseFilename resolved      [RASResultsMap.cs:3326]
        │   ├─ _overwriteOutputFilename empty → Path.Combine(OutputFolder, Description)
        │   └─ set → Path.Combine(OutputFolder, _overwriteOutputFilename)  ⚠ Bug §5.2
        │
        └─ MapProcessingEngine.StoreMap(map, terrain, baseFilename, reporter)
                                              [MapProcessingEngine.cs:1320]
            │
            ├─ [A] new MapProcessingEngine(map, terrain, baseFilename, reporter)
            │
            ├─ [B] PreprocessData()           [MapProcessingEngine.cs:1386]
            │   ├─ D2FlowArea / XS / StorageArea feature tables loaded into memory
            │   ├─ MeshFV2D.PixelRenderingCutoff = 0  (forces full-res render)
            │   ├─ ThreadLocal<RASGeometryMapPoints> created (one per thread)
            │   ├─ mesh cell spatial index + cell areas cached per 2D area
            │   ├─ terrain.Level0StitchTIN()
            │   └─ TiffMetadata built: NoData=-9999, Artist="HEC-RAS", units, rounding
            │
            ├─ [C] Route by MapType + ProfileIndex  [MapProcessingEngine.cs:1324]
            │   ├─ Depth×Vel, DV², EnergyDepth, EnergyElevation, Froude
            │   │   + Max or Min profile → StoreMapBruteForceMaxMin(useMax)
            │   └─ all other types (WSE, Depth, Velocity, …) → StoreMap() [instance]
            │
            └─ [D] StoreMap() [instance]      [MapProcessingEngine.cs:1943]
                │
                ├─ Renderer.GetSimplifiedComputer(cache, profileIndex, mapType, …)
                │
                └─ for each terrain raster tile file i:
                    │
                    ├─ EnsureMapFilename(i)    → output .tif path
                    ├─ FloatTiffReader(terrain raster i) → get width/height/extent
                    ├─ FloatTiffWriter(output.tif, width, height, ComputeStats=true)
                    ├─ TileProcessingEngine.Start()
                    ├─ Parallel.For(0, numTiles) → render each tile
                    │   └─ each tile: sample HDF results + interpolate → write floats
                    └─ TileProcessingEngine.WaitForExit()
                    │
                    ├─ if raster has data:
                    │   │
                    │   ├─ InterpolatedLayer.AddHistogram(tif)
                    │   │   [MapProcessingEngine.cs:2018]
                    │   │
                    │   └─ InterpolatedLayer.AddSRSGeorefAddo(tif, w, h, minX, maxY, cellSize, …)
                    │       [MapProcessingEngine.cs:2031] [InterpolatedLayer.cs:5156]
                    │       │
                    │       ├─ AddProjection(filename)       [InterpolatedLayer.cs:5087]
                    │       │   ├─ SRSFilenameShowStatus()
                    │       │   │   ├─ SRSFilename empty → false → skip
                    │       │   │   └─ file readable → true
                    │       │   └─ if true:
                    │       │       ├─ SharedData.SRSProjection   [SharedData.cs:1291]
                    │       │       │   ├─ SRSIsESRI()  → ext == ".prj"   [SharedData.cs:1685]
                    │       │       │   ├─ SRSIsWKT()   → ext == ".wkt"   [SharedData.cs:1698]
                    │       │       │   ├─ SRSIsProj4() → ext == ".proj4" [SharedData.cs:1711]
                    │       │       │   ├─ SRSIsEPSG()  → ext == ".epsg"  [SharedData.cs:1724]
                    │       │       │   └─ ".tmp" matches none → _projection = null ⚠ Bug §5.1
                    │       │       └─ Metadata.SetProjectionInfo(file, null)
                    │       │           → null.GetSRS() → NullReferenceException ⚠
                    │       │           [Metadata.cs:67]
                    │       │
                    │       ├─ AddGeoreferencing(file, minX, maxY, cellSize)
                    │       │   └─ Metadata.SetLocationInfo() [Metadata.cs:110]
                    │       │       └─ dataset.SetGeoTransform([minX, cellSize, 0, maxY, 0, -cellSize])
                    │       │
                    │       └─ AddOverlays() → gdal_addo (pyramid overviews)
                    │
                    └─ else (no data): delete .tif
                    │
                InterpolatedLayer.CreateVRT(tifs[], vrtFilename)
                    └─ gdalbuildvrt → assembles per-terrain .tif files into final .vrt
```

---

## 4. `.rasmap` XML Role

`StoreMapCommand` does **not** read the `.rasmap` file at all.
The `.rasmap` file is only relevant when using `StoreAllMapsCommand`.

### `.rasmap` serialization round-trip (for reference)

| Direction | Entry point | Code |
|---|---|---|
| Load | `RASMapperCom.LoadRasMap(filename, xmlText)` | [RASMapperCom.cs:1477](../archive/DLLs/RasMapperLib/RasMapperLib/RASMapperCom.cs#L1477) |
| Load | `RASMapper.LoadRASMapFile(doc, filename)` | [RASMapper.cs:12548](../archive/DLLs/RasMapperLib/RasMapperLib/RASMapper.cs#L12548) |
| Load | `RASResultsMap.XMLLoadMetadata(xmlEl)` | [RASResultsMap.cs:7958](../archive/DLLs/RasMapperLib/RasMapperLib/RASResultsMap.cs#L7958) |
| Save | `RASMapperCom.SaveProject()` | [RASMapperCom.cs:1821](../archive/DLLs/RasMapperLib/RasMapperLib/RASMapperCom.cs#L1821) |
| Save | `RASMapper.SaveRASMapFile()` | [RASMapper.cs:12601](../archive/DLLs/RasMapperLib/RasMapperLib/RASMapper.cs#L12601) |
| Save | `RASResultsMap.XMLSaveMetadata(doc, el)` | [RASResultsMap.cs:7870](../archive/DLLs/RasMapperLib/RasMapperLib/RASResultsMap.cs#L7870) |

### Result map XML structure inside `.rasmap`

```xml
<RASMapper>
  <RASProjectionFilename Filename=".\GIS_Data\Projection.prj" />  <!-- required for SRS -->
  ...
  <Results>
    <Layer Name="Plan 03" Type="RASResults" Filename=".\model.p03.hdf">
      <Layer Name="WSE" Type="RASResultsMap">
        <MapParameters
          MapType="elevation"
          ProfileIndex="216"
          ProfileName="03Jan1999 00:00:00"
          OutputMode="StoredCurrentTerrain"
          StoredFilename=".\RasMaps\WSE(03Jan1999 00:00:00).vrt"
          OverwriteOutputFilename="D:\custom\path\output_stem" />
      </Layer>
    </Layer>
  </Results>
</RASMapper>
```

`<MapParameters>` attributes read by
[`RASResultsMap.XMLLoadMetadata`](../archive/DLLs/RasMapperLib/RasMapperLib/RASResultsMap.cs#L7958):

| Attribute | Field | Notes |
|---|---|---|
| `MapType` | `_mapType` | string name, loaded via `MapTypes.LoadFromXML` |
| `ProfileIndex` | `_profileIndex` | integer; `2147483647` = Max, `-2147483648` = Min |
| `ProfileName` | resolved against results | used to verify/correct ProfileIndex |
| `OutputMode` | `_outputMode` | `DynamicSurface`, `StoredCurrentTerrain`, `StoredPolygonSpecifiedDepth`, etc. |
| `StoredFilename` | `SourceFilename` | relative path to existing stored output |
| `OverwriteOutputFilename` | `_overwriteOutputFilename` | custom output base path (no extension) |
| `Terrain` | `_overwriteTerrainLayerName` | override terrain layer name |
| `LayerName` | `_baseLayerName` | display name |
| `OutputBlock` | `_outputBlockName` | e.g. `"Base Output"` |
| `ArrivalDepth` | `_depthThreshold` | float threshold for arrival time maps |

---

## 5. Bugs Found

### 5.1 Projection crash — `.tmp` extension not recognized

**Symptom:** `NullReferenceException` at
[`Metadata.SetProjectionInfo`](../archive/DLLs/Gepspatial_GDALAssist/Geospatial.GDALAssist/Metadata.cs#L67)
after the raster tiles are written (crash at the "Adding Overlays" step).

**Crash stack from `log_custom_output.txt`:**
```
System.NullReferenceException: Object reference not set to an instance of an object.
  at Geospatial.GDALAssist.Metadata.SetProjectionInfo(String Filename, Projection Proj)
  at RasMapperLib.MapProcessingEngine.StoreMap()
  at RasMapperLib.MapProcessingEngine.StoreMap(RASResultsMap map, TerrainLayer terrain, …)
  at RasMapperLib.RASResultsMap.StoreMap(ProgressReporter reporter, Boolean showFinishedMessage)
  at RasMapperLib.Scripting.StoreMapCommand.Execute(ProgressReporter prog)
  at RasProcess.Program.Main(String[] args)
```

**Root cause chain:**

1. [`SetSRSHelper(terrainLayer)`](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/SetSRSHelper.cs#L16)
   calls `Path.GetTempFileName()` → returns a file with `.tmp` extension.
2. Writes ESRI WKT to that file; sets `SharedData.SRSFilename = file.tmp`.
3. Later, [`SharedData.SRSProjection`](../archive/DLLs/RasMapperLib/RasMapperLib/SharedData.cs#L1291)
   tries to parse the file. All four format checks compare **file extension only**:
   - [`SRSIsESRI()`](../archive/DLLs/RasMapperLib/RasMapperLib/SharedData.cs#L1685) → `.prj` only
   - [`SRSIsWKT()`](../archive/DLLs/RasMapperLib/RasMapperLib/SharedData.cs#L1692) → `.wkt` only
   - [`SRSIsProj4()`](../archive/DLLs/RasMapperLib/RasMapperLib/SharedData.cs#L1705) → `.proj4` only
   - [`SRSIsEPSG()`](../archive/DLLs/RasMapperLib/RasMapperLib/SharedData.cs#L1718) → `.epsg` only
4. `.tmp` matches none → `_projection` is never set → `SRSProjection` returns `null`.
5. [`Metadata.SetProjectionInfo(filename, null)`](../archive/DLLs/Gepspatial_GDALAssist/Geospatial.GDALAssist/Metadata.cs#L67)
   calls `null.GetSRS()` → `NullReferenceException`.

**Fix: switch to `StoreAllMapsCommand`** (see §6).
`StoreAllMapsCommand` uses
[`SetSRSHelper(RasMapFilename)`](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/SetSRSHelper.cs#L43)
which reads `<RASProjectionFilename>` from the `.rasmap` file — a real `.prj` file —
recognized correctly by `SRSIsESRI()`.

---

### 5.2 `OverwriteOutputFilename` path joined onto `OutputFolder`

**Location:** [`RASResultsMap.cs:3337`](../archive/DLLs/RasMapperLib/RasMapperLib/RASResultsMap.cs#L3337)

```csharp
// StoredMapBaseFilename getter
return Path.Combine(Results.OutputFolder, _overwriteOutputFilename);
```

Even when `OverwriteOutputFilename` is an absolute path, `Path.Combine` on Windows
will still prepend `OutputFolder` if the value does not begin with a drive letter root
**and** the setter strips the extension but keeps the directory portion.

**Setter behaviour** ([`RASResultsMap.cs:3322`](../archive/DLLs/RasMapperLib/RasMapperLib/RASResultsMap.cs#L3322)):
```csharp
set {
    _overwriteOutputFilename = Path.Combine(
        Path.GetDirectoryName(value),
        Path.GetFileNameWithoutExtension(value));
}
```

`Path.Combine(OutputFolder, absolutePath)` in .NET returns `absolutePath` unchanged
when `absolutePath` is rooted — so a fully absolute `OutputBaseFilename` (e.g.
`D:\custom\out\stem`) **does** work correctly. The issue only appears when a
relative path is supplied.

**Workaround:** always supply a fully-absolute path in `<OutputBaseFilename>`.

---

## 6. Recommended Approach for Custom Output Location

Use `StoreAllMapsCommand` instead of `StoreMapCommand`. It properly resolves the SRS
from the `.rasmap` file and supports `OverwriteOutputFilename` through the standard
`.rasmap` `<MapParameters>` element.

### Step 1 — add the result map to the `.rasmap` file

Under the appropriate `<Layer Type="RASResults">` node, add:

```xml
<Layer Name="WSE" Type="RASResultsMap">
  <MapParameters
    MapType="elevation"
    ProfileIndex="216"
    ProfileName="03Jan1999 00:00:00"
    OutputMode="StoredCurrentTerrain"
    OverwriteOutputFilename="D:\custom\output\wse_stem" />
</Layer>
```

The `.rasmap` file must also have:
```xml
<RASProjectionFilename Filename=".\GIS_Data\Projection.prj" />
```

### Step 2 — run `StoreAllMaps` command

```xml
<Command Type="StoreAllMaps">
  <RasMapFilename>D:\path\to\model.rasmap</RasMapFilename>
  <ResultFilename>D:\path\to\model.p03.hdf</ResultFilename>
</Command>
```

`<ResultFilename>` is optional — if omitted, all result layers in the `.rasmap` are processed.
See [`StoreAllMapsCommand.Execute()`](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreAllMapsCommand.cs#L46).

### SRS resolution in `StoreAllMapsCommand`

```
SetSRSHelper(RasMapFilename)          [SetSRSHelper.cs:43]
  └─ RASMapperCom.GetSRSFromRasmapDoc(doc, rasMapFilename)
      [RASMapperCom.cs:5120]
      └─ doc → <RASMapper> → <RASProjectionFilename Filename="…">
          └─ MakeAbsolute(Filename, rasMapFilename) → full .prj path
          └─ SharedData.SRSFilename = "D:\...\Projection.prj"
                                      ← extension ".prj" ✓ recognized
```

---

## 7. `StoreAllMapsCommand` vs `StoreMapCommand` Comparison

| Feature | `StoreMapCommand` | `StoreAllMapsCommand` |
|---|---|---|
| SRS source | terrain raster (temp `.tmp` file) ⚠ | `.rasmap` `<RASProjectionFilename>` ✓ |
| Custom output | `<OutputBaseFilename>` in XML | `OverwriteOutputFilename` in `.rasmap` |
| Profile selection | `<ProfileName>` in XML | `ProfileIndex`/`ProfileName` in `.rasmap` |
| Multiple maps | one per invocation | all stored maps in one run |
| Reads `.rasmap` | no | yes (required) |
| Key source files | [StoreMapCommand.cs](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreMapCommand.cs) | [StoreAllMapsCommand.cs](../archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreAllMapsCommand.cs) |
