/*
 * RasMapperStoreMap.exe
 *
 * Thin wrapper around RasMapperLib.Scripting.StoreAllMapsCommand that initialises
 * SharedData render-mode state before executing the command.  RasProcess.exe never
 * sets render mode (SharedData defaults to basic Sloping/JustFacepoints), so rasters
 * produced by RasProcess.exe always ignore the <RenderMode> settings in the .rasmap
 * file.  This stub fixes that by reading the render-mode arguments and calling the
 * appropriate SharedData setter before delegating to StoreAllMapsCommand.
 *
 * RasMapperLib.dll is loaded dynamically at runtime from the HEC-RAS installation
 * directory so this project has zero compile-time references to HEC-RAS assemblies.
 *
 * Usage:
 *   RasMapperStoreMap.exe
 *       -RasMapFilename=<path>
 *       [-ResultFilename=<path>]
 *       [-RenderMode=sloping|slopingPretty|horizontal]   (default: sloping)
 *       [-UseDepthWeightedFaces=true|false]              (default: false)
 *       [-ReduceShallowToHorizontal=true|false]          (default: true)
 *       [-RasMapperLibDir=<HEC-RAS install dir>]
 *
 * If -RasMapperLibDir is omitted the exe tries:
 *   1. Environment variable HEC_RAS_LIB_DIR
 *   2. Common HEC-RAS installation paths (6.6 down to 6.1)
 *
 * Source references:
 *   archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreAllMapsCommand.cs
 *   archive/DLLs/RasMapperLib/RasMapperLib/SharedData.cs  (~line 1765, ~line 1926)
 *   archive/DLLs/RasMapperLib/RasMapperLib/RASMapper.cs   (~line 12876)
 *   docs/rasprocess_render_mode_limitation.md
 */

using System.Reflection;
using System.Runtime.Loader;

// ── Argument parsing ──────────────────────────────────────────────────────────

string? rasMapFilename = null;
string? resultFilename = null;
string  renderMode               = "sloping";
bool    useDepthWeightedFaces    = false;
bool    reduceShallowToHorizontal = true;
string? libDir = null;

foreach (var arg in args)
{
    if      (Starts(arg, "-RasMapFilename="))           rasMapFilename            = Val(arg);
    else if (Starts(arg, "-ResultFilename="))            resultFilename            = Val(arg);
    else if (Starts(arg, "-RenderMode="))                renderMode                = Val(arg);
    else if (Starts(arg, "-UseDepthWeightedFaces="))     useDepthWeightedFaces     = Bool(arg);
    else if (Starts(arg, "-ReduceShallowToHorizontal=")) reduceShallowToHorizontal = Bool(arg);
    else if (Starts(arg, "-RasMapperLibDir="))           libDir                    = Val(arg);
}

if (rasMapFilename is null)
{
    Console.Error.WriteLine(
        "RasMapperStoreMap: stores HEC-RAS result maps with full render-mode fidelity.\n" +
        "\n" +
        "Usage:\n" +
        "  RasMapperStoreMap.exe -RasMapFilename=<path>\n" +
        "                        [-ResultFilename=<path>]\n" +
        "                        [-RenderMode=sloping|slopingPretty|horizontal]\n" +
        "                        [-UseDepthWeightedFaces=true|false]\n" +
        "                        [-ReduceShallowToHorizontal=true|false]\n" +
        "                        [-RasMapperLibDir=<HEC-RAS install dir>]");
    return 1;
}

// ── Locate RasMapperLib.dll ───────────────────────────────────────────────────

libDir ??= ResolveLibDir();

if (libDir is null)
{
    Console.Error.WriteLine(
        "RasMapperStoreMap: cannot locate RasMapperLib.dll.\n" +
        "Pass -RasMapperLibDir=<dir>, or set the HEC_RAS_LIB_DIR environment variable.");
    return 1;
}

string rasMapperLibPath = Path.Combine(libDir, "RasMapperLib.dll");
if (!File.Exists(rasMapperLibPath))
{
    Console.Error.WriteLine($"RasMapperStoreMap: RasMapperLib.dll not found in '{libDir}'.");
    return 1;
}

// ── Load RasMapperLib + resolve transitive dependencies ──────────────────────

// All HEC-RAS assemblies (Utility.dll, etc.) live in the same directory.
AssemblyLoadContext.Default.Resolving += (ctx, name) =>
{
    var candidate = Path.Combine(libDir, name.Name + ".dll");
    return File.Exists(candidate) ? ctx.LoadFromAssemblyPath(candidate) : null;
};

Assembly asm;
try
{
    asm = AssemblyLoadContext.Default.LoadFromAssemblyPath(rasMapperLibPath);
}
catch (Exception ex)
{
    Console.Error.WriteLine($"RasMapperStoreMap: failed to load RasMapperLib.dll — {ex.Message}");
    return 1;
}

// ── Apply render mode to SharedData ──────────────────────────────────────────
//
// SharedData module-level defaults (used by RasProcess.exe) are:
//   CellRenderMode  = Sloping
//   CellStencilMode = JustFacepoints   ← excludes face centroids
//   UseDepthWeightedFaces = false
//   ShallowBehaviorMode   = ReduceToHorizontal
//
// Only SetSlopingPrettyRenderingMode() sets CellStencilMode = WithFaces, which
// enables face-centroid interpolation and depth-weighted faces.

var sharedDataType = asm.GetType("RasMapperLib.SharedData")
    ?? throw new InvalidOperationException("RasMapperLib.SharedData not found in assembly.");

Console.WriteLine($"RasMapperStoreMap: RenderMode={renderMode} " +
                  $"UseDepthWeightedFaces={useDepthWeightedFaces} " +
                  $"ReduceShallowToHorizontal={reduceShallowToHorizontal}");

switch (renderMode.ToLowerInvariant())
{
    case "horizontal":
        InvokeStatic(sharedDataType, "SetHorizontalRenderingMode");
        break;

    case "sloping":
        InvokeStatic(sharedDataType, "SetSlopingRenderingMode");
        break;

    case "slopingpretty":
        // Signature: SetSlopingPrettyRenderingMode(bool reduceShallow, bool depthWeights)
        // Matches RASMapper.XMLLoad() call order (SharedData.cs ~line 1765).
        InvokeStatic(sharedDataType, "SetSlopingPrettyRenderingMode",
            reduceShallowToHorizontal, useDepthWeightedFaces);
        break;

    default:
        Console.Error.WriteLine(
            $"RasMapperStoreMap: unknown RenderMode '{renderMode}'. " +
            "Use horizontal, sloping, or slopingPretty.");
        return 1;
}

// ── Instantiate and execute StoreAllMapsCommand ───────────────────────────────
//
// Constructor: StoreAllMapsCommand(string rmFilename, string resFilename = "")
// Execute(ProgressReporter prog = null) — null is handled internally via
//   ProgressReporter.None(), so no Utility.dll reference is needed here.

var cmdType = asm.GetType("RasMapperLib.Scripting.StoreAllMapsCommand")
    ?? throw new InvalidOperationException(
           "RasMapperLib.Scripting.StoreAllMapsCommand not found in assembly.");

var ctor = cmdType.GetConstructor([typeof(string), typeof(string)])
    ?? throw new InvalidOperationException(
           "StoreAllMapsCommand(string, string) constructor not found.");

var cmd = ctor.Invoke([rasMapFilename, resultFilename ?? ""]);

var executeMethod = cmdType.GetMethod("Execute")
    ?? throw new InvalidOperationException("StoreAllMapsCommand.Execute() not found.");

try
{
    executeMethod.Invoke(cmd, [null]);
}
catch (TargetInvocationException tie) when (tie.InnerException is not null)
{
    Console.Error.WriteLine($"RasMapperStoreMap: StoreAllMapsCommand failed — {tie.InnerException.Message}");
    return 1;
}

return 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

static bool   Starts(string arg, string prefix) =>
    arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);

static string Val(string arg) =>
    arg[(arg.IndexOf('=') + 1)..];

static bool Bool(string arg) =>
    Val(arg).Equals("true", StringComparison.OrdinalIgnoreCase);

static void InvokeStatic(Type type, string methodName, params object[] methodArgs)
{
    var method = type.GetMethod(methodName, BindingFlags.Public | BindingFlags.Static)
        ?? throw new InvalidOperationException($"{type.Name}.{methodName} not found.");
    method.Invoke(null, methodArgs.Length > 0 ? methodArgs : null);
}

static string? ResolveLibDir()
{
    // 1. Environment variable
    var envDir = Environment.GetEnvironmentVariable("HEC_RAS_LIB_DIR");
    if (envDir is not null && File.Exists(Path.Combine(envDir, "RasMapperLib.dll")))
        return envDir;

    // 2. Common HEC-RAS installation paths, newest first
    string[] candidates =
    [
        @"C:\Program Files (x86)\HEC\HEC-RAS\6.6",
        @"C:\Program Files (x86)\HEC\HEC-RAS\6.5",
        @"C:\Program Files (x86)\HEC\HEC-RAS\6.4.1",
        @"C:\Program Files (x86)\HEC\HEC-RAS\6.3.1",
        @"C:\Program Files (x86)\HEC\HEC-RAS\6.1",
        @"C:\Program Files\HEC\HEC-RAS\6.6",
    ];

    foreach (var dir in candidates)
    {
        if (File.Exists(Path.Combine(dir, "RasMapperLib.dll")))
            return dir;
    }

    return null;
}
