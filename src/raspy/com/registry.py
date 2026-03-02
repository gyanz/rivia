import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# -------------------------
# 1) Version -> registry XXX
# -------------------------

_VERSION_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")
_VERSION_XXXX_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?")

def ras_registry_xxx(version: Union[str, int]) -> str:
    """
    Convert a HEC-RAS version into the registry COM ProgID suffix XXX.

    Examples
    --------
    6.4.1  -> "641"   (RAS641.HECRASController)
    6.3    -> "63"   (patch defaults to 0)
    5.0.7  -> "507"
    5.1    -> "51"   (patch defaults to 0)
    510    -> "51"   (interpreted as 5.1.0)
    51     -> "51"   (interpreted as 5.1.0)
    4.1    -> "41"    (4.x uses major+minor)
    4.1.0  -> "41"    (patch ignored for 4.x)
    410    -> "41"    (interpreted as 4.1.0 for 4.x)
    "RAS630" -> "63"
    """
    s = str(version).strip()

    # Allow common prefixes like "RAS630" / "v6.4.1"
    s = re.sub(r"^\s*(ras|v)\s*", "", s, flags=re.IGNORECASE).strip()

    # Case A: dotted input like "6.4.1" or "5.1." etc.
    if "." in s:
        parts = [p for p in s.split(".") if p.strip() != ""]
        if not parts or not all(p.isdigit() for p in parts):
            raise ValueError(f"Unrecognized HEC-RAS version format: {version!r}")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) >= 2 else 0
        patch = int(parts[2]) if len(parts) >= 3 else 0
        return _to_xxx(major, minor, patch)

    # Case B: pure digits like "641", "630", "510", "41", "410", "6"
    if s.isdigit():
        digits = s
        # Single digit: treat as major only (e.g. "6" -> 600, "4" -> 40)
        if len(digits) == 1:
            major = int(digits)
            minor = 0
            patch = 0
            return _to_xxx(major, minor, patch)

        # Two digits: "41" (4.1) or "51" (5.1.0)
        if len(digits) == 2:
            major = int(digits[0])
            minor = int(digits[1])
            patch = 0
            return _to_xxx(major, minor, patch)

        # Three digits: "641" (6.4.1) or "630" (6.3.0) or "507" (5.0.7)
        # Special-case 4.x where people may write "410" meaning "4.1.0" -> "41"
        if len(digits) >= 3:
            major = int(digits[0])
            minor = int(digits[1])
            patch = int(digits[2])
            return _to_xxx(major, minor, patch)

        # 4+ digits: interpret as major + minor + patch by splitting:
        # - assume 1 digit major, 1 digit minor, rest patch (rare, future-proof-ish)
        # major = int(digits[0])
        # minor = int(digits[1])
        # patch = int(digits[2:]) if digits[2:] else 0
        return _to_xxx(major, minor, patch)

    # Case C: Extract first version-like token from the string
    m = _VERSION_RE.search(s)
    if not m:
        raise ValueError(f"Unrecognized HEC-RAS version format: {version!r}")
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    patch = int(m.group(3) or 0)
    return _to_xxx(major, minor, patch)


def _to_xxx(major: int, minor: int, patch: int) -> str:
    """
    Apply HEC-RAS COM ProgID rules:

    """
    if major < 0 or minor < 0 or patch < 0:
        raise ValueError(f"Negative version parts are not valid: {major}.{minor}.{patch}")

    if patch:
        return f"{major}{minor}{patch}"
    return f"{major}{minor}"


def hec_ras_progid(version: Union[str, int], kind: str = "controller") -> str:
    """
    Build the COM ProgID string for the requested HEC-RAS version.

    kind: "controller" or "geometry"
    """
    xxx = ras_registry_xxx(version)
    kind_l = kind.strip().lower()
    if kind_l in ("controller", "hecrascontroller"):
        suffix = "HECRASController"
    elif kind_l in ("geometry", "hecrasgeometry"):
        suffix = "HECRASGeometry"
    elif kind_l in ("flow", "hecrasflow"):
        suffix = "HECRASFlow"
    else:
        raise ValueError("kind must be 'controller' or 'geometry'")
    return f"RAS{xxx}.{suffix}"


# --------------------------------------------
# 2) Enumerate installed HEC-RAS programs (HKCU/HKLM)
# --------------------------------------------

def find_hec_ras_installations() -> List[Dict[str, Any]]:
    """
    Return a list of HEC-RAS installations found in Windows Uninstall registry keys.

    Searches:
      - HKCU\\...\\Uninstall (per-user)
      - HKLM\\...\\Uninstall (system)
      - HKLM\\...\\WOW6432Node\\Uninstall (system 32-bit view)

    Each item includes:
      scope: "user" | "system" | "system_wow6432"
      display_name, display_version, parsed_version, registry_xxx,
      install_location, publisher, uninstall_string, registry_key
    """
    if sys.platform != "win32":
        raise RuntimeError("find_hec_ras_installations() only works on Windows.")

    import winreg  # type: ignore

    uninstall_paths = [
        ("user", winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        ("system", winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        ("system_wow6432", winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]

    results: List[Dict[str, Any]] = []

    for scope, hive, path in uninstall_paths:
        try:
            with winreg.OpenKey(hive, path) as root:
                n_subkeys, _, _ = winreg.QueryInfoKey(root)
                for i in range(n_subkeys):
                    subkey_name = winreg.EnumKey(root, i)
                    subkey_path = f"{path}\\{subkey_name}"
                    try:
                        with winreg.OpenKey(hive, subkey_path) as k:
                            item = _read_uninstall_entry(winreg, k, scope, hive_name=_hive_name(hive), key_path=subkey_path)
                            if item:
                                results.append(item)
                    except OSError:
                        continue
        except OSError:
            continue

    # De-duplicate (same install shows up twice sometimes)
    uniq = {}
    for r in results:
        key = (r.get("display_name"), r.get("display_version"), r.get("install_location"), r.get("scope"))
        uniq[key] = r
    return sorted(uniq.values(), key=lambda d: (d.get("parsed_version") or "", d.get("display_name") or ""))


def _read_uninstall_entry(winreg, k, scope: str, hive_name: str, key_path: str) -> Optional[Dict[str, Any]]:
    def q(name: str) -> Optional[str]:
        try:
            v, _t = winreg.QueryValueEx(k, name)
            return str(v) if v is not None else None
        except OSError:
            return None

    display_name = q("DisplayName") or ""
    if not display_name:
        return None

    # Match typical names: "HEC-RAS 6.4.1", "HEC-RAS River Analysis System", etc.
    if "hec-ras" not in display_name.lower() and "hec ras" not in display_name.lower():
        return None

    display_version = q("DisplayVersion")
    install_location = q("InstallLocation")
    publisher = q("Publisher")
    uninstall_string = q("UninstallString")

    # Try to parse a usable version string:
    # 1) from DisplayVersion
    # 2) else from DisplayName
    parsed_version = _extract_version_token(display_version) or _extract_version_token(display_name)

    registry_xxx = None
    if parsed_version:
        try:
            registry_xxx = ras_registry_xxx(parsed_version)
        except ValueError:
            registry_xxx = None
    
    version_xxxx = _extract_version_token2(display_version) or _extract_version_token2(display_name)

    return {
        "scope": scope,
        "display_name": display_name,
        "display_version": display_version,
        "parsed_version": parsed_version,
        "registry_xxx": registry_xxx,
        "version_xxxx": version_xxxx,
        "install_location": install_location,
        "publisher": publisher,
        "uninstall_string": uninstall_string,
        "registry_key": f"{hive_name}\\{key_path}",
    }


def _extract_version_token(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = _VERSION_RE.search(text)
    if not m:
        return None
    # Keep as dotted version (major.minor.patch?) for readability
    major = m.group(1)
    minor = m.group(2)
    patch = m.group(3)
    if minor is None:
        return major
    if patch is None:
        return f"{major}.{minor}"
    return f"{major}.{minor}.{patch}"

def _extract_version_token2(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = _VERSION_XXXX_RE.search(text)
    if not m:
        return None
    major = m.group(1)
    minor = m.group(2)
    patch = m.group(3)
    beta = m.group(4)
    if minor is None:
        minor = "0"
    if patch is None:
        patch = "0"
    if beta is None:
        beta = "0"
    
    return int(f"{major}{minor}{patch}{beta}")

def _hive_name(hive) -> str:
    # Minimal friendly labels
    import winreg  # type: ignore
    if hive == winreg.HKEY_CURRENT_USER:
        return "HKCU"
    if hive == winreg.HKEY_LOCAL_MACHINE:
        return "HKLM"
    return "HK?"

# ---------------------------------------------------------
# COM registry lookup: ProgID -> (exists, CLSID, server path)
# ---------------------------------------------------------

def _com_progid_info(progid: str) -> Dict[str, Any]:
    """
    Check HKCR\\<ProgID>. If present, resolve CLSID and LocalServer32/InprocServer32.

    Returns dict with:
      exists: bool
      progid: str
      view: "64-bit" | "32-bit" | None     (which registry view we found it in)
      clsid: str | None
      server: str | None                  (LocalServer32 or InprocServer32 default value)
    """
    if sys.platform != "win32":
        raise RuntimeError("com_progid_info() is Windows-only.")

    import winreg  # type: ignore

    def try_open(view_flag: int) -> Optional[Dict[str, Any]]:
        access = winreg.KEY_READ | view_flag
        try:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, progid, 0, access) as k:
                clsid = None
                try:
                    with winreg.OpenKey(k, "CLSID", 0, access) as ck:
                        clsid, _ = winreg.QueryValueEx(ck, "")
                except OSError:
                    pass

                server = None
                if clsid:
                    # Prefer LocalServer32 (out-of-proc COM), else InprocServer32 (DLL)
                    for sub in ("LocalServer32", "InprocServer32"):
                        try:
                            with winreg.OpenKey(
                                winreg.HKEY_CLASSES_ROOT,
                                fr"CLSID\{clsid}\{sub}",
                                0,
                                access,
                            ) as sk:
                                server, _ = winreg.QueryValueEx(sk, "")
                                break
                        except OSError:
                            continue

                return {"exists": True, "progid": progid, "clsid": clsid, "server": server}
        except OSError:
            return None

    # On 64-bit Windows, COM registrations can live in either view.
    found64 = try_open(getattr(winreg, "KEY_WOW64_64KEY", 0))
    if found64:
        found64["view"] = "64-bit"
        return found64

    found32 = try_open(getattr(winreg, "KEY_WOW64_32KEY", 0))
    if found32:
        found32["view"] = "32-bit"
        return found32

    return {"exists": False, "progid": progid, "view": None, "clsid": None, "server": None}


# ---------------------------------------------------------
# Join: installed HEC-RAS (Uninstall keys) -> ProgIDs in HKCR
# (calls your earlier find_hec_ras_installations() if you have it)
# ---------------------------------------------------------
def installed_ras_progids() -> List[Dict[str, Any]]:
    """
    Given a function that returns installed HEC-RAS entries (like find_hec_ras_installations()),
    enrich each entry with the actual registered COM ProgID info for Controller and Geometry.

    Usage:
        installs = installed_ras_progids(find_hec_ras_installations)
    """
    installs = find_hec_ras_installations()
    out: List[Dict[str, Any]] = []

    for inst in installs:
        parsed_version = inst.get("parsed_version") or inst.get("display_version") or inst.get("display_name")
        if not parsed_version:
            out.append(inst)
            continue

        try:
            xxx = ras_registry_xxx(parsed_version)
        except Exception:
            xxx = None

        controller = None
        geometry = None
        flow = None
        if xxx:
            controller_progid = f"RAS{xxx}.HECRASController"
            geometry_progid = f"RAS{xxx}.HECRASGeometry"
            flow_progid = f"RAS{xxx}.HECRASFlow"
            controller = _com_progid_info(controller_progid)
            geometry = _com_progid_info(geometry_progid)
            flow = _com_progid_info(flow_progid)

        enriched = dict(inst)
        enriched["registry_xxx"] = xxx
        enriched["controller"] = controller
        enriched["geometry"] = geometry
        enriched["flow"] = flow
        out.append(enriched)

    return out

# -------------------------
# Quick usage examples
# -------------------------
if __name__ == "__main__":
    for v in ["6.4.1", "6.3", "5.0.7", "5.1", "510", 51, "4.1.0", "410", "RAS630"]:
        xxx = ras_registry_xxx(v)
        print(v, "->", xxx, "|", hec_ras_progid(v, "controller"))

    # List installed HEC-RAS entries from registry (Windows only)
    if sys.platform == "win32":
        installs = find_hec_ras_installations()
        for r in installs:
            print(r["scope"], r["display_name"], r["parsed_version"], r["registry_xxx"], r["install_location"])