"""Cached lookup of installed HEC-RAS versions and their COM ProgIDs."""

from functools import cache

from .registry import installed_ras_progids, ras_registry_xxx


@cache
def _cached_installs() -> tuple[dict, ...]:
    """Return installed HEC-RAS entries as an immutable tuple (cache-safe)."""
    return tuple(installed_ras_progids())


def installed_ras_versions(descriptive: bool = False) -> list[str]:
    """Return a list of installed HEC-RAS versions.

    Parameters
    ----------
    descriptive:
        If True, return human-readable display names (e.g. "HEC-RAS 6.4.1").
        If False (default), return numeric version codes (e.g. 6410).
    """
    installs = _cached_installs()
    if descriptive:
        return [e["display_name"] for e in installs if e.get("display_name")]
    return [e["version_xxxx"] for e in installs if e.get("version_xxxx")]


def installed_ras_progid(version: str | int) -> tuple[int, dict[str, str | None]]:
    """Return (version_xxxx, progids) for the requested HEC-RAS version.

    Parameters
    ----------
    version:
        Any format accepted by ras_registry_xxx (e.g. "6.4.1", 641, "RAS63").

    Returns
    -------
    version_xxxx:
        Numeric version code (e.g. 6410).
    progids:
        Dict with keys "controller", "geometry", "flow". Each value is the
        COM ProgID string if registered, otherwise None.

    Raises
    ------
    RuntimeError
        If the requested version is not found in the installed HEC-RAS entries.
    """
    xxx = ras_registry_xxx(version)
    entry = next((e for e in _cached_installs() if e.get("registry_xxx") == xxx), None)
    if entry is None:
        raise RuntimeError(f"HEC-RAS {version} is not installed.")

    version_xxxx = int(entry["version_xxxx"])
    progids: dict[str, str | None] = {
        "controller": None,
        "geometry": None,
        "flow": None,
    }
    for key in progids:
        com = entry.get(key)
        if com and com["exists"]:
            progids[key] = com["progid"]

    return version_xxxx, progids


def installed_ras_display_name(version: str | int) -> str | None:
    xxx = ras_registry_xxx(version)
    entry = next((e for e in _cached_installs() if e.get("registry_xxx") == xxx), None)
    if entry is None:
        raise RuntimeError(f"HEC-RAS {version} is not installed.")

    return entry["display_name"]
