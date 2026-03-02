"""Shared helper functions used across raspy subpackages."""

import datetime as dt
from pathlib import Path


def fix_ras_dates(dates: list) -> list[dt.datetime]:
    """Convert HEC-RAS date serial numbers to datetime objects.

    HEC-RAS uses an Excel-style 1900 epoch with a -2 day adjustment.

    Derived from archive/pyras/controllers/hecras/hecrascontroller/ras41.py.
    """
    init = dt.datetime(1900, 1, 1) - dt.timedelta(2)
    return [dt.timedelta(d) + init for d in dates[1:]]


def ensure_dir(filepath: str | Path) -> str:
    """Ensure the parent directory of filepath exists, creating it if needed.

    Returns the resolved absolute path as a string, suitable for passing to
    COM methods that expect a plain string path.

    Derived from archive/pyras/controllers/hecras/hecrascontroller/ras41.py.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
