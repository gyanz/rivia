"""Shared helper functions used across rivia subpackages."""

import datetime as dt
import functools
import logging
import re
import time
from pathlib import Path

_MONTHS = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"}
_DATE_RE = re.compile(r"^(\d{2})([A-Za-z]{3})(\d{4})$")
_INTERVAL_RE = re.compile(r"^(\d+(?:\.\d*)?)\s*([A-Za-z]+)$")
_INTERVAL_UNITS: dict[str, dt.timedelta] = {
    "SEC":   dt.timedelta(seconds=1),
    "MIN":   dt.timedelta(minutes=1),
    "HR":    dt.timedelta(hours=1),
    "HOUR":  dt.timedelta(hours=1),
    "DAY":   dt.timedelta(days=1),
    "WEEK":  dt.timedelta(weeks=1),
    "MONTH": dt.timedelta(days=30),
    "YEAR":  dt.timedelta(days=365),
}


def parse_interval(text: str | bytes) -> dt.timedelta:
    """Parse a HEC-RAS interval string and return a :class:`datetime.timedelta`.

    HEC-RAS writes interval strings as a number immediately followed by a unit
    abbreviation, with optional whitespace between them, e.g. ``'20SEC'``,
    ``'5MIN'``, ``'1HR'``, ``'1HOUR'``, ``'1DAY'``, ``'2WEEK'``.

    Supported units (case-insensitive):

    ======  =====================================
    Unit    Timedelta
    ======  =====================================
    SEC     ``timedelta(seconds=n)``
    MIN     ``timedelta(minutes=n)``
    HR      ``timedelta(hours=n)``
    HOUR    ``timedelta(hours=n)``
    DAY     ``timedelta(days=n)``
    WEEK    ``timedelta(weeks=n)``
    MONTH   ``timedelta(days=n*30)`` (approximate)
    YEAR    ``timedelta(days=n*365)`` (approximate)
    ======  =====================================

    Parameters
    ----------
    text:
        Raw interval string from a HEC-RAS HDF attribute or plan file.
        Bytes are decoded as UTF-8 before parsing.

    Returns
    -------
    datetime.timedelta

    Raises
    ------
    ValueError
        If *text* does not match the expected ``<number><unit>`` format or
        the unit is not recognised.
    """
    if isinstance(text, (bytes, bytearray)):
        text = text.decode()
    text = str(text).strip()
    m = _INTERVAL_RE.match(text)
    if not m:
        raise ValueError(
            f"Cannot parse interval {text!r}. "
            "Expected a number followed by a unit, e.g. '5MIN' or '1 HOUR'."
        )
    value = float(m.group(1))
    unit = m.group(2).upper()
    unit_td = next((td for key, td in _INTERVAL_UNITS.items() if unit == key), None)
    if unit_td is None:
        raise ValueError(
            f"Unrecognised interval unit {m.group(2)!r} in {text!r}. "
            f"Supported units: {', '.join(_INTERVAL_UNITS)}."
        )
    return value * unit_td


def check_sim_date(date: str) -> None:
    """Raise ``ValueError`` if *date* is not in ``DDMONYYYY`` format.

    The month abbreviation is case-insensitive (e.g. ``"01jan2020"`` is valid).
    """
    m = _DATE_RE.match(date)
    if not m or m.group(2).upper() not in _MONTHS:
        raise ValueError(
            f"Invalid simulation date {date!r}. "
            "Expected DDMONYYYY (e.g. '01JAN2020')."
        )


def check_sim_time(time_str: str) -> None:
    """Raise ``ValueError`` if *time_str* is not ``HHMM`` or ``HHMMSS`` with HH <= 24."""
    t = time_str.strip()
    if len(t) not in (4, 6) or not t.isdigit():
        raise ValueError(
            f"Invalid simulation time {time_str!r}. Expected HHMM or HHMMSS."
        )
    hh = int(t[:2])
    if hh > 24:
        raise ValueError(
            f"Invalid simulation time {time_str!r}: hours {hh} exceeds 24."
        )

# Custom level for timer output — sits above WARNING (30) so it survives
# benchmark runs that suppress INFO/DEBUG to reduce I/O overhead.
TIMER: int = 35
logging.addLevelName(TIMER, "TIMER")


def timed(level: int = TIMER):
    """Decorator that logs the elapsed wall-clock time of a function call.

    Parameters
    ----------
    level : int
        ``logging`` level constant (e.g. ``logging.INFO``, ``logging.DEBUG``).
        Defaults to ``logging.DEBUG``.

    Examples
    --------
    >>> from rivia.utils import timed
    >>> import logging
    >>> @timed(logging.INFO)
    ... def my_func(): ...
    """

    def decorator(fn: callable) -> callable:
        _log = logging.getLogger(fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            _log.log(
                level,
                "%s completed in %.3fs",
                fn.__qualname__,
                time.perf_counter() - t0,
            )
            return result

        return wrapper

    return decorator


def log_call(level: int = logging.INFO):
    """Decorator that logs when a function is called.

    Parameters
    ----------
    level : int
        ``logging`` level constant (e.g. ``logging.INFO``, ``logging.DEBUG``).
        Defaults to ``logging.INFO``.

    Examples
    --------
    >>> from rivia.utils import log_call
    >>> import logging
    >>> @log_call(logging.INFO)
    ... def my_func(): ...
    """

    def decorator(fn: callable) -> callable:
        _log = logging.getLogger(fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _log.log(level, "%s called", fn.__qualname__)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


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
