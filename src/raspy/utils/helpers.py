"""Shared helper functions used across raspy subpackages."""

import datetime as dt
import functools
import logging
import time
from pathlib import Path


def timed(level: int = logging.DEBUG):
    """Decorator that logs the elapsed wall-clock time of a function call.

    Parameters
    ----------
    level : int
        ``logging`` level constant (e.g. ``logging.INFO``, ``logging.DEBUG``).
        Defaults to ``logging.DEBUG``.

    Examples
    --------
    >>> from raspy.utils import timed
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
    >>> from raspy.utils import log_call
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
