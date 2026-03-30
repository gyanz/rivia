"""Filesystem utilities."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def assert_path_writable(path: str | Path) -> None:
    """Raise ``PermissionError`` early if *path* cannot be written.

    Tests the actual filesystem operations that a write requires:

    - If the file **exists**: attempts a rename to a temporary name and back.
      On Windows this requires ``DELETE`` access on the source — the same
      permission GDAL needs to delete-and-rewrite the file.  GIS applications
      (ArcGIS, QGIS, RASMapper) typically hold files open without
      ``FILE_SHARE_DELETE``, so the rename fails if any of them have the file
      open, even on network/SMB shares where ``CreateFileW`` checks are
      unreliable.

    - If the file **does not exist**: creates and immediately removes a
      zero-byte sentinel in the parent directory to verify write permission.

    Parameters
    ----------
    path:
        Output file path to check.

    Raises
    ------
    PermissionError
        If the path cannot be written, with a message identifying the file and
        suggesting the user close it in any open application.
    """
    p = Path(path).resolve()

    if p.exists():
        tmp = p.with_suffix(p.suffix + ".rivia_write_check")
        try:
            os.rename(p, tmp)
        except OSError as err:
            msg = (
                f"Output file is locked by another application: {p}\n"
                "Close the file (e.g. in ArcGIS, QGIS, or RASMapper) and retry."
            )
            logger.error(msg)
            raise PermissionError(msg) from err
        try:
            os.rename(tmp, p)
        except OSError:
            # Rename-back failed — restore from tmp so we don't lose the file.
            # Re-raise the original lock check error is not applicable here;
            # surface this as an unexpected OS error.
            raise
    else:
        parent = p.parent
        parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_name = tempfile.mkstemp(dir=parent)
            os.close(fd)
            os.unlink(tmp_name)
        except OSError as err:
            msg = (
                f"Cannot write to directory: {parent}\n"
                "Check that you have write permission to the output folder."
            )
            logger.error(msg)
            raise PermissionError(msg) from err
