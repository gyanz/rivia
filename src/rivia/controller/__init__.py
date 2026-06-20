"""COM interface to run and control HEC-RAS."""

import pywintypes

from .controller import HecRasComputeError, connect  # noqa: F401

ComError = pywintypes.com_error

__all__ = ["connect", "HecRasComputeError", "ComError"]
