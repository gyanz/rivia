import logging

import psutil
import pywintypes
import win32com.client

from .ras import installed_ras_progid, installed_ras_display_name
from ._runtime import Runtime, kill_hecras_version
from ._geometry import GeometryBase as _GeometryBase
from ._ver400 import Controller as C400, RASEvents as E400
from ._ver500 import Controller as C500, RASEvents as E500
from ._ver503 import Controller as C503, RASEvents as E503

logger = logging.getLogger("rivia.controller")


class HecRasComputeError(RuntimeError):
    """Raised when a HEC-RAS computation fails or the COM call errors.

    Attributes
    ----------
    messages : tuple[str, ...]
        Messages returned by HEC-RAS at the time of failure. Empty when the
        version does not expose messages or when the error is COM-level.
    com_error : pywintypes.com_error or None
        The underlying COM exception, if the failure was a COM-level error.
        ``None`` when HEC-RAS returned ``success=False`` without a COM error.
    """

    def __init__(
        self,
        message: str,
        messages: tuple[str, ...] = (),
        com_error=None,
    ):
        super().__init__(message)
        self.messages = messages
        self.com_error = com_error


def controller(version: str | int):
    """Create a version-appropriate HEC-RAS controller for the given version.

    Closes any already-running instance of the requested HEC-RAS version before
    launching a new one, leaving other installed versions untouched. Selects the
    correct controller class based on the resolved version number.

    Parameters
    ----------
    version : str or int
        HEC-RAS version to connect to. Accepts a version string (e.g. ``"6.3"``,
        ``"5.0.3"``) or an integer version code (e.g. ``6030``). Must match an
        installed HEC-RAS entry in the Windows registry.

    Returns
    -------
    _Controller400 or _Controller500 or _Controller503
        A controller instance connected to the requested HEC-RAS version. The
        exact type depends on the resolved version number: ``_Controller400`` for
        versions below 5000, ``_Controller500`` for 5000-5029, and
        ``_Controller503`` for 5030 and above.

    Raises
    ------
    RuntimeError
        If the requested HEC-RAS version is not found in the Windows registry.
    """
    version_xxxx, info = installed_ras_progid(version)
    geometry_progid = info["geometry"]
    flow_progid = info["flow"]
    controller_progid = info["controller"]

    # Close any existing instance of this specific HEC-RAS version before
    # opening a new one, leaving other versions or unrelated sessions untouched.
    if controller_progid is not None:
        kill_hecras_version(installed_ras_display_name(version_xxxx))
        _rc = _dispatch(controller_progid)
        _geom = _dispatch(geometry_progid)
        _flow = _dispatch(flow_progid) if flow_progid is not None else None
        _events = None

        if version_xxxx < 5000:
            _events = _bind_events(_rc, E400)
            rc = _Controller400(_rc, _geom, _flow, _events, version_xxxx)

        elif version_xxxx < 5030:
            _events = _bind_events(_rc, E500)
            rc = _Controller500(_rc, _geom, _flow, _events, version_xxxx)

        else:
            _events = _bind_events(_rc, E503)
            rc = _Controller503(_rc, _geom, _flow, _events, version_xxxx)

        return rc


def _dispatch(prog_id: str):
    return win32com.client.DispatchEx(prog_id)

def _dispatch2(prog_id: str):
    return win32com.client.Dispatch(prog_id)

def _bind_events(com_obj, event_class):
    return win32com.client.WithEvents(com_obj, event_class)


class _ControllerBase:
    def __init__(self, rc, geom, flow, events, version_xxxx):
        self._rc = rc
        self._geometry = geom
        self._flow = flow
        self._events = events
        self._rasver = version_xxxx

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def pause(self, time):
        self._runtime.pause(time)

    def runtime(self):
        return self._runtime

    def ras_version(self, descriptive=False):
        if descriptive:
            return self.HECRASVersion()
        return self._rasver

    @property
    def exe(self):
        return self._runtime.exe

    @property
    def is_alive(self) -> bool:
        """Return True if the HEC-RAS process and COM server are still responsive."""
        pid = self._runtime.parent_pid
        if pid is None:
            return False
        try:
            proc = psutil.Process(pid)
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                return False
        except psutil.NoSuchProcess:
            return False
        try:
            _ = self._rc.HECRASVersion()
            return True
        except pywintypes.com_error:
            return False

    def close(self):
        if self.is_alive:
            self._runtime.close()

    # ------------------------------------------------------------------
    # Window visibility
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Display the main HEC-RAS window.

        Works on all supported HEC-RAS versions.
        """
        self._rc.ShowRas()

    def hide(self) -> None:
        """Hide the main HEC-RAS window.

        Notes
        -----
        HEC-RAS 4.x does not expose a window-hide COM method. Calling this on
        a version-4 controller logs a warning and does nothing. For version 5.0
        and above, ``QuitRas()`` is used, which hides (minimises) the window.
        """
        if self._rasver < 5000:
            logger.warning(
                "show/hide: HEC-RAS %d does not support window hide via COM; "
                "ignoring hide() call.",
                self._rasver,
            )
            return
        self._rc.QuitRas()

    # ------------------------------------------------------------------
    # Project lifecycle (version-gated)
    # ------------------------------------------------------------------

    def Project_Close(self) -> None:
        """Close the currently open HEC-RAS project.

        Raises
        ------
        NotImplementedError
            If the connected HEC-RAS version is older than 5.0.3 (version code
            5030), which does not expose ``Project_Close`` via COM.
        """
        if self._rasver < 5030:
            raise NotImplementedError(
                f"Project_Close() requires HEC-RAS 5.0.3+ (version code ≥ 5030); "
                f"connected version is {self._rasver}."
            )
        self._rc.Project_Close()

    # ------------------------------------------------------------------
    # Compute helpers (version-gated)
    # ------------------------------------------------------------------

    def Compute_Complete(self) -> bool:
        """Return ``True`` once an asynchronous computation has finished.

        Notes
        -----
        Only meaningful when ``BlockingMode=False`` was passed to
        :meth:`Compute_CurrentPlan`. Poll this in a loop while waiting.

        Raises
        ------
        NotImplementedError
            For HEC-RAS versions below 5.0, which do not expose this method.
        """
        if self._rasver < 5000:
            raise NotImplementedError(
                f"Compute_Complete() requires HEC-RAS 5.0+ (version code ≥ 5000); "
                f"connected version is {self._rasver}."
            )
        return self._rc.Compute_Complete()

    def Compute_StartedFromController(self) -> bool:
        """Return ``True`` if the current computation was started via COM.

        Notes
        -----
        Requires ``BlockingMode=False`` in :meth:`Compute_CurrentPlan` to be meaningful.

        Raises
        ------
        NotImplementedError
            For HEC-RAS versions below 5.0.
        """
        if self._rasver < 5000:
            raise NotImplementedError(
                f"Compute_StartedFromController() requires HEC-RAS 5.0+ "
                f"(version code ≥ 5000); connected version is {self._rasver}."
            )
        return self._rc.Compute_StartedFromController

    def Compute_Cancel(self) -> None:
        """Cancel a running computation.

        Notes
        -----
        Only available in HEC-RAS 4.x. This method was removed in version 5.0.
        Use :meth:`compute` with ``blocking=True`` (default) to avoid needing
        cancellation on modern versions.

        Raises
        ------
        NotImplementedError
            For HEC-RAS 5.0 and above, where this COM method no longer exists.
        """
        if self._rasver >= 5000:
            raise NotImplementedError(
                f"Compute_Cancel() was removed in HEC-RAS 5.0; "
                f"connected version is {self._rasver}."
            )
        self._rc.Compute_Cancel()

    def Compute_IsStillComputing(self) -> bool:
        """Return ``True`` if a computation is still running.

        Notes
        -----
        Only available in HEC-RAS 4.x. Removed in version 5.0. On 5.0+ use
        :meth:`Compute_Complete` instead.

        Raises
        ------
        NotImplementedError
            For HEC-RAS 5.0 and above.
        """
        if self._rasver >= 5000:
            raise NotImplementedError(
                f"Compute_IsStillComputing() was removed in HEC-RAS 5.0; "
                f"use Compute_Complete() instead. "
                f"Connected version is {self._rasver}."
            )
        return self._rc.Compute_IsStillComputing()

    def Compute_CurrentPlan(  # noqa: N802
        self, BlockingMode: bool = True
    ) -> tuple[bool, tuple[str, ...]]:
        """Compute the current plan, compatible with all HEC-RAS versions.

        Parameters
        ----------
        BlockingMode : bool, optional
            If True (default), block until computation completes. If False,
            return immediately while HEC-RAS computes in the background.
            Ignored for HEC-RAS versions below 5.0 (always blocking).

        Returns
        -------
        success : bool
            True if the computation completed successfully.
        messages : tuple[str, ...]
            Messages returned by HEC-RAS during computation. Empty for
            versions below 5.0.3.
        """
        rc = self._rc
        version = self.ras_version()

        try:
            if version < 5000:
                if not BlockingMode:
                    logger.debug(
                        "compute: blocking unavailable in HEC-RAS version %d", version
                    )
                res = rc.Compute_CurrentPlan(None, None)
                success = res[0]
                logger.debug("compute: version %d returns no messages", version)
                if not success:
                    raise HecRasComputeError("HEC-RAS computation failed.")
                return success, ()

            elif version < 5030:
                res = rc.Compute_CurrentPlan(None, None, BlockingMode)
                success = res[0]
                logger.debug("compute: version %d returns no messages", version)
                if not success:
                    raise HecRasComputeError("HEC-RAS computation failed.")
                return success, ()

            else:
                res = rc.Compute_CurrentPlan(None, None, int(BlockingMode))
                success = res[0]
                # res layout for v5.0.3+: (status, nmsg, Msg, BlockingMode)
                raw = res[2]
                if raw is None:
                    messages = ()
                elif isinstance(raw, (list, tuple)):
                    messages = raw
                else:
                    messages = (str(raw),) if raw else ()
                if not success:
                    detail = "; ".join(messages) if messages else "no details available"
                    raise HecRasComputeError(
                        f"HEC-RAS computation failed: {detail}",
                        messages=messages,
                    )
                return success, messages

        except HecRasComputeError:
            raise
        except pywintypes.com_error as e:
            raise HecRasComputeError(
                f"COM error during HEC-RAS computation: {e}",
                com_error=e,
            ) from e

    def __del__(self):
        try:
            logger.debug("HEC-RAS Controller destructor called.")
        except Exception:
            pass
        try:
            self.close()
        except Exception:
            pass


class _Controller400(_ControllerBase, C400, _GeometryBase):
    def __init__(self, rc, geom, flow, events, version_xxxx):
        super().__init__(rc, geom, flow, events, version_xxxx)
        self._runtime = Runtime(self, installed_ras_display_name(version_xxxx))


class _Controller500(_ControllerBase, C500, _GeometryBase):
    def __init__(self, rc, geom, flow, events, version_xxxx):
        super().__init__(rc, geom, flow, events, version_xxxx)
        self._runtime = Runtime(self, installed_ras_display_name(version_xxxx))


class _Controller503(_ControllerBase, C503, _GeometryBase):
    def __init__(self, rc, geom, flow, events, version_xxxx):
        super().__init__(rc, geom, flow, events, version_xxxx)
        self._runtime = Runtime(self, installed_ras_display_name(version_xxxx))


# class _ControllerGeometry(_GeometryBase):
#    def __init__(self, geometry):
#        self._geometry = geometry
