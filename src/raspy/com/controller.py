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

    def controller(self):
        return self._rc

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

    def __del__(self):
        logging.debug("HEC-RAS Controller destructor called.")
        self.close()


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
