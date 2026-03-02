from __future__ import print_function
import logging
import traceback
import win32com.client
from typing import Union, Any, Optional

from .ras import installed_ras_progid
from ._runtime import Runtime, kill_hecras
from ._geometry import GeometryBase as _GeometryBase
from ._ver400 import Controller as C400, RASEvents as E400
from ._ver500 import Controller as C500, RASEvents as E500
from ._ver503 import Controller as C503, RASEvents as E503


def controller(version:Union[str | int]):
    version_xxxx,info = installed_ras_progid(version)
    print(info)
    geometry_progid = info["geometry"]
    flow_progid = info["flow"]
    controller_progid = info["controller"]

    # Necessary so that only one HEC-RAS process is open at a time
    if not controller_progid is None:
        # extra carefull although controller can't be None as it would have already raised error
        kill_hecras()
        
        _rc = _call_comobject(controller_progid)
        _geom = _call_comobject(geometry_progid)
        _flow = None
        _events = None

        if not flow_progid is None:
            _flow = _call_comobject(flow_progid)
        
        if version_xxxx < 5000:
            _events = _call_comobject(_rc,E400)
            rc = _Controller400(_rc, _geom, _flow, _events, version_xxxx)

        elif version_xxxx <5030:
            _events = _call_comobject(_rc,E500)
            rc = _Controller500(_rc, _geom, _flow, _events, version_xxxx)

        else:
            _events = _call_comobject(_rc,E503)
            rc = _Controller503(_rc, _geom, _flow, _events, version_xxxx)
        
        return rc

def _call_comobject(prog_id, event=None):
    if not event is None:
        com = win32com.client.WithEvents(prog_id, event)
    else:
        com = win32com.client.DispatchEx(prog_id)
    
    return com

class _ControllerBase:
    def __init__(self,rc,geom,flow,events,version_xxxx):
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

    def ras_version(self):
        return self._rasver

    def controller(self):
        return self._rc

    @property
    def exe(self):
        return self._runtime.exe

    def close(self):
        self._runtime.close()

    def __del__(self):
        logging.debug('HEC-RAS Controller destructor called.')
        self.close()

class _Controller400(_ControllerBase,C400,_GeometryBase):
    def __init__(self,rc,geom,flow,events,version_xxxx):
        super().__init__(rc,geom,flow,events,version_xxxx)
        self._runtime = Runtime(self)

class _Controller500(_ControllerBase,C500,_GeometryBase):
    def __init__(self,rc,geom,flow,events,version_xxxx):
        super().__init__(rc,geom,flow,events,version_xxxx)
        self._runtime = Runtime(self)

class _Controller503(_ControllerBase,C503,_GeometryBase):
    def __init__(self,rc,geom,flow,events,version_xxxx):
        super().__init__(rc,geom,flow,events,version_xxxx)
        self._runtime = Runtime(self)

#class _ControllerGeometry(_GeometryBase):
#    def __init__(self, geometry):
#        self._geometry = geometry
